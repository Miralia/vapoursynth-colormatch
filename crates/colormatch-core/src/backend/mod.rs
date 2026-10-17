//! Backend abstraction.
//!
//! Every apply method goes through a backend so that adding GPU support does not change the
//! plugin, the wrapper, or the model format. Backends are selected explicitly or by
//! [`BackendKind::autoselect`], and the choice that actually ran is reported to the caller so
//! that a silent fallback can never happen.

use crate::apply::{self, ApplyOptions, apply_native};
use crate::model::Model;

/// Which backend implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Pick the fastest backend that is actually available on this machine.
    #[default]
    Auto,
    /// Portable scalar code. Always available; the reference implementation.
    Scalar,
    /// CPU SIMD: eight pixels per iteration via `wide` (NEON on aarch64, AVX/SSE on x86_64).
    ///
    /// Bit-identical to [`BackendPreference::Scalar`], so the two are interchangeable and the choice
    /// is purely about speed. See [`crate::apply::apply_native_wide`].
    Simd,
    /// GPU compute.
    Gpu,
    /// Bake the model to a 3D LUT and sample it. Cost is independent of the slice count.
    Lut,
}

impl BackendPreference {
    /// Parses the user-facing string form, matching `vs_colorfix`'s lowercase aliases.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "scalar" => Some(Self::Scalar),
            "simd" => Some(Self::Simd),
            "gpu" | "wgpu" => Some(Self::Gpu),
            "lut" => Some(Self::Lut),
            _ => None,
        }
    }
}

/// The backend that actually executed an apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendKind {
    /// Portable scalar path.
    Scalar,
    /// CPU SIMD path.
    Simd,
    /// GPU path, with the adapter it landed on.
    Gpu {
        /// Human-readable adapter name.
        name: String,
        /// Graphics API used (`metal`, `vulkan`, `dx12`).
        api: String,
    },
    /// 3D LUT path with the LUT edge length.
    Lut {
        /// Number of grid nodes per axis (17, 33 or 65).
        size: u32,
    },
}

impl BackendKind {
    /// Short machine-readable label, written to the `_CM_Backend` frame property.
    pub fn label(&self) -> String {
        match self {
            Self::Scalar => "scalar".to_string(),
            Self::Simd => "simd".to_string(),
            Self::Gpu { name, api } => format!("gpu:{api}:{name}"),
            Self::Lut { size } => format!("lut:{size}"),
        }
    }
}

/// Describes a GPU adapter that a backend could use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    /// Human-readable adapter name.
    pub name: String,
    /// Graphics API (`metal`, `vulkan`, `dx12`, `gl`, …).
    pub api: String,
    /// `discrete`, `integrated`, `virtual` or `cpu`.
    pub device_type: String,
    /// Vendor id.
    pub vendor: u32,
    /// Device id.
    pub device: u32,
}

/// Enumerates the GPU adapters visible to the GPU backend.
///
/// Returns an empty list when the GPU backend is not compiled in or no adapter is usable. The
/// plugin exposes this through `cm.Devices` so the Python wrapper can print a diagnostic table
/// and make an informed automatic choice — the same information `vs-jetpack` obtains from
/// OpenCL, but sourced from the backend that will actually run.
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    #[cfg(feature = "gpu")]
    {
        crate::backend::gpu::enumerate_adapters()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

/// Whether the GPU **apply** path can run on adapter `device_id`.
///
/// Adapter enumeration is not enough to answer this, and neither is the compile-time feature: a
/// build with `gpu` enabled on a machine with no usable adapter must not advertise a GPU backend, and
/// neither must a caller asking for adapter 3 on a one-GPU machine. This creates the device once
/// (cached, per index) and reports whether that worked, so the answer is real.
///
/// The plugin's contract is that the reported backend is the one that ran, so `select` will not pick
/// GPU while this returns `false`, and the frame property therefore cannot claim a GPU that did not
/// do the work.
pub fn gpu_apply_available(device_id: u32) -> bool {
    #[cfg(feature = "gpu")]
    {
        crate::backend::gpu::available(device_id)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = device_id;
        false
    }
}

/// Chooses the backend that will actually be used.
///
/// `preference` is honoured when possible; otherwise the fastest available backend is used. The
/// returned [`BackendKind`] is what the caller must report — never assume the request was granted.
pub fn select(preference: BackendPreference, slices: usize) -> BackendKind {
    select_for(preference, slices, 0)
}

/// Chooses the backend, considering a specific GPU adapter index.
///
/// `device_id` indexes [`enumerate_adapters`], so a caller that listed the adapters can pick one.
/// An index with no adapter behind it is *not* silently replaced by adapter 0: the GPU backend is
/// reported only when that adapter is the one that will run.
pub fn select_for(preference: BackendPreference, slices: usize, device_id: u32) -> BackendKind {
    let gpu = enumerate_adapters();
    let chosen = gpu.get(device_id as usize);
    match preference {
        BackendPreference::Scalar => BackendKind::Scalar,
        BackendPreference::Simd => BackendKind::Simd,
        BackendPreference::Lut => BackendKind::Lut { size: 33 },
        BackendPreference::Gpu => match (gpu_apply_available(device_id), chosen) {
            (true, Some(a)) => BackendKind::Gpu {
                name: a.name.clone(),
                api: a.api.clone(),
            },
            _ => BackendKind::Simd,
        },
        BackendPreference::Auto => {
            // A GPU is only worth the upload/download round trip once the work per pixel is large
            // enough; below that the SIMD path wins outright.
            match (gpu_apply_available(device_id), chosen) {
                (true, Some(a)) if slices >= 8 => BackendKind::Gpu {
                    name: a.name.clone(),
                    api: a.api.clone(),
                },
                _ => BackendKind::Simd,
            }
        }
    }
}

/// Runs an apply with the scalar backend.
///
/// This is the golden reference: every other backend must agree with it. The wide path is bit-exact
/// against it (see `apply::wide_tests`), and the GPU is within 1 LSB.
pub fn apply_scalar(model: &Model, planes: [&mut [f32]; 3], options: &ApplyOptions) {
    apply_native(model, planes, options);
}

/// Runs an apply with the wide (portable SIMD) backend.
///
/// Bit-identical to [`apply_scalar`] by construction and by test, so the choice between them is
/// purely about speed. Falls back to the scalar path on a target `wide` cannot vectorise, which no
/// platform this plugin ships to hits today.
pub fn apply_simd(model: &Model, planes: [&mut [f32]; 3], options: &ApplyOptions) {
    if apply::wide_available() {
        apply::apply_native_wide(model, planes, options);
    } else {
        apply_native(model, planes, options);
    }
}

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_parsing_matches_vs_colorfix_aliases() {
        assert_eq!(
            BackendPreference::parse("auto"),
            Some(BackendPreference::Auto)
        );
        assert_eq!(
            BackendPreference::parse("CPU"),
            Some(BackendPreference::Scalar)
        );
        assert_eq!(
            BackendPreference::parse(" cpu "),
            Some(BackendPreference::Scalar)
        );
        assert_eq!(
            BackendPreference::parse("simd"),
            Some(BackendPreference::Simd)
        );
        assert_eq!(
            BackendPreference::parse("gpu"),
            Some(BackendPreference::Gpu)
        );
        assert_eq!(
            BackendPreference::parse("wgpu"),
            Some(BackendPreference::Gpu)
        );
        assert_eq!(
            BackendPreference::parse("lut"),
            Some(BackendPreference::Lut)
        );
        assert_eq!(BackendPreference::parse("tensorrt"), None);
    }

    #[test]
    fn explicit_cpu_preferences_are_never_upgraded() {
        assert_eq!(select(BackendPreference::Scalar, 64), BackendKind::Scalar);
        assert_eq!(select(BackendPreference::Simd, 64), BackendKind::Simd);
        assert_eq!(
            select(BackendPreference::Lut, 64),
            BackendKind::Lut { size: 33 }
        );
    }

    #[test]
    fn gpu_is_never_reported_while_the_apply_path_is_missing() {
        // The reported backend must be the one that ran. Until the WGSL pipelines land, a GPU
        // request has to degrade to SIMD *and say so*, even when adapters are present.
        if !gpu_apply_available(0) {
            assert_eq!(select(BackendPreference::Gpu, 32), BackendKind::Simd);
            assert_eq!(select(BackendPreference::Auto, 32), BackendKind::Simd);
        }
    }

    #[test]
    fn auto_keeps_small_workloads_on_cpu() {
        assert_eq!(select(BackendPreference::Auto, 1), BackendKind::Simd);
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(BackendKind::Scalar.label(), "scalar");
        assert_eq!(BackendKind::Simd.label(), "simd");
        assert_eq!(BackendKind::Lut { size: 65 }.label(), "lut:65");
        assert_eq!(
            BackendKind::Gpu {
                name: "Test Adapter".into(),
                api: "metal".into()
            }
            .label(),
            "gpu:metal:Test Adapter"
        );
    }
}
