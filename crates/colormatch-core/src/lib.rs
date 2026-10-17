//! `colormatch-core` — the pure algorithm core for vapoursynth-colormatch.
//!
//! This crate has **no** VapourSynth dependency, so it can be unit-tested, used from the CLI,
//! and reused by future backends (AviSynth+, GPU) without touching the plugin layer.
//!
//! # Model representation
//!
//! The colour transform is a composition of `K` *slices*. Each slice projects RGB onto three
//! orthonormal axes, applies an independent monotone 1-D transfer function per axis, and rotates
//! the correction back into RGB:
//!
//! ```text
//! for k in 0..K:
//!     u = R_k · rgb
//!     y = T_k(u)                  // componentwise, monotone
//!     rgb += R_kᵀ · (y − u)
//! ```
//!
//! This is the fixed point iteration that DrDre's ColorMatch performs, and it is exactly a
//! sequential sliced-Wasserstein gradient flow with unit step size.
//!
//! Each 1-D transfer function is stored as a **uniform-grid lookup table** so that evaluation is
//! `idx = (v − lo) · scale; i = idx as usize; lerp(tab[i], tab[i+1], idx − i)` — no binary search.
//! Table gathers dominate the cost rather than the arithmetic, so keeping the table small and hot in
//! cache is the single most important performance property of the whole pipeline.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod apply;
pub mod backend;
pub mod build;
pub mod color;
pub mod lut;
pub mod model;

pub use apply::{ApplyOptions, Gamut, apply_native};
pub use backend::{BackendKind, BackendPreference};
pub use build::{AxesScheme, BuildOptions, build_model};
pub use color::SpaceInfo;
pub use model::{Axis, FORMAT_VERSION, MAGIC, Model, ModelError, ModelStats, Slice, TABLE_NODES};

/// The plugin version, kept in sync with the Cargo workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// VapourSynth plugin namespace (`core.cm.*`).
pub const PLUGIN_NAMESPACE: &str = "cm";

/// VapourSynth plugin identifier.
pub const PLUGIN_IDENTIFIER: &str = "dev.miralia.colormatch";

/// Human-readable plugin name, as VapourSynth reports it.
pub const PLUGIN_NAME: &str = "ColorMatch";

/// Plugin version reported to VapourSynth, as `major << 16 | minor`.
///
/// Derived from [`VERSION`] rather than hard-coded, so the version VapourSynth shows, the version the
/// crate reports, and the version in the wheel cannot disagree. VapourSynth's plugin version has only
/// a major and a minor field, so the patch level is not representable here; [`VERSION`] is
/// authoritative and is what `cm.Devices` reports.
pub const PLUGIN_VERSION: i32 = {
    let v = env!("CARGO_PKG_VERSION").as_bytes();
    // "0.2.0" -> 0 << 16 | 2
    let major = (v[0] - b'0') as i32;
    let minor = (v[2] - b'0') as i32;
    (major << 16) | minor
};

/// Frame property carrying the serialized model on a model clip.
pub const PROP_MODEL: &str = "_CM_Model";

/// Frame property carrying model statistics (see [`ModelStats::as_props`]).
pub const PROP_STATS: &str = "_CM_Stats";

/// Frame property reporting the backend that actually ran.
pub const PROP_BACKEND: &str = "_CM_Backend";

/// Frame property reporting the apply method that actually ran.
pub const PROP_METHOD: &str = "_CM_Method";
