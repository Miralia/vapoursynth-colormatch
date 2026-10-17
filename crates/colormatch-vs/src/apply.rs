//! `cm.Apply` — applies a model to a clip.
//!
//! Contract: the output is bit-identical to the input in width, height, format, framerate and
//! length. The output `VSVideoInfo` is a copy of the source's, and nothing in this filter ever
//! resizes or changes a format.
//!
//! Threading: the filter is `fmParallel` and stateless. It requests frame `n` from the source
//! (strictly spatial) and frame `0` from the model clip when one is supplied. Because a model clip
//! is a single 1×1 frame, that is a constant, cache-resident request — the design deliberately
//! avoids requesting other source frames so that `fmParallel` is correct without needing
//! `fmParallelRequests`.

use std::sync::OnceLock;

use anyhow::{Context, Error, Result, bail};
use colormatch_core::apply::{ApplyOptions, Gamut, apply_native};
use colormatch_core::backend::{BackendKind, BackendPreference, select_for};
use colormatch_core::lut::{SUPPORTED_SIZES, apply_lut_tetrahedral, bake_lut};
use colormatch_core::model::Model;
use colormatch_core::{PROP_BACKEND, PROP_METHOD, PROP_MODEL};
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::*;

use crate::frame_io::{ChromaMode, ClipFormat, frame_to_rgb, rgb_to_frame};

/// Which evaluation strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Evaluate the slice chain directly. Exact; cost scales with the slice count.
    Native,
    /// Bake to a 3D LUT and sample it with tetrahedral interpolation. Cost is independent of the
    /// slice count, at the price of interpolation error.
    Lut,
}

impl Method {
    fn parse(s: &str, func: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "native" => Ok(Self::Native),
            "lut" => Ok(Self::Lut),
            other => bail!("{func}: `method` must be \"native\" or \"lut\", got {other:?}"),
        }
    }
}

/// Everything resolved once the model is known.
struct Resolved {
    model: Model,
    backend: BackendKind,
    /// Baked LUT, present only when the LUT path is in use.
    lut: Option<Vec<[f32; 3]>>,
}

/// The `cm.Apply` filter.
pub struct ApplyFilter<'core> {
    source: Node<'core>,
    model_node: Option<Node<'core>>,
    model: Option<Model>,
    fmt: ClipFormat,
    planes: Option<Vec<usize>>,
    options: ApplyOptions,
    preference: BackendPreference,
    method: Method,
    lut_size: u32,
    chroma: ChromaMode,
    device_id: u32,
    resolved: OnceLock<Resolved>,
}

impl<'core> ApplyFilter<'core> {
    /// Creates the filter from a source clip and either an in-memory model or a model clip.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Node<'core>,
        model_node: Option<Node<'core>>,
        model: Option<Model>,
        fmt: ClipFormat,
        planes: Option<Vec<usize>>,
        options: ApplyOptions,
        preference: BackendPreference,
        method: Method,
        lut_size: u32,
        chroma: ChromaMode,
        device_id: u32,
    ) -> Self {
        Self {
            source,
            model_node,
            model,
            fmt,
            planes,
            options,
            preference,
            method,
            lut_size,
            chroma,
            device_id,
            resolved: OnceLock::new(),
        }
    }

    /// Resolves the model, the backend and (if requested) the baked LUT exactly once.
    fn resolve(&self, func: &str) -> Result<&Resolved> {
        if let Some(r) = self.resolved.get() {
            return Ok(r);
        }
        let model = match (&self.model, &self.model_node) {
            (Some(m), _) => m.clone(),
            (None, Some(node)) => {
                let frame = node.get_frame(0).map_err(|e| {
                    anyhow::anyhow!("{func}: could not read the model clip's frame 0: {e}")
                })?;
                let props = frame.props();
                let bytes = props.get_data(PROP_MODEL).map_err(|_| {
                    anyhow::anyhow!(
                        "{func}: the model clip has no `{PROP_MODEL}` property; pass the result of \
                         `cm.BuildModel` or a `model_file`"
                    )
                })?;
                Model::from_bytes(bytes)
                    .map_err(|e| anyhow::anyhow!("{func}: invalid model in the model clip: {e}"))?
            }
            (None, None) => bail!("{func}: exactly one of `model` or `model_file` is required"),
        };

        let slices = model.len();
        let backend = select_for(self.preference, slices, self.device_id);
        let lut = match (self.method, &backend) {
            (Method::Lut, _) | (_, BackendKind::Lut { .. }) => Some(
                bake_lut(&model, self.lut_size)
                    .with_context(|| format!("{func}: baking a {}-node LUT", self.lut_size))?,
            ),
            _ => None,
        };

        let resolved = Resolved {
            model,
            backend,
            lut,
        };
        // `set` can only fail if another thread won the race, in which case its value is equally
        // valid; either way the `get` below succeeds.
        let _ = self.resolved.set(resolved);
        Ok(self.resolved.get().expect("just set"))
    }
}

impl<'core> Filter<'core> for ApplyFilter<'core> {
    fn video_info(
        &self,
        _api: API,
        _core: CoreRef<'core>,
    ) -> Vec<vapoursynth::video_info::VideoInfo<'core>> {
        // The output is the input, untouched in shape and format.
        vec![self.source.info()]
    }

    fn get_frame_initial(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        self.source.request_frame_filter(context, n);
        if let Some(m) = &self.model_node {
            // A model clip is a single 1x1 frame; requesting frame 0 is constant work.
            m.request_frame_filter(context, 0);
        }
        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        const FUNC: &str = "cm.Apply";

        let resolved = self.resolve(FUNC)?;
        if resolved.model.is_empty() {
            bail!("{FUNC}: the model is empty (no slices); nothing to apply");
        }

        let frame = self
            .source
            .get_frame_filter(context, n)
            .ok_or_else(|| anyhow::anyhow!("{FUNC}: could not get source frame {n}"))?;

        let mut out = FrameRefMut::copy_of(core, &frame);

        // Which output planes the transform is allowed to touch. Planes outside this set keep the
        // copied source data.
        let all_planes: Vec<usize> = (0..self.fmt.planes).collect();
        let planes = self.planes.as_ref().unwrap_or(&all_planes);

        // The fused GPU route, when it applies: a GPU backend with no baked LUT, every plane
        // selected, and no chroma subsampling. It skips the host-side `f32` planes entirely.
        // Everything else takes the planar route below, which is the reference and always works.
        //
        // `ClipFormat` carries the chroma subsampling as shift counts; `native_layout` converts them
        // to the factors the shader expects. Passing the shift straight through made every pixel look
        // co-sited for 4:2:0, which wrote the chroma planes out of bounds into a constant.
        #[cfg(feature = "gpu")]
        if resolved.lut.is_none()
            && matches!(resolved.backend, BackendKind::Gpu { .. })
            && planes.len() == self.fmt.planes
            && self.chroma == ChromaMode::Nearest
            && crate::frame_io::native_layout(&self.fmt).is_some()
        {
            crate::frame_io::apply_native_gpu(
                &frame,
                &mut out,
                &self.fmt,
                &resolved.model,
                &self.options,
                self.device_id,
            )
            .map_err(|e| anyhow::anyhow!("{FUNC}: {e}"))?;
            out.props_mut()
                .set_data(PROP_BACKEND, resolved.backend.label().as_bytes())?;
            out.props_mut().set_data(PROP_METHOD, b"native")?;
            return Ok(out.into());
        }

        let [mut r, mut g, mut b] = frame_to_rgb(&frame, &self.fmt)?;

        match &resolved.lut {
            Some(lut) => {
                apply_lut_tetrahedral(lut, self.lut_size, [&mut r, &mut g, &mut b], &self.options)
                    .map_err(|e| anyhow::anyhow!("{FUNC}: {e}"))?
            }
            None => match resolved.backend {
                // `select` only returns a GPU backend when `gpu_apply_available()` is true, which
                // creates the device. A failure here is therefore a real fault rather than a missing
                // capability, and it must not be papered over with a silent CPU fallback: the frame
                // would then claim a GPU backend that did not do the work, which is exactly the lie
                // this plugin refuses to tell.
                #[cfg(feature = "gpu")]
                BackendKind::Gpu { .. } => colormatch_core::backend::gpu::apply_gpu(
                    &resolved.model,
                    [&mut r, &mut g, &mut b],
                    &self.options,
                    self.device_id,
                )
                .map_err(|e| anyhow::anyhow!("{FUNC}: {e}"))?,
                BackendKind::Simd => colormatch_core::backend::apply_simd(
                    &resolved.model,
                    [&mut r, &mut g, &mut b],
                    &self.options,
                ),
                _ => apply_native(&resolved.model, [&mut r, &mut g, &mut b], &self.options),
            },
        }

        if planes.len() == self.fmt.planes {
            rgb_to_frame(&mut out, &self.fmt, [&r, &g, &b], self.chroma)?;
        } else {
            // Partial selection: build the transformed frame separately, then copy only the
            // selected planes over the original.
            let mut tmp = FrameRefMut::copy_of(core, &frame);
            rgb_to_frame(&mut tmp, &self.fmt, [&r, &g, &b], self.chroma)?;
            for p in 0..self.fmt.planes {
                if planes.contains(&p) {
                    let h = frame.height(p);
                    let bytes = frame.width(p) * self.fmt.bytes as usize;
                    for row in 0..h {
                        out.data_row_mut(p, row)[..bytes]
                            .copy_from_slice(&tmp.data_row(p, row)[..bytes]);
                    }
                }
            }
        }

        out.props_mut()
            .set_data(PROP_BACKEND, resolved.backend.label().as_bytes())?;
        let method = match &resolved.lut {
            Some(_) => format!("lut:{}", self.lut_size),
            None => "native".to_string(),
        };
        out.props_mut().set_data(PROP_METHOD, method.as_bytes())?;
        Ok(out.into())
    }
}

/// Parses the `method` argument.
pub fn parse_method(spec: &str, func: &str) -> Result<Method> {
    Method::parse(spec, func)
}

/// Parses and validates the `lut_size` argument.
pub fn parse_lut_size(v: i64, func: &str) -> Result<u32> {
    if !(0..=1024).contains(&v) {
        bail!("{func}: `lut_size` must be in 0..=1024, got {v}");
    }
    let size = v as u32;
    if !SUPPORTED_SIZES.contains(&size) {
        bail!("{func}: `lut_size` must be one of {SUPPORTED_SIZES:?}, got {size}");
    }
    Ok(size)
}

/// Parses the `gamut` argument.
pub fn parse_gamut(spec: &str, func: &str) -> Result<Gamut> {
    match spec.trim().to_ascii_lowercase().as_str() {
        // The default. Out-of-range values are left for the encoder to clamp, which is the only
        // choice that preserves limited-range YUV headroom.
        "" | "none" => Ok(Gamut::None),
        "clip" => Ok(Gamut::Clip),
        "soft" => Ok(Gamut::Soft),
        other => bail!("{func}: `gamut` must be \"none\", \"clip\" or \"soft\", got {other:?}"),
    }
}

/// Parses and validates the `luma_lock` argument.
pub fn parse_luma_lock(v: f64, func: &str) -> Result<f32> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        bail!("{func}: `luma_lock` must be in 0.0..=1.0, got {v}");
    }
    Ok(v as f32)
}

/// Reads the model from a `.cm` file.
pub fn read_model_file(path: &str, func: &str) -> Result<Model> {
    let bytes =
        std::fs::read(path).with_context(|| format!("{func}: reading model file {path}"))?;
    Model::from_bytes(&bytes).with_context(|| format!("{func}: parsing {path} as a .cm model"))
}
