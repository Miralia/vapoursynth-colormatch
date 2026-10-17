//! `cm.BuildModel` — pools samples from several clips into one global model.
//!
//! The output is a **single 1×1 GRAY8 frame** carrying the serialized model in the `_CM_Model`
//! frame property and the build statistics in `_CM_Stats`.
//!
//! Why a one-frame clip rather than a side channel: VapourSynth filters cannot return arbitrary
//! data, frame properties are the only supported transport, and a 1×1 frame keeps the transfer
//! nearly free (a K=32 model is ~100 KB). Because the clip is exactly one frame long, the filter
//! body runs **once** for the whole build, so it can request dozens of frames from its inputs
//! without any of the cache-pressure concerns that affect per-frame temporal filters.
//!
//! The two sides are independent: `test` is the material being corrected and `ref` is the target
//! look. They need not have the same length, resolution, format or frame count. All samples from
//! the `test` side form the source distribution and all samples from the `ref` side form the
//! target distribution.

use std::sync::OnceLock;

use anyhow::{Context, Error, Result, bail};
use colormatch_core::build::{BuildOptions, Samples, build_model};
use colormatch_core::model::Model;
use colormatch_core::{PROP_MODEL, PROP_STATS};
use vapoursynth::core::CoreRef;
use vapoursynth::format::FormatID;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::*;
use vapoursynth::video_info::{Framerate, Resolution, VideoInfo};

use crate::frame_io::ClipFormat;
use crate::sample::sample_frames;

/// GRAY8 preset format id: `VS_MAKE_VIDEO_ID(cfGray, stInteger, 8, 0, 0)`.
const GRAY8: i32 = (1 << 28) | (8 << 16);

/// One input clip plus how to sample it.
pub struct Input<'core> {
    /// The clip.
    pub node: Node<'core>,
    /// Its resolved format.
    pub fmt: ClipFormat,
    /// Which frames to read.
    pub frames: Vec<usize>,
    /// Relative weight within its side.
    pub weight: f64,
}

/// The outcome of a build, cached for the lifetime of the filter.
struct Built {
    model: Model,
}

/// The `cm.BuildModel` filter.
pub struct BuildModelFilter<'core> {
    inputs: Vec<Input<'core>>,
    /// Number of leading inputs that belong to the `test` side.
    num_test: usize,
    options: BuildOptions,
    /// Total sample-pixel budget per side.
    samples_per_side: usize,
    /// Optional side-car path; the serialized model is written there on a successful build so the
    /// Python wrapper's cache can be populated without a second pass.
    cache_path: Option<String>,
    result: OnceLock<std::result::Result<Built, String>>,
}

impl<'core> BuildModelFilter<'core> {
    /// Creates the filter.
    pub fn new(
        inputs: Vec<Input<'core>>,
        num_test: usize,
        options: BuildOptions,
        samples_per_side: usize,
    ) -> Self {
        Self {
            inputs,
            num_test,
            options,
            samples_per_side,
            cache_path: None,
            result: OnceLock::new(),
        }
    }

    /// Sets a path to write the serialized model to after a successful build.
    pub fn with_cache_path(mut self, path: Option<String>) -> Self {
        self.cache_path = path;
        self
    }

    /// Builds the model exactly once and caches both success and failure.
    ///
    /// Caching the failure matters: without it, a misconfigured build would be retried on every
    /// frame request and VapourSynth would log the same error repeatedly.
    fn build(&self, context: FrameContext) -> Result<&Built, Error> {
        if let Some(r) = self.result.get() {
            return match r {
                Ok(b) => Ok(b),
                Err(e) => Err(anyhow::anyhow!("{e}")),
            };
        }
        match self.build_inner(context) {
            Ok(built) => {
                let _ = self.result.set(Ok(built));
                match self.result.get() {
                    Some(Ok(b)) => Ok(b),
                    _ => unreachable!("just stored a success"),
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                let _ = self.result.set(Err(msg.clone()));
                Err(anyhow::anyhow!(msg))
            }
        }
    }

    fn build_inner(&self, context: FrameContext) -> Result<Built> {
        const FUNC: &str = "cm.BuildModel";

        // `compat` is resolved to a `Builder` at creation time and lives in `self.options`, so there
        // is nothing to validate here. An earlier version re-checked it on every frame request and
        // rejected `drdre`, which meant the builder could never be reached.
        let test_weight: f64 = self.inputs[..self.num_test].iter().map(|i| i.weight).sum();
        let ref_weight: f64 = self.inputs[self.num_test..].iter().map(|i| i.weight).sum();
        if test_weight <= 0.0 || ref_weight <= 0.0 {
            bail!("{FUNC}: both sides need at least one clip with a positive weight");
        }

        let mut test = Samples::default();
        let mut reference = Samples::default();

        for (i, input) in self.inputs.iter().enumerate() {
            let on_test = i < self.num_test;
            let side_weight = if on_test { test_weight } else { ref_weight };
            let budget = ((self.samples_per_side as f64) * (input.weight / side_weight))
                .round()
                .max(1.0);
            let per_frame = ((budget as usize) / input.frames.len().max(1)).max(1);

            let cloud = sample_frames(&input.frames, &input.fmt, per_frame, |n| {
                input.node.get_frame_filter(context, n).ok_or_else(|| {
                    anyhow::anyhow!("{FUNC}: could not read frame {n} from input clip {i}")
                })
            })
            .with_context(|| format!("{FUNC}: sampling input clip {i}"))?;

            if on_test {
                test.extend(&cloud);
            } else {
                reference.extend(&cloud);
            }
        }

        if test.is_empty() {
            bail!("{FUNC}: the test side produced no samples");
        }
        if reference.is_empty() {
            bail!("{FUNC}: the reference side produced no samples");
        }

        let model = build_model(&test, &reference, &self.options)
            .map_err(|e| anyhow::anyhow!("{FUNC}: {e}"))?;

        if let Some(path) = &self.cache_path {
            std::fs::write(path, model.to_bytes())
                .with_context(|| format!("{FUNC}: writing the model cache file {path}"))?;
        }

        Ok(Built { model })
    }
}

/// The 1x1 GRAY8 shape every model clip has.
fn model_video_info<'core>(core: CoreRef<'core>) -> VideoInfo<'core> {
    let format = core
        .get_format(FormatID::from(GRAY8))
        .expect("GRAY8 is a preset format");
    VideoInfo {
        format,
        resolution: Resolution {
            width: 1,
            height: 1,
        }
        .into(),
        framerate: Framerate {
            numerator: 1,
            denominator: 1,
        }
        .into(),
        num_frames: 1usize,
    }
}

/// Packs a model and its statistics into the frame a model clip carries.
///
/// Shared by the builder and the reloader so that a model obtained either way is byte-identical
/// downstream; `cm.Apply` cannot tell them apart, and neither should a test.
fn model_frame<'core>(core: CoreRef<'core>, model: &Model) -> Result<FrameRef<'core>, Error> {
    let format = core
        .get_format(FormatID::from(GRAY8))
        .expect("GRAY8 is a preset format");
    let bytes = model.to_bytes();
    let mut frame = unsafe {
        FrameRefMut::new_uninitialized(
            core,
            None,
            format,
            Resolution {
                width: 1,
                height: 1,
            },
        )
    };
    frame
        .props_mut()
        .set_data(PROP_MODEL, &bytes)
        .with_context(|| "attaching the model to the output frame".to_string())?;
    frame
        .props_mut()
        .set_float_array(PROP_STATS, &model.stats.as_props())
        .with_context(|| "attaching model statistics".to_string())?;
    frame
        .props_mut()
        .set_int("_CM_ModelBytes", bytes.len() as i64)?;
    Ok(frame.into())
}

/// A model clip carrying an already-loaded model and no inputs.
///
/// Produced by `cm.BuildModel(model_file=...)`, so that reloading a serialized model yields the same
/// kind of clip as building one. Without it, the only way to get a model clip back from a file was
/// to run `cm.Apply` on a throwaway 1x1 clip, which conflated loading with applying.
pub struct ModelOnlyFilter {
    model: Model,
}

impl ModelOnlyFilter {
    /// Wraps a loaded model as a model clip.
    pub fn new(model: Model) -> Self {
        Self { model }
    }
}

impl<'core> Filter<'core> for ModelOnlyFilter {
    fn video_info(&self, _api: API, core: CoreRef<'core>) -> Vec<VideoInfo<'core>> {
        vec![model_video_info(core)]
    }

    /// Produces the frame here rather than in `get_frame`.
    ///
    /// This filter has no inputs, so it never receives an `arAllFramesReady` activation and
    /// `get_frame` is never called: VapourSynth would report "No frame returned at the end of
    /// processing". A filter that needs no input frames must generate its output in
    /// `get_frame_initial`, as the trait documents.
    fn get_frame_initial(
        &self,
        _api: API,
        core: CoreRef<'core>,
        _context: FrameContext,
        _n: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        Ok(Some(model_frame(core, &self.model)?))
    }

    fn get_frame(
        &self,
        _api: API,
        core: CoreRef<'core>,
        _context: FrameContext,
        _n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        // Unreachable in practice, but returning the frame is correct and keeps the impl total.
        model_frame(core, &self.model)
    }
}

impl<'core> Filter<'core> for BuildModelFilter<'core> {
    fn video_info(&self, _api: API, core: CoreRef<'core>) -> Vec<VideoInfo<'core>> {
        vec![model_video_info(core)]
    }

    fn get_frame_initial(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        _n: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        // Request everything up front. This runs once for the whole clip, so a large burst is
        // safe: there is exactly one outstanding request set, not one per output frame.
        for input in &self.inputs {
            for &f in &input.frames {
                input.node.request_frame_filter(context, f);
            }
        }
        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        core: CoreRef<'core>,
        context: FrameContext,
        _n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        let built = self.build(context)?;
        model_frame(core, &built.model)
    }
}

/// Reads a serialized model out of a model clip's frame 0 (used by `cm.ExportLUT` and tests).
pub fn model_from_node<'core>(node: &Node<'core>, func: &str) -> Result<Model> {
    let frame = node
        .get_frame(0)
        .map_err(|e| anyhow::anyhow!("{func}: could not read the model clip's frame 0: {e}"))?;
    let props = frame.props();
    let bytes = props.get_data(PROP_MODEL).map_err(|_| {
        anyhow::anyhow!(
            "{func}: the clip has no `{PROP_MODEL}` property; pass the result of `cm.BuildModel`"
        )
    })?;
    Model::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{func}: invalid model: {e}"))
}
