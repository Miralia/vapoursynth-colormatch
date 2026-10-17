//! `cm.ExportLUT` — writes a model out as a 3D LUT and passes the input clip through.
//!
//! The write happens at filter *creation* time, which is a pure, one-shot operation on a model
//! that is already fully determined. Passing the input clip through unchanged means the filter can
//! be dropped anywhere in a graph without altering the output.
//!
//! Supported targets:
//!
//! * `cube` — Adobe `.cube`, clamped to `[0, 1]`
//! * `clut` — Resolve `.clut`, raw little-endian `f32` triplets
//! * `cm` — the model itself, for later reuse via `cm.Apply(model_file=...)`

use anyhow::{Context, Result, bail};
use colormatch_core::lut::{SUPPORTED_SIZES, write_clut, write_cube};
use colormatch_core::model::Model;
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::*;

use crate::build::model_from_node;

/// The `cm.ExportLUT` filter.
pub struct ExportLutFilter<'core> {
    source: Node<'core>,
}

impl<'core> ExportLutFilter<'core> {
    /// Writes the LUT, then returns a filter that passes `source` through.
    pub fn new(
        source: Node<'core>,
        model: Option<Model>,
        model_node: Option<Node<'core>>,
        path: &str,
        size: u32,
        format: &str,
    ) -> Result<Self> {
        const FUNC: &str = "cm.ExportLUT";

        if !SUPPORTED_SIZES.contains(&size) {
            bail!("{FUNC}: `size` must be one of {SUPPORTED_SIZES:?}, got {size}");
        }
        let model = match (model, model_node) {
            (Some(m), _) => m,
            (None, Some(node)) => model_from_node(&node, FUNC)?,
            (None, None) => bail!("{FUNC}: exactly one of `model` or `model_file` is required"),
        };
        if model.is_empty() {
            bail!("{FUNC}: the model is empty (no slices); nothing to export");
        }

        match format.trim().to_ascii_lowercase().as_str() {
            "cube" => {
                let text = write_cube(&model, size, None)
                    .with_context(|| format!("{FUNC}: baking a {size}-node .cube"))?;
                std::fs::write(path, text).with_context(|| format!("{FUNC}: writing {path}"))?;
            }
            "clut" => {
                let bytes = write_clut(&model, size)
                    .with_context(|| format!("{FUNC}: baking a {size}-node .clut"))?;
                std::fs::write(path, bytes).with_context(|| format!("{FUNC}: writing {path}"))?;
            }
            "cm" => {
                std::fs::write(path, model.to_bytes())
                    .with_context(|| format!("{FUNC}: writing {path}"))?;
            }
            other => bail!("{FUNC}: `format` must be \"cube\", \"clut\" or \"cm\", got {other:?}"),
        }
        Ok(Self { source })
    }
}

impl<'core> Filter<'core> for ExportLutFilter<'core> {
    fn video_info(
        &self,
        _api: API,
        _core: CoreRef<'core>,
    ) -> Vec<vapoursynth::video_info::VideoInfo<'core>> {
        vec![self.source.info()]
    }

    fn get_frame_initial(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<Option<FrameRef<'core>>, anyhow::Error> {
        self.source.request_frame_filter(context, n);
        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<FrameRef<'core>, anyhow::Error> {
        self.source
            .get_frame_filter(context, n)
            .ok_or_else(|| anyhow::anyhow!("cm.ExportLUT: could not get frame {n}"))
    }
}
