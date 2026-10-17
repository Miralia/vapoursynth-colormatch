//! VapourSynth plugin `cm` — rotated-marginal sliced-OT colour matching.
//!
//! Four filters, and deliberately no more:
//!
//! | Filter | Purpose |
//! |---|---|
//! | `cm.BuildModel` | pool samples from several clips into one global model |
//! | `cm.Apply` | apply a model to a clip |
//! | `cm.Devices` | report the GPU adapters the GPU backend can see |
//! | `cm.ExportLUT` | write a model out as `.cube` / `.clut` / `.cm` |
//!
//! Everything else — slicing clips into segments, scene handling, cache paths, backend policy,
//! format normalisation, statistics curves — lives in the Python wrapper `vs_colormatch`, where it
//! can be written in ordinary graph-building code.
//!
//! # A note on threading
//!
//! `vapoursynth-rs` creates every filter as `fmParallel` with no declared dependencies. That is
//! safe for this plugin by construction:
//!
//! * `cm.Apply` requests only frame `n` from its source (strictly spatial) plus frame `0` from the
//!   model clip, which is a single 1×1 frame.
//! * `cm.BuildModel` produces a one-frame clip, so its `get_frame_initial` burst of requests
//!   happens exactly once for the whole build.
//!
//! Neither filter issues a large burst of requests per output frame, which is the situation that
//! would require `fmParallelRequests` plus `rpStrictSpatial`/`rpFrameReuseLastOnly` hints.

#[macro_use]
extern crate vapoursynth;

mod apply;
mod build;
mod devices;
mod export;
mod frame_io;
mod sample;

use anyhow::{Context, Error, Result, bail};
use colormatch_core::backend::BackendPreference;
use colormatch_core::build::{AxesScheme, BuildOptions};
use colormatch_core::model::TABLE_NODES;
use std::ffi::CString;
use vapoursynth::core::CoreRef;
use vapoursynth::map::ValueIter;

use vapoursynth::plugins::{Filter, FilterArgument, FilterFunction, Metadata};
use vapoursynth::prelude::*;

use crate::apply::{
    ApplyFilter, parse_gamut, parse_luma_lock, parse_lut_size, parse_method, read_model_file,
};
use crate::build::{BuildModelFilter, Input, ModelOnlyFilter};
use crate::devices::DevicesFilter;
use crate::export::ExportLutFilter;
use crate::frame_io::{ChromaMode, ClipFormat};
use crate::sample::FrameSelection;

/// Reads an optional integer argument, applying a range check.
fn opt_int(map: &Map, key: &str, default: i64, lo: i64, hi: i64, func: &str) -> Result<i64> {
    let v = map.get_int(key).unwrap_or(default);
    if v < lo || v > hi {
        bail!("{func}: `{key}` must be in {lo}..={hi}, got {v}");
    }
    Ok(v)
}

/// Reads an optional float argument, applying a range check.
fn opt_float(map: &Map, key: &str, default: f64, lo: f64, hi: f64, func: &str) -> Result<f64> {
    let v = map.get_float(key).unwrap_or(default);
    if !v.is_finite() || v < lo || v > hi {
        bail!("{func}: `{key}` must be a finite value in {lo}..={hi}, got {v}");
    }
    Ok(v)
}

/// Reads an optional `data` argument as UTF-8.
fn opt_utf8(map: &Map, key: &str, default: &str, func: &str) -> Result<String> {
    match map.get_data(key) {
        Ok(b) => std::str::from_utf8(b)
            .map(str::to_owned)
            .map_err(|_| anyhow::anyhow!("{func}: `{key}` is not valid UTF-8")),
        Err(_) => Ok(default.to_owned()),
    }
}

/// Reads an optional `data` argument as an `Option<String>`.
fn maybe_utf8(map: &Map, key: &str, func: &str) -> Result<Option<String>> {
    match map.get_data(key) {
        Ok(b) => Ok(Some(
            std::str::from_utf8(b)
                .map_err(|_| anyhow::anyhow!("{func}: `{key}` is not valid UTF-8"))?
                .to_owned(),
        )),
        Err(_) => Ok(None),
    }
}

/// Validates `matrix` / `range` argument values.
fn check_space_args(matrix: Option<i64>, range: Option<i64>, func: &str) -> Result<()> {
    if let Some(m) = matrix
        && !(0..=255).contains(&m)
    {
        bail!("{func}: `matrix` must be in 0..=255, got {m}");
    }
    if let Some(r) = range
        && !(0..=1).contains(&r)
    {
        bail!("{func}: `range` must be 0 (limited) or 1 (full), got {r}");
    }
    Ok(())
}

/// Parses the `frames` selection string.
///
/// Accepted forms:
///
/// * `"all"` (or empty) — up to 64 frames spread evenly across the clip
/// * `"even:N"` — `N` frames spread evenly across the clip (the default)
/// * `"every:N"` — every `N`-th frame
/// * `"0,5,9"` — an explicit list
fn parse_frame_selection(spec: &str, num_frames: usize, func: &str) -> Result<FrameSelection> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
        return Ok(FrameSelection::Even(64.min(num_frames.max(1))));
    }
    if let Some(rest) = spec.strip_prefix("even:") {
        let n: usize = rest
            .trim()
            .parse()
            .with_context(|| format!("{func}: `frames` expected `even:<count>`, got {spec:?}"))?;
        if n == 0 {
            bail!("{func}: `frames=even:0` is meaningless");
        }
        return Ok(FrameSelection::Even(n));
    }
    if let Some(rest) = spec.strip_prefix("every:") {
        let n: usize = rest
            .trim()
            .parse()
            .with_context(|| format!("{func}: `frames` expected `every:<step>`, got {spec:?}"))?;
        if n == 0 {
            bail!("{func}: `frames=every:0` is meaningless");
        }
        return Ok(FrameSelection::Every(n));
    }
    if spec
        .chars()
        .all(|c| c.is_ascii_digit() || c == ',' || c == ' ')
    {
        let list: Result<Vec<usize>, _> = spec
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::parse::<usize>)
            .collect();
        let list =
            list.with_context(|| format!("{func}: `frames` is not a valid frame list: {spec:?}"))?;
        return Ok(FrameSelection::Explicit(list));
    }
    bail!(
        "{func}: `frames` must be \"all\", \"even:<n>\", \"every:<n>\" or a comma-separated list, \
         got {spec:?}"
    )
}

/// Parses the `axes` argument into an [`AxesScheme`].
/// Parses the `compat` argument: which builder fits the transfer tables.
///
/// `"v2"` is the default builder this project is built around. `"drdre"` reproduces the original's
/// structure so the two can be compared; see [`colormatch_core::build::Builder::DrDre`] for exactly
/// what is and is not reproduced, because the name would otherwise overclaim.
fn parse_compat(spec: &Option<String>, func: &str) -> Result<colormatch_core::build::Builder> {
    use colormatch_core::build::Builder;
    match spec.as_deref().map(str::trim) {
        None | Some("") => Ok(Builder::Splat),
        Some(s) if s.eq_ignore_ascii_case("v2") => Ok(Builder::Splat),
        Some(s) if s.eq_ignore_ascii_case("drdre") => Ok(Builder::DrDre),
        Some(other) => bail!("{func}: `compat` must be \"v2\" or \"drdre\", got {other:?}"),
    }
}

fn parse_axes(spec: &str, func: &str) -> Result<AxesScheme> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "" | "fibonacci" => Ok(AxesScheme::Fibonacci),
        "drdre" => Ok(AxesScheme::DrDre),
        other => bail!(
            "{func}: `axes` must be \"fibonacci\" or \"drdre\", got {other:?}. Reproducible tests \
             inject explicit directions through the Rust API instead"
        ),
    }
}

/// Resolves a model argument pair into either an in-memory model or a model node.
fn split_model_args<'core>(
    model: Option<Node<'core>>,
    model_file: Option<&[u8]>,
    func: &str,
) -> Result<(Option<Node<'core>>, Option<colormatch_core::model::Model>)> {
    if model.is_some() && model_file.is_some() {
        bail!("{func}: pass either `model` or `model_file`, not both");
    }
    if model.is_none() && model_file.is_none() {
        bail!("{func}: one of `model` or `model_file` is required");
    }
    let in_memory = match model_file {
        Some(b) => {
            let path = std::str::from_utf8(b)
                .map_err(|_| anyhow::anyhow!("{func}: `model_file` is not valid UTF-8"))?;
            Some(read_model_file(path, func)?)
        }
        None => None,
    };
    Ok((model, in_memory))
}

/// `cm.BuildModel`.
///
/// Hand-written rather than macro-generated because the reference-side argument must be exposed to
/// users as `ref`, which is a Rust keyword and cannot be used as a parameter name.
struct BuildModelFunction {
    args: String,
}

impl BuildModelFunction {
    fn new() -> Self {
        Self {
            args: concat!(
                "test:vnode[]:opt;",
                "ref:vnode[]:opt;",
                "weights:float[]:opt;",
                "samples:int:opt;",
                "directions:int:opt;",
                "bins:int:opt;",
                "axes:data:opt;",
                "frames:data:opt;",
                "seed:int:opt;",
                "passes:int:opt;",
                "step:float:opt;",
                "matrix:int:opt;",
                "range:int:opt;",
                "matrices:int[]:opt;",
                "ranges:int[]:opt;",
                "compat:data:opt;",
                "cache:data:opt;",
                "model_file:data:opt;",
            )
            .to_owned(),
        }
    }
}

impl FilterFunction for BuildModelFunction {
    fn name(&self) -> &str {
        "BuildModel"
    }

    fn args(&self) -> &str {
        &self.args
    }

    fn create<'core>(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        map: &Map<'core>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        const FUNC: &str = "cm.BuildModel";

        // Either build from clips, or reload a serialized model. Both produce a model clip, so the
        // two ways of obtaining a model are interchangeable downstream and `cm.Apply` does not care
        // which was used.
        let model_file = maybe_utf8(map, "model_file", FUNC)?;
        if let Some(path) = &model_file {
            if map.get_video_node_iter("test").is_ok() || map.get_video_node_iter("ref").is_ok() {
                bail!(
                    "{FUNC}: pass either `test`/`ref` or `model_file`, not both. Reloading a model \
                     does not need clips to sample from"
                );
            }
            let model = read_model_file(path, FUNC)?;
            return Ok(Some(Box::new(ModelOnlyFilter::new(model))));
        }

        let test: Vec<Node<'core>> = match map.get_video_node_iter("test") {
            Ok(it) => it.collect(),
            Err(_) => bail!(
                "{FUNC}: `test` is required (an array of clips to sample), unless `model_file` is \
                 given to reload a serialized model"
            ),
        };
        let refs: Vec<Node<'core>> = match map.get_video_node_iter("ref") {
            Ok(it) => it.collect(),
            Err(_) => bail!(
                "{FUNC}: `ref` is required (an array of clips to sample), unless `model_file` is \
                 given to reload a serialized model"
            ),
        };
        if test.is_empty() {
            bail!("{FUNC}: `test` must contain at least one clip");
        }
        if refs.is_empty() {
            bail!("{FUNC}: `ref` must contain at least one clip");
        }

        let n_inputs = test.len() + refs.len();
        let mut weights = vec![1.0f64; n_inputs];
        if let Ok(it) = map.get_float_iter("weights") {
            let w: Vec<f64> = it.collect();
            if w.len() != n_inputs {
                bail!(
                    "{FUNC}: `weights` must have {n_inputs} entries (one per input clip), got {}",
                    w.len()
                );
            }
            for (i, v) in w.iter().enumerate() {
                if !v.is_finite() || *v < 0.0 {
                    bail!("{FUNC}: `weights[{i}]` must be a finite non-negative number, got {v}");
                }
            }
            weights = w;
        }

        let samples = opt_int(map, "samples", 262_144, 1024, 64_000_000, FUNC)?;
        let directions = opt_int(map, "directions", 32, 1, 256, FUNC)?;
        // Capped at the model format's fixed node count: a finer histogram has nowhere to go, and
        // the core rejects it rather than truncating.
        let bins = opt_int(map, "bins", 256, 16, (TABLE_NODES - 1) as i64, FUNC)?;
        let passes = opt_int(map, "passes", 1, 1, 16, FUNC)?;
        let step = opt_float(map, "step", 1.0, 0.05, 1.5, FUNC)?;
        let seed = opt_int(map, "seed", 0x5EED, 0, i64::MAX, FUNC)? as u64;
        let axes_spec = opt_utf8(map, "axes", "fibonacci", FUNC)?;
        let frames_spec = opt_utf8(map, "frames", "even:64", FUNC)?;
        // Validated here rather than lazily in the filter: an unsupported value should fail at
        // graph-construction time, not on the first frame request.
        let compat = maybe_utf8(map, "compat", FUNC)?;
        let cache = maybe_utf8(map, "cache", FUNC)?;

        let matrix = map.get_int("matrix").ok();
        let range = map.get_int("range").ok();
        check_space_args(matrix, range, FUNC)?;

        // Per-input overrides. A build can mix families -- a YUV test clip against an RGB reference
        // is an ordinary thing to do -- and the two need different matrices, so one scalar for the
        // whole call cannot express it. `-1` means "not specified", which is the family default.
        let matrices: Option<Vec<i64>> = map.get_int_iter("matrices").ok().map(|it| it.collect());
        let ranges: Option<Vec<i64>> = map.get_int_iter("ranges").ok().map(|it| it.collect());
        if matrix.is_some() && matrices.is_some() {
            bail!("{FUNC}: pass either `matrix` or `matrices`, not both");
        }
        if range.is_some() && ranges.is_some() {
            bail!("{FUNC}: pass either `range` or `ranges`, not both");
        }
        for (name, list) in [("matrices", &matrices), ("ranges", &ranges)] {
            if let Some(v) = list
                && v.len() != n_inputs
            {
                bail!(
                    "{FUNC}: `{name}` must have one entry per input clip ({n_inputs}), got {}",
                    v.len()
                );
            }
        }

        let options = BuildOptions {
            directions: directions as usize,
            bins: bins as usize,
            axes: parse_axes(&axes_spec, FUNC)?,
            passes: passes as usize,
            step: step as f32,
            seed,
            binning: colormatch_core::build::Binning::default(),
            builder: parse_compat(&compat, FUNC)?,
            ..Default::default()
        };

        let mut inputs = Vec::with_capacity(n_inputs);
        for (i, node) in test.iter().chain(refs.iter()).enumerate() {
            let info = node.info();
            // Per-input when given, otherwise the scalar, otherwise the family default. `-1` is the
            // "unspecified" sentinel.
            let m_i = matrices
                .as_ref()
                .map(|v| v[i])
                .filter(|v| *v >= 0)
                .or(matrix);
            let r_i = ranges.as_ref().map(|v| v[i]).filter(|v| *v >= 0).or(range);
            let fmt = ClipFormat::from_info(&info, m_i, r_i, FUNC)
                .map_err(|e| anyhow::anyhow!("{FUNC}: input clip {i}: {e:#}"))?;
            let selection = parse_frame_selection(&frames_spec, info.num_frames, FUNC)
                .map_err(|e| anyhow::anyhow!("{FUNC}: input clip {i}: {e:#}"))?;
            let frames = selection.resolve(info.num_frames);
            if frames.is_empty() {
                bail!("{FUNC}: input clip {i} has no frames to sample");
            }
            inputs.push(Input {
                node: node.clone(),
                fmt,
                frames,
                weight: weights[i],
            });
        }

        let filter = BuildModelFilter::new(inputs, test.len(), options, samples as usize)
            .with_cache_path(cache);
        Ok(Some(Box::new(filter)))
    }
}

make_filter_function! {
    ApplyFunction, "Apply"

    #[allow(clippy::too_many_arguments)]
    fn create_apply<'core>(
        _api: API,
        _core: CoreRef<'core>,
        clip: Node<'core>,
        model: Option<Node<'core>>,
        model_file: Option<&[u8]>,
        backend: Option<&[u8]>,
        device_id: Option<i64>,
        method: Option<&[u8]>,
        lut_size: Option<i64>,
        planes: Option<ValueIter<'_, 'core, i64>>,
        gamut: Option<&[u8]>,
        luma_lock: Option<f64>,
        chroma: Option<&[u8]>,
        matrix: Option<i64>,
        range: Option<i64>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        const FUNC: &str = "cm.Apply";

        let (model_node, in_memory) = split_model_args(model, model_file, FUNC)?;
        let info = clip.info();
        let fmt = ClipFormat::from_info(&info, matrix, range, FUNC)?;

        let plane_list = match planes {
            Some(it) => {
                let mut v = Vec::new();
                for p in it {
                    if p < 0 || p as usize >= fmt.planes {
                        bail!("{FUNC}: `planes` contains {p}, but the clip has {} planes", fmt.planes);
                    }
                    let p = p as usize;
                    if !v.contains(&p) {
                        v.push(p);
                    }
                }
                if v.is_empty() {
                    bail!("{FUNC}: `planes` must not be empty");
                }
                Some(v)
            }
            None => None,
        };

        let preference = match backend {
            Some(b) => {
                let s = std::str::from_utf8(b)
                    .map_err(|_| anyhow::anyhow!("{FUNC}: `backend` is not valid UTF-8"))?;
                BackendPreference::parse(s).ok_or_else(|| {
                    anyhow::anyhow!(
                        "{FUNC}: `backend` must be one of auto, cpu, simd, gpu, lut; got {s:?}"
                    )
                })?
            }
            None => BackendPreference::Auto,
        };
        let device_id = device_id.unwrap_or(0);
        if !(0..=64).contains(&device_id) {
            bail!("{FUNC}: `device_id` must be in 0..=64, got {device_id}");
        }

        let method = match method {
            Some(b) => parse_method(
                std::str::from_utf8(b)
                    .map_err(|_| anyhow::anyhow!("{FUNC}: `method` is not valid UTF-8"))?,
                FUNC,
            )?,
            None => apply::Method::Native,
        };
        let lut_size = parse_lut_size(lut_size.unwrap_or(33), FUNC)?;
        let gamut = match gamut {
            Some(b) => parse_gamut(
                std::str::from_utf8(b)
                    .map_err(|_| anyhow::anyhow!("{FUNC}: `gamut` is not valid UTF-8"))?,
                FUNC,
            )?,
            None => colormatch_core::apply::Gamut::None,
        };
        let luma_lock = parse_luma_lock(luma_lock.unwrap_or(0.0), FUNC)?;
        let chroma = match chroma {
            Some(b) => ChromaMode::parse(
                std::str::from_utf8(b)
                    .map_err(|_| anyhow::anyhow!("{FUNC}: `chroma` is not valid UTF-8"))?,
                FUNC,
            )?,
            None => ChromaMode::default(),
        };

        let options = colormatch_core::apply::ApplyOptions {
            gamut,
            luma_lock,
            ..colormatch_core::apply::ApplyOptions::default()
        };

        Ok(Some(Box::new(ApplyFilter::new(
            clip,
            model_node,
            in_memory,
            fmt,
            plane_list,
            options,
            preference,
            method,
            lut_size,
            chroma,
            device_id as u32,
        ))))
    }
}

make_filter_function! {
    DevicesFunction, "Devices"

    fn create_devices<'core>(
        _api: API,
        _core: CoreRef<'core>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        Ok(Some(Box::new(DevicesFilter)))
    }
}

make_filter_function! {
    ExportLUTFunction, "ExportLUT"

    #[allow(clippy::too_many_arguments)]
    fn create_export_lut<'core>(
        _api: API,
        _core: CoreRef<'core>,
        clip: Node<'core>,
        model: Option<Node<'core>>,
        model_file: Option<&[u8]>,
        path: &[u8],
        size: Option<i64>,
        format: Option<&[u8]>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        const FUNC: &str = "cm.ExportLUT";

        let (model_node, in_memory) = split_model_args(model, model_file, FUNC)?;
        let path = std::str::from_utf8(path)
            .map_err(|_| anyhow::anyhow!("{FUNC}: `path` is not valid UTF-8"))?;
        let size = parse_lut_size(size.unwrap_or(33), FUNC)?;
        let format = match format {
            Some(b) => std::str::from_utf8(b)
                .map_err(|_| anyhow::anyhow!("{FUNC}: `format` is not valid UTF-8"))?
                .to_owned(),
            None => "cube".to_owned(),
        };
        let filter = ExportLutFilter::new(clip, in_memory, model_node, path, size, &format)?;
        Ok(Some(Box::new(filter)))
    }
}

/// Number of grid nodes per axis, re-exported for tests and tooling.
pub const NODES: usize = TABLE_NODES;

/// Plugin entry point.
///
/// Hand-written rather than macro-generated for one reason: `export_vapoursynth_plugin!` passes a
/// hard-coded `1` as the plugin version, so `core.plugins()` would report `0.1` for every release and
/// the version a user reads there would be wrong. Everything else mirrors what the macro does — the
/// same `Metadata`, the same panic guard, and filter registration through the crate's own
/// `call_register_func` — so nothing else about the plugin's behaviour changes.
///
/// # Safety
///
/// Called by VapourSynth with a valid `VSPlugin` and plugin API table. The pointers are only used
/// within this function and are not stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn VapourSynthPluginInit2(
    plugin: *mut vapoursynth::ffi::VSPlugin,
    vspapi: *const vapoursynth::ffi::VSPLUGINAPI,
) {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::process;

    use vapoursynth::plugins::ffi::call_register_func;

    let metadata = Metadata {
        identifier: colormatch_core::PLUGIN_IDENTIFIER,
        namespace: colormatch_core::PLUGIN_NAMESPACE,
        name: colormatch_core::PLUGIN_NAME,
        read_only: true,
    };

    let closure = move || {
        let identifier = CString::new(metadata.identifier).expect("identifier has no NUL");
        let namespace = CString::new(metadata.namespace).expect("namespace has no NUL");
        let name = CString::new(metadata.name).expect("name has no NUL");

        let api_version = (vapoursynth::ffi::VAPOURSYNTH_API_MAJOR << 16
            | vapoursynth::ffi::VAPOURSYNTH_API_MINOR) as i32;
        // `read_only` means the plugin's functions cannot be modified or removed.
        let flags = if metadata.read_only {
            0
        } else {
            vapoursynth::ffi::VSPluginConfigFlags_pcModifiable as i32
        };

        // Safety: VapourSynth passes a valid plugin API table, and the CStrings outlive the call.
        unsafe {
            let configure = (*vspapi)
                .configPlugin
                .expect("configPlugin is missing from the VapourSynth plugin API");
            configure(
                identifier.as_ptr(),
                namespace.as_ptr(),
                name.as_ptr(),
                colormatch_core::PLUGIN_VERSION,
                api_version,
                flags,
                plugin,
            );

            call_register_func(vspapi, plugin, BuildModelFunction::new());
            call_register_func(vspapi, plugin, ApplyFunction::new());
            call_register_func(vspapi, plugin, DevicesFunction::new());
            call_register_func(vspapi, plugin, ExportLUTFunction::new());
        }
    };

    // A panic across the FFI boundary is undefined behaviour; abort instead, as the crate's macro
    // does.
    if catch_unwind(AssertUnwindSafe(closure)).is_err() {
        process::abort();
    }
}
