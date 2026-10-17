//! `cm.Devices` — reports the GPU adapters the GPU backend can see.
//!
//! `vs-jetpack` obtains the same information from OpenCL via `pyopencl` and then guesses which of
//! several *third-party* plugins to dispatch to. Our backends are our own, so the enumeration is
//! done by the code that will actually run, and no extra Python dependency is needed.
//!
//! Output is a 1×1 GRAY8 clip whose frame properties describe the adapters:
//!
//! * `_CM_AdapterCount` — integer
//! * `_CM_Adapter{n}_Name`, `_CM_Adapter{n}_Api`, `_CM_Adapter{n}_Type` — strings
//! * `_CM_Adapter{n}_Vendor`, `_CM_Adapter{n}_Device` — integers
//! * `_CM_Backends` — comma-separated list of compiled-in backends
//! * `_CM_GpuEnabled` — `1` when the `gpu` feature is compiled in

use anyhow::{Error, Result};
use colormatch_core::backend::enumerate_adapters;
use vapoursynth::core::CoreRef;
use vapoursynth::format::FormatID;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::*;
use vapoursynth::video_info::{Framerate, Resolution, VideoInfo};

/// GRAY8 preset format id.
const GRAY8: i32 = (1 << 28) | (8 << 16);

/// The `cm.Devices` filter.
pub struct DevicesFilter;

impl<'core> Filter<'core> for DevicesFilter {
    fn video_info(&self, _api: API, core: CoreRef<'core>) -> Vec<VideoInfo<'core>> {
        let format = core
            .get_format(FormatID::from(GRAY8))
            .expect("GRAY8 is a preset format");
        vec![VideoInfo {
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
        }]
    }

    fn get_frame_initial(
        &self,
        _api: API,
        core: CoreRef<'core>,
        _context: FrameContext,
        _n: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        let adapters = enumerate_adapters();
        let format = core
            .get_format(FormatID::from(GRAY8))
            .expect("GRAY8 is a preset format");
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
        let mut props = frame.props_mut();
        props.set_int("_CM_AdapterCount", adapters.len() as i64)?;
        for (i, a) in adapters.iter().enumerate() {
            props.set_data(&format!("_CM_Adapter{i}_Name"), a.name.as_bytes())?;
            props.set_data(&format!("_CM_Adapter{i}_Api"), a.api.as_bytes())?;
            props.set_data(&format!("_CM_Adapter{i}_Type"), a.device_type.as_bytes())?;
            props.set_int(&format!("_CM_Adapter{i}_Vendor"), a.vendor as i64)?;
            props.set_int(&format!("_CM_Adapter{i}_Device"), a.device as i64)?;
        }
        let backends = if cfg!(feature = "gpu") {
            "scalar,simd,gpu,lut"
        } else {
            "scalar,simd,lut"
        };
        props.set_data("_CM_Backends", backends.as_bytes())?;
        props.set_int("_CM_GpuEnabled", i64::from(cfg!(feature = "gpu")))?;
        // Adapter enumeration can exist while the compute pipelines do not. Callers must be able to
        // tell the difference, otherwise `auto` would promise a backend that cannot run.
        props.set_int(
            "_CM_GpuApply",
            i64::from(colormatch_core::backend::gpu_apply_available(0)),
        )?;
        props.set_data("_CM_Version", colormatch_core::VERSION.as_bytes())?;
        Ok(Some(frame.into()))
    }

    fn get_frame(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        _context: FrameContext,
        _n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        unreachable!("generated in get_frame_initial")
    }
}
