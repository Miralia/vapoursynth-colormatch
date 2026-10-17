//! GPU backend: adapter enumeration and the WGSL compute pipeline.
//!
//! # What runs where
//!
//! [`enumerate_adapters`] lists what is available; [`available`] reports whether the apply path can
//! actually run, by creating the device once and caching the outcome; [`apply_gpu`] does the work.
//! [`crate::backend::gpu_apply_available`] forwards [`available`], so a build with the `gpu` feature
//! compiled in but no usable adapter does not advertise a GPU backend, and the `_CM_Backend` frame
//! property cannot name a GPU that did not run.
//!
//! The shader is a transliteration of `apply_impl` in `crate::apply`, in the same order: sanitise the
//! input, run the slices, apply the luma lock, then the gamut policy. `tests/gpu_matches_scalar.rs`
//! compares the two on random models and every gamut and luma-lock combination, and the integration
//! suite compares them on real frames in RGB48, YUV444P10 and YUV420P10. An identity model is an exact
//! no-op through the GPU, so the two paths are interchangeable.
//!
//! One WGSL shader compiles to Metal, Vulkan and DX12, with no vendor SDK, no runtime installer and
//! no per-platform code path.
//!
//! Cost here is nearly flat in `K`: a gather that costs a CPU a cache miss is a shared-memory load on
//! a GPU, so memory traffic rather than the slice count sets the pace. The CPU path scales linearly
//! in `K`, which is why `auto` switches to the GPU only above a threshold.
//!
//! The GPU does not make the pipeline free. It moves the transform out of the way and leaves the
//! fixed cost — format conversion, allocation, frame I/O — which is roughly a third of the total at
//! `K=32` and is what an upload-and-download design cannot remove. See `docs/performance.md`.
//!
//! # GPU-resident frames
//!
//! VapourSynth 4.3 exposes `getVulkanAPI()` and the `ffGPUOutput` filter flag, which would let a
//! filter hand back GPU-resident frames and skip the download half of the round trip. That path needs
//! a VapourSynth built with Vulkan support and a Vulkan driver on the target machine, and neither is
//! guaranteed across the platforms this plugin ships to. This backend therefore uploads and
//! downloads, which works everywhere the GPU backend itself does.
//!
//! The decision is isolated on purpose: the pipeline takes storage buffers, so replacing the staging
//! step with external-memory interop would not touch the shader. `cm.Devices` reports what the host
//! actually has, so a caller can tell whether the interop path is even a possibility there.

use std::sync::Arc;

use crate::backend::AdapterInfo;
use crate::model::Model;

/// Describes one `wgpu` backend in the vocabulary the rest of the code uses.
fn backend_label(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        _ => "other",
    }
}

/// Describes one `wgpu` device type in the vocabulary the rest of the code uses.
fn device_label(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        _ => "other",
    }
}

/// Rank for automatic selection: discrete beats integrated beats virtual beats a software adapter.
fn rank(t: wgpu::DeviceType) -> u8 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::VirtualGpu => 1,
        _ => 0,
    }
}

/// Enumerates the adapters a compute backend could run on, best first.
///
/// Software adapters (`DeviceType::Cpu`, e.g. lavapipe) are filtered out: they are slower than the
/// SIMD path and would make `backend="auto"` a pessimisation.
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    // `..._from_env` honours WGPU_BACKEND / WGPU_ADAPTER_NAME and friends, which is how a user
    // pins a specific adapter without us inventing another knob.
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

    let mut adapters: Vec<(u8, AdapterInfo)> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .map(|a| {
                let info = a.get_info();
                (
                    rank(info.device_type),
                    AdapterInfo {
                        name: info.name.clone(),
                        api: backend_label(info.backend).to_owned(),
                        device_type: device_label(info.device_type).to_owned(),
                        vendor: info.vendor,
                        device: info.device,
                    },
                )
            })
            .filter(|(r, _)| *r > 0)
            .collect();

    // Stable ordering: best device type first, then by name so the list is deterministic.
    adapters.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
    adapters.into_iter().map(|(_, info)| info).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_is_deterministic_and_excludes_software_adapters() {
        let a = enumerate_adapters();
        let b = enumerate_adapters();
        assert_eq!(a, b, "adapter enumeration must be stable");
        for adapter in &a {
            assert_ne!(
                adapter.device_type, "cpu",
                "software adapters must be filtered out"
            );
            assert!(!adapter.api.is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------------------------

/// The WGSL source, compiled into the binary so the plugin has no runtime asset dependency.
const SHADER: &str = include_str!("apply.wgsl");

/// Pixels per workgroup. 64 is the portable maximum for a 1-D compute shader on every backend we
/// target, and it divides 1080p exactly.
const WORKGROUP: u32 = 64;

/// Anything that can go wrong setting up or running the GPU path.
#[derive(Debug)]
pub enum GpuError {
    /// No adapter was usable.
    NoAdapter,
    /// Adapter or device creation failed.
    Device(String),
    /// The model could not be packed for the shader.
    Model(String),
    /// A buffer could not be mapped for readback.
    Readback(String),
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no usable GPU adapter"),
            Self::Device(m) => write!(f, "GPU device setup failed: {m}"),
            Self::Model(m) => write!(f, "model cannot be uploaded: {m}"),
            Self::Readback(m) => write!(f, "GPU readback failed: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A live GPU context: device, queue and the compiled pipeline.
///
/// Created once per process and shared. Creating a device is expensive (hundreds of milliseconds on
/// a cold GPU driver) and the pipeline is stateless, so there is nothing per-frame or per-filter
/// about it.
pub struct GpuContext {
    /// Pipeline for the native-sample entry point. Separate from `pipeline` because it takes the
    /// clip's own samples rather than three planar `f32` buffers.
    native_pipeline: wgpu::ComputePipeline,
    /// Group 0 for the native pipeline: the model buffer alone, at the binding the shader uses.
    native_model_layout: wgpu::BindGroupLayout,
    /// Group 1 for the native pipeline: parameters, input words, output words.
    native_io_layout: wgpu::BindGroupLayout,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    info: AdapterInfo,
}

impl GpuContext {
    /// Creates a context on the adapter at `device_id` in the ranked list.
    ///
    /// `device_id` is an index into the same ordering [`enumerate_adapters`] returns, so a caller
    /// that lists the adapters can pick one by position. Out of range is an error rather than a
    /// silent fall back to adapter 0: a caller that asked for a specific GPU and got a different one
    /// has a result it did not ask for.
    fn create(device_id: u32) -> Result<Self, GpuError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

        // Same ranking as `enumerate_adapters`: a discrete GPU beats an integrated one, and a
        // software adapter is not a GPU backend at all.
        let mut usable: Vec<_> = adapters
            .into_iter()
            .filter(|a| rank(a.get_info().device_type) > 0)
            .collect();
        usable.sort_by(|a, b| {
            let (ia, ib) = (a.get_info(), b.get_info());
            rank(ib.device_type)
                .cmp(&rank(ia.device_type))
                .then_with(|| ia.name.cmp(&ib.name))
        });

        let adapter = usable
            .into_iter()
            .nth(device_id as usize)
            .ok_or(GpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_info = AdapterInfo {
            name: info.name.clone(),
            api: backend_label(info.backend).to_owned(),
            device_type: device_label(info.device_type).to_owned(),
            vendor: info.vendor,
            device: info.device,
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("colormatch"),
            // No features or limits beyond the defaults: the shader is plain f32 arithmetic and a
            // storage buffer, which every backend supports.
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(format!("{e:?}")))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colormatch::apply"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("colormatch::bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("colormatch::layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("colormatch::pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // wgpu requires every module-scope binding to appear in the pipeline layout, even one the
        // entry point does not read. So the native pipeline's group 0 mirrors `main`'s three
        // bindings rather than declaring only the model it actually uses.
        let native_model_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("colormatch::native model"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let native_io_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("colormatch::native io"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let native_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("colormatch::native layout"),
                bind_group_layouts: &[Some(&native_model_layout), Some(&native_io_layout)],
                immediate_size: 0,
            });
        let native_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("colormatch::native pipeline"),
            layout: Some(&native_pipeline_layout),
            module: &module,
            entry_point: Some("main_native"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            native_pipeline,
            native_model_layout,
            native_io_layout,
            info: adapter_info,
        })
    }

    /// The context for adapter `device_id`, created once and cached.
    ///
    /// Cached per index because both the availability check and the apply path need it, and neither
    /// should pay for device creation twice. A failure is cached too, so a machine with no usable
    /// adapter does not retry the enumeration on every frame.
    pub fn shared(device_id: u32) -> Option<Arc<Self>> {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        static CONTEXTS: OnceLock<Mutex<HashMap<u32, Option<Arc<GpuContext>>>>> = OnceLock::new();
        let map = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));

        let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .entry(device_id)
            .or_insert_with(|| match Self::create(device_id) {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(e) => {
                    // Worth knowing about, but not fatal: the caller falls back to a CPU backend and
                    // reports that, so this is a diagnostic rather than an error.
                    eprintln!(
                        "colormatch: GPU adapter {device_id} unavailable, falling back to CPU: {e}"
                    );
                    None
                }
            })
            .clone()
    }

    /// The adapter this context landed on.
    pub fn adapter(&self) -> &AdapterInfo {
        &self.info
    }
}

/// Packs a model into the flat `f32` layout the shader indexes.
///
/// Returns the buffer contents and the offset at which the axes begin.
fn pack_model(model: &Model) -> (Vec<f32>, u32) {
    let k = model.slices.len();
    let nodes = model.nodes as usize;
    let stride = 2 + nodes;
    let mut out = Vec::with_capacity(k * 9 + k * 3 * stride);
    for slice in &model.slices {
        for row in &slice.rot {
            out.extend_from_slice(row);
        }
    }
    let axes_base = out.len() as u32;
    for slice in &model.slices {
        for axis in &slice.axes {
            out.push(axis.lo);
            out.push(axis.scale);
            out.extend_from_slice(&axis.tab[..nodes]);
        }
    }
    (out, axes_base)
}

/// Applies `model` to three planar `f32` buffers on the GPU.
///
/// The buffers are uploaded, transformed and read back, so the round trip is part of the cost. That
/// is deliberate: VapourSynth only hands out GPU-resident frames through its Vulkan interop, which
/// this build has no access to, so the honest thing is to measure the round trip rather than to
/// pretend it away.
///
/// The result is compared against the scalar reference by `tests/gpu_matches_scalar.rs`.
pub fn apply_gpu(
    model: &Model,
    planes: [&mut [f32]; 3],
    options: &crate::apply::ApplyOptions,
    device_id: u32,
) -> Result<(), GpuError> {
    let ctx = GpuContext::shared(device_id).ok_or(GpuError::NoAdapter)?;
    let [pr, pg, pb] = planes;
    let n = pr.len();
    if n == 0 {
        return Ok(());
    }
    if pg.len() != n || pb.len() != n {
        return Err(GpuError::Model(format!(
            "plane lengths differ: {} {} {}",
            n,
            pg.len(),
            pb.len()
        )));
    }
    if model.slices.is_empty() {
        return Err(GpuError::Model("the model has no slices".to_owned()));
    }

    let (packed, axes_base) = pack_model(model);
    let (k, nodes) = (model.slices.len() as u32, model.nodes);

    let gamut = match options.gamut {
        crate::apply::Gamut::None => 0u32,
        crate::apply::Gamut::Clip => 1,
        crate::apply::Gamut::Soft => 2,
    };
    // The uniform block is 32 bytes; the shader's struct is 8 f32-equivalents, so this matches.
    let params: [f32; 8] = [
        f32::from_bits(k),
        f32::from_bits(nodes),
        f32::from_bits(gamut),
        f32::from_bits(n as u32),
        f32::from_bits(axes_base),
        options.luma_lock,
        options.clamp_lo,
        options.clamp_hi,
    ];
    let params_bytes = bytemuck::cast_slice(&params);
    let model_bytes = bytemuck::cast_slice(&packed);

    let storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::planes"),
        size: (n * 3 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // A readback buffer is separate because WebGPU forbids combining `MAP_READ` with `STORAGE`.
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::readback"),
        size: (n * 3 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Upload the three planes. They are uploaded as they are, because the sanitising and clamping
    // belong in the shader where the CPU does them.
    let plane_bytes = n * 4;
    for (i, plane) in [&pr, &pg, &pb].into_iter().enumerate() {
        ctx.queue.write_buffer(
            &storage,
            (i * plane_bytes) as u64,
            bytemuck::cast_slice(plane),
        );
    }
    let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::params"),
        size: params_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&params_buf, 0, params_bytes);

    let model_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::model"),
        size: model_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&model_buf, 0, model_bytes);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("colormatch::group"),
        layout: &ctx.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: model_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: storage.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("colormatch::encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("colormatch::pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, (n * 3 * 4) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| GpuError::Readback(format!("{e:?}")))?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(GpuError::Readback(format!("{e:?}"))),
        Err(e) => return Err(GpuError::Readback(format!("{e}"))),
    }

    {
        let data = slice
            .get_mapped_range()
            .map_err(|e| GpuError::Readback(format!("{e:?}")))?;
        let all: &[f32] = bytemuck::cast_slice(&data[..]);
        pr.copy_from_slice(&all[..n]);
        pg.copy_from_slice(&all[n..2 * n]);
        pb.copy_from_slice(&all[2 * n..3 * n]);
    }
    readback.unmap();
    Ok(())
}

/// Whether the GPU **apply** path can actually run on adapter `device_id`.
///
/// This is a real check, not a feature flag: it creates the context once and reports whether that
/// succeeded. The plugin's contract is that the reported backend is the one that ran, so a build with
/// the `gpu` feature compiled in but no usable adapter — or a caller asking for adapter 3 on a
/// one-GPU machine — must not advertise a GPU backend.
pub fn available(device_id: u32) -> bool {
    GpuContext::shared(device_id).is_some()
}

// ---------------------------------------------------------------------------------------------
// Native-sample path
// ---------------------------------------------------------------------------------------------

/// How a plane's samples are packed in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSample {
    /// One byte per sample, four per `u32` word.
    U8,
    /// Two bytes per sample, two per `u32` word.
    U16,
    /// One `f32` per word, bitcast.
    F32,
}

/// Which planes a clip has and how they relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeFamily {
    /// Three full-resolution planes.
    Rgb,
    /// One plane, replicated into RGB and collapsed back.
    Gray,
    /// Luma plus two chroma planes, which may be subsampled.
    Yuv,
}

/// Everything the native shader needs beyond the model itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeLayout {
    /// How samples are packed.
    pub sample: NativeSample,
    /// Which planes the clip has.
    pub family: NativeFamily,
    /// Luma width; with `height`, also the dispatch size.
    pub width: u32,
    /// Luma height.
    pub height: u32,
    /// Chroma width, equal to `width` when not subsampled.
    pub cw: u32,
    /// Chroma height, equal to `height` when not subsampled.
    pub ch: u32,
    /// Horizontal chroma subsampling factor, 1 or 2.
    pub sub_h: u32,
    /// Vertical chroma subsampling factor, 1 or 2.
    pub sub_v: u32,
    /// Normalisation constants for the luma plane, and for RGB's three planes.
    pub luma: crate::color::PlaneScale,
    /// Normalisation constants for the chroma planes.
    pub chroma: crate::color::PlaneScale,
    /// Luma coefficient `Kr` for the YUV matrix.
    pub kr: f32,
    /// Luma coefficient `Kb` for the YUV matrix.
    pub kb: f32,
}

impl NativeLayout {
    /// Samples in one plane.
    fn plane_len(&self, plane: usize) -> usize {
        match (self.family, plane) {
            (NativeFamily::Gray, 0) => (self.width * self.height) as usize,
            (NativeFamily::Yuv, 0) | (NativeFamily::Rgb, _) => (self.width * self.height) as usize,
            (NativeFamily::Yuv, _) => (self.cw * self.ch) as usize,
            (NativeFamily::Gray, _) => 0,
        }
    }

    /// `u32` words one plane occupies.
    fn plane_words(&self, plane: usize) -> usize {
        let len = self.plane_len(plane);
        match self.sample {
            NativeSample::U8 => len.div_ceil(4),
            NativeSample::U16 => len.div_ceil(2),
            NativeSample::F32 => len,
        }
    }

    /// Number of planes actually present.
    fn plane_count(&self) -> usize {
        match self.family {
            NativeFamily::Gray => 1,
            _ => 3,
        }
    }
}

/// Applies a model to native samples, converting on the GPU.
///
/// This is the route the plugin takes for any layout the shader implements. `tests/gpu_native_matches_cpu.rs`
/// holds it to the CPU reference across every such layout, and requires an exact no-op for an
/// identity model.
///
/// `planes` are `(input, output)` byte slices in the clip's own sample format; the output slice is
/// overwritten. This replaces the host-side round trip through three planar `f32` buffers, which was
/// the largest fixed cost of the GPU path.
///
/// Correctness is not taken on trust: `tests/gpu_native_matches_cpu.rs` runs every supported format
/// against the CPU path and requires agreement to within one 16-bit LSB, and an exact no-op for an
/// identity model.
pub fn apply_gpu_native(
    model: &Model,
    mut planes: [(&[u8], &mut [u8]); 3],
    layout: &NativeLayout,
    options: &crate::apply::ApplyOptions,
    device_id: u32,
) -> Result<(), GpuError> {
    let ctx = GpuContext::shared(device_id).ok_or(GpuError::NoAdapter)?;
    if model.slices.is_empty() {
        return Err(GpuError::Model("the model has no slices".to_owned()));
    }
    let count = layout.plane_count();
    for (i, (input, output)) in planes.iter().enumerate() {
        if i < count && input.len() != output.len() {
            return Err(GpuError::Model(format!(
                "plane {i}: input is {} bytes, output is {} bytes",
                input.len(),
                output.len()
            )));
        }
    }

    let (packed, axes_base) = pack_model(model);

    // Word offsets for each plane, in the input and the output buffers.
    let words: Vec<usize> = (0..count).map(|p| layout.plane_words(p)).collect();
    let mut offsets = [0u32; 3];
    let mut cursor = 0usize;
    for p in 0..count {
        offsets[p] = cursor as u32;
        cursor += words[p];
    }
    let total_words = cursor.max(1);
    let total_bytes = (total_words * 4) as u64;

    let to_bits = |v: f32| v.to_bits();
    let sample_code = match layout.sample {
        NativeSample::U8 => 0u32,
        NativeSample::U16 => 1,
        NativeSample::F32 => 2,
    };
    let family_code = match layout.family {
        NativeFamily::Rgb => 0u32,
        NativeFamily::Gray => 1,
        NativeFamily::Yuv => 2,
    };
    let gamut = match options.gamut {
        crate::apply::Gamut::None => 0u32,
        crate::apply::Gamut::Clip => 1,
        crate::apply::Gamut::Soft => 2,
    };

    // 44 `u32` slots, matching `NativeParams` in the shader field for field.
    let mut params = [0u32; 44];
    params[0] = model.slices.len() as u32;
    params[1] = model.nodes;
    params[2] = gamut;
    params[3] = layout.width;
    params[4] = layout.height;
    params[5] = axes_base;
    params[6] = to_bits(options.luma_lock);
    params[7] = to_bits(options.clamp_lo);
    params[8] = to_bits(options.clamp_hi);
    params[9] = layout.cw;
    params[10] = layout.ch;
    params[11] = family_code;
    params[12] = sample_code;
    params[13] = layout.sub_h;
    params[14] = layout.sub_v;
    // 15 is padding.
    params[16] = offsets[0];
    params[17] = offsets[1];
    params[18] = offsets[2];
    params[20] = offsets[0];
    params[21] = offsets[1];
    params[22] = offsets[2];
    let l = &layout.luma;
    let c = &layout.chroma;
    params[24] = to_bits(l.norm_scale);
    params[25] = to_bits(l.norm_offset);
    params[26] = to_bits(l.norm_lo);
    params[27] = to_bits(l.norm_hi);
    params[28] = to_bits(c.norm_scale);
    params[29] = to_bits(c.norm_offset);
    params[30] = to_bits(c.norm_lo);
    params[31] = to_bits(c.norm_hi);
    params[32] = to_bits(l.denorm_scale);
    params[33] = to_bits(l.denorm_offset);
    params[34] = to_bits(l.peak);
    params[36] = to_bits(c.denorm_scale);
    params[37] = to_bits(c.denorm_offset);
    params[38] = to_bits(c.peak);
    params[40] = to_bits(layout.kr);
    params[41] = to_bits(layout.kb);

    let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra,
            mapped_at_creation: false,
        })
    };
    let input = storage(
        "colormatch::native in",
        total_bytes,
        wgpu::BufferUsages::empty(),
    );
    let output = storage(
        "colormatch::native out",
        total_bytes,
        wgpu::BufferUsages::empty(),
    );
    let dummy = storage("colormatch::native dummy", 4, wgpu::BufferUsages::empty());
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::native readback"),
        size: total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Upload each plane's packed words. The caller's bytes are copied into a word-aligned staging
    // buffer first, because a plane whose length is not a multiple of the word size would otherwise
    // leave the trailing bytes of its last word uninitialised.
    let mut staging = vec![0u8; total_words * 4];
    for p in 0..count {
        let (input_bytes, _) = planes[p];
        let start = offsets[p] as usize * 4;
        staging[start..start + input_bytes.len()].copy_from_slice(input_bytes);
    }
    ctx.queue.write_buffer(&input, 0, &staging);

    let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::native params"),
        size: (params.len() * 4) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let model_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colormatch::native model"),
        size: bytemuck::cast_slice::<f32, u8>(&packed).len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&model_buf, 0, bytemuck::cast_slice(&packed));

    let group0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("colormatch::native group0"),
        layout: &ctx.native_model_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: model_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dummy.as_entire_binding(),
            },
        ],
    });
    let group1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("colormatch::native group1"),
        layout: &ctx.native_io_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("colormatch::native encoder"),
        });
    // The shader accumulates u8 and u16 samples with atomics, which is race-free only because the
    // buffer starts at zero.
    encoder.clear_buffer(&output, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("colormatch::native pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.native_pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.dispatch_workgroups((layout.width * layout.height).div_ceil(WORKGROUP), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, total_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| GpuError::Readback(format!("{e:?}")))?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(GpuError::Readback(format!("{e:?}"))),
        Err(e) => return Err(GpuError::Readback(format!("{e}"))),
    }

    {
        let data = slice
            .get_mapped_range()
            .map_err(|e| GpuError::Readback(format!("{e:?}")))?;
        for p in 0..count {
            let (_, output_bytes) = &mut planes[p];
            let start = offsets[p] as usize * 4;
            output_bytes.copy_from_slice(&data[start..start + output_bytes.len()]);
        }
    }
    readback.unmap();
    Ok(())
}
