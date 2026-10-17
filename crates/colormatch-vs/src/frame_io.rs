//! Frame ↔ planar `f32` conversion, including chroma resampling.
//!
//! All inputs are converted into a single internal RGB working space before either sampling (build)
//! or transforming (apply). Using the *same* conversion on both paths is a correctness requirement:
//! building in one space and applying in another silently degrades the match instead of failing.
//!
//! # Subsampled chroma
//!
//! The transform mixes R, G and B *at one spatial location*, so it needs co-sited full-resolution
//! planes. 4:2:0 and 4:2:2 input is therefore upsampled to 4:4:4, transformed, and downsampled back
//! so that the output format matches the input exactly. That is what every other 3D-LUT tool does,
//! and there is no way around it: a 3D transform is not separable in the chroma sense.
//!
//! Chroma sample `(i, j)` is taken to be co-sited with luma `(2i, 2j)` — the "left"/"top-left"
//! convention. Bilinear upsampling evaluated at a grid knot returns the knot exactly, so
//! [`ChromaMode::Nearest`] downsampling makes the round trip **exact**: an identity model is a true
//! no-op even on 4:2:0. [`ChromaMode::Box`] averages the covered block instead, which resamples more
//! correctly but breaks that exactness. The default is `Nearest`, because being able to verify a
//! pipeline by feeding it the same clip twice is worth more than a second-order difference in chroma
//! reconstruction.

use anyhow::{Result, bail};
use colormatch_core::color::{self, ColorError, PlaneKind, SpaceInfo};
use vapoursynth::format::{ColorFamily, SampleType};
use vapoursynth::frame::{Frame, FrameRefMut};
use vapoursynth::video_info::{Property, VideoInfo};

/// How full-resolution chroma is reduced back to the clip's chroma grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaMode {
    /// Point-sample the co-sited position. Makes an identity model an exact no-op.
    #[default]
    Nearest,
    /// Average the covered block. Better reconstruction, not identity-exact.
    Box,
}

impl ChromaMode {
    /// Parses the user-facing string form.
    pub fn parse(spec: &str, func: &str) -> Result<Self> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "" | "nearest" | "point" => Ok(Self::Nearest),
            "box" | "average" | "mean" => Ok(Self::Box),
            other => bail!("{func}: `chroma` must be \"nearest\" or \"box\", got {other:?}"),
        }
    }
}

/// Everything needed to interpret a clip's samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipFormat {
    /// Colour family.
    pub family: ColorFamily,
    /// Integer or float samples.
    pub sample_type: SampleType,
    /// Bits per sample.
    pub bits: u8,
    /// Bytes per sample.
    pub bytes: u8,
    /// Whether samples are full range.
    pub full_range: bool,
    /// Matrix coefficients, resolved (never `UNSPECIFIED`).
    pub matrix: u8,
    /// Number of planes.
    pub planes: usize,
    /// Chroma subsampling, horizontal.
    pub sub_w: u8,
    /// Chroma subsampling, vertical.
    pub sub_h: u8,
    /// Luma width.
    pub width: usize,
    /// Luma height.
    pub height: usize,
}

/// Resolves the matrix coefficient.
///
/// The plugin never guesses a matrix for YUV input: an unresolved matrix is a hard error, because
/// building a model in one space and applying it in another degrades the match silently. The Python
/// wrapper reads `_Matrix` from the clip and passes it explicitly.
fn resolve_matrix(explicit: Option<i64>, family: ColorFamily, func: &str) -> Result<u8> {
    if let Some(m) = explicit {
        if !(0..=255).contains(&m) {
            bail!("{func}: `matrix` must be in 0..=255, got {m}");
        }
        return Ok(m as u8);
    }
    match family {
        ColorFamily::RGB | ColorFamily::Gray => Ok(color::matrix::RGB),
        _ => bail!(
            "{func}: YUV input needs an explicit `matrix` (0=rgb, 1=bt709, 5=bt470bg, 6=smpte170m, \
             9=bt2020nc); pass the clip's `_Matrix` property value. The `vs_colormatch` wrapper \
             does this automatically"
        ),
    }
}

/// Resolves the colour range from an explicit argument, falling back to VapourSynth's documented
/// default for the colour family (full range for RGB/GRAY, limited for YUV).
fn resolve_range(explicit: Option<i64>, family: ColorFamily) -> u8 {
    if let Some(r) = explicit {
        return if r == 0 {
            color::range::LIMITED
        } else {
            color::range::FULL
        };
    }
    match family {
        ColorFamily::RGB | ColorFamily::Gray => color::range::FULL,
        _ => color::range::LIMITED,
    }
}

impl ClipFormat {
    /// Builds a [`ClipFormat`] from a clip's [`VideoInfo`].
    ///
    /// Resolved once, at filter creation, so that the per-frame path carries no validation cost and
    /// so that a mismatch is reported before any frames are processed.
    pub fn from_info(
        info: &VideoInfo,
        explicit_matrix: Option<i64>,
        explicit_range: Option<i64>,
        func: &str,
    ) -> Result<Self> {
        let fmt = info.format;
        let family = fmt.color_family();
        if family == ColorFamily::Undefined {
            bail!(
                "{func}: variable-format clips are not supported; make the format constant first"
            );
        }
        if info.num_frames == 0 {
            bail!("{func}: the clip has no frames");
        }
        let (width, height) = match info.resolution {
            Property::Constant(r) => (r.width, r.height),
            Property::Variable => bail!("{func}: variable-resolution clips are not supported"),
        };
        if width == 0 || height == 0 {
            bail!("{func}: the clip has zero dimensions ({width}x{height})");
        }

        let sample_type = fmt.sample_type();
        let bits = fmt.bits_per_sample() as u32;
        if sample_type == SampleType::Integer && !matches!(bits, 8 | 10 | 12 | 14 | 16) {
            bail!("{func}: unsupported integer bit depth {bits}; use 8, 10, 12, 14 or 16");
        }
        if sample_type == SampleType::Float && bits != 32 {
            bail!("{func}: only 32-bit float samples are supported, got {bits}-bit");
        }

        let matrix = resolve_matrix(explicit_matrix, family, func)?;
        let full_range = resolve_range(explicit_range, family) == color::range::FULL;
        SpaceInfo {
            matrix,
            range: u8::from(full_range),
            transfer: 0,
            primaries: 0,
        }
        .validate()
        .map_err(|e| anyhow::anyhow!("{func}: {e}"))?;

        Ok(Self {
            family,
            sample_type,
            bits: bits as u8,
            bytes: fmt.bytes_per_sample(),
            full_range,
            matrix,
            planes: fmt.plane_count(),
            sub_w: fmt.sub_sampling_w(),
            sub_h: fmt.sub_sampling_h(),
            width,
            height,
        })
    }

    /// Returns `true` when the clip is grayscale.
    pub fn is_gray(&self) -> bool {
        self.family == ColorFamily::Gray
    }

    /// Returns `true` when the clip is RGB.
    pub fn is_rgb(&self) -> bool {
        self.family == ColorFamily::RGB
    }

    /// Returns `true` when the chroma planes are smaller than the luma plane.
    pub fn is_subsampled(&self) -> bool {
        self.family == ColorFamily::YUV && (self.sub_w != 0 || self.sub_h != 0)
    }

    /// Chroma plane dimensions, derived the way VapourSynth does.
    pub fn chroma_dims(&self) -> (usize, usize) {
        let w = self.width >> self.sub_w;
        let h = self.height >> self.sub_h;
        (w.max(1), h.max(1))
    }

    /// Kr/Kb luma coefficients for this format's matrix.
    pub fn luma(&self) -> Result<(f32, f32), ColorError> {
        color::luma_coefficients(self.matrix)
    }

    /// Whether a plane carries luma or chroma, which decides its range normalisation.
    pub fn plane_kind(&self, plane: usize) -> PlaneKind {
        if self.family == ColorFamily::YUV && plane > 0 {
            PlaneKind::Chroma
        } else {
            PlaneKind::Luma
        }
    }
}

/// Reads one plane into normalised `f32` in `[0, 1]` (full range) or studio swing.
pub fn read_plane(frame: &Frame, plane: usize, fmt: &ClipFormat) -> Vec<f32> {
    let h = frame.height(plane);
    let w = frame.width(plane);
    let kind = fmt.plane_kind(plane);
    let mut out = Vec::with_capacity(w * h);
    match (fmt.sample_type, fmt.bytes) {
        (SampleType::Float, 4) => {
            for row in 0..h {
                out.extend_from_slice(frame.plane_row::<f32>(plane, row));
            }
        }
        (SampleType::Integer, 1) => {
            for row in 0..h {
                out.extend(
                    frame
                        .plane_row::<u8>(plane, row)
                        .iter()
                        .map(|&v| color::normalize_sample(v as f32, 8, fmt.full_range, kind)),
                );
            }
        }
        (SampleType::Integer, 2) => {
            for row in 0..h {
                out.extend(frame.plane_row::<u16>(plane, row).iter().map(|&v| {
                    color::normalize_sample(v as f32, fmt.bits as u32, fmt.full_range, kind)
                }));
            }
        }
        (SampleType::Integer, 4) => {
            for row in 0..h {
                out.extend(frame.plane_row::<u32>(plane, row).iter().map(|&v| {
                    color::normalize_sample(v as f32, fmt.bits as u32, fmt.full_range, kind)
                }));
            }
        }
        _ => unreachable!("unsupported sample layout guarded in ClipFormat::from_info"),
    }
    out
}

/// Writes normalised `f32` back into one plane.
pub fn write_plane(frame: &mut FrameRefMut, plane: usize, fmt: &ClipFormat, data: &[f32]) {
    let h = frame.height(plane);
    let w = frame.width(plane);
    debug_assert_eq!(data.len(), w * h);
    let bits = fmt.bits as u32;
    let kind = fmt.plane_kind(plane);
    match (fmt.sample_type, fmt.bytes) {
        (SampleType::Float, 4) => {
            for row in 0..h {
                let dst = frame.plane_row_mut::<f32>(plane, row);
                dst.copy_from_slice(&data[row * w..(row + 1) * w]);
            }
        }
        (SampleType::Integer, 1) => {
            for row in 0..h {
                let dst = frame.plane_row_mut::<u8>(plane, row);
                for (d, &v) in dst.iter_mut().zip(&data[row * w..(row + 1) * w]) {
                    *d = color::denormalize_sample(v, 8, fmt.full_range, kind) as u8;
                }
            }
        }
        (SampleType::Integer, 2) => {
            for row in 0..h {
                let dst = frame.plane_row_mut::<u16>(plane, row);
                for (d, &v) in dst.iter_mut().zip(&data[row * w..(row + 1) * w]) {
                    *d = color::denormalize_sample(v, bits, fmt.full_range, kind) as u16;
                }
            }
        }
        (SampleType::Integer, 4) => {
            for row in 0..h {
                let dst = frame.plane_row_mut::<u32>(plane, row);
                for (d, &v) in dst.iter_mut().zip(&data[row * w..(row + 1) * w]) {
                    *d = color::denormalize_sample(v, bits, fmt.full_range, kind) as u32;
                }
            }
        }
        _ => unreachable!("unsupported sample layout guarded in ClipFormat::from_info"),
    }
}

/// Bilinear chroma upsampling onto the luma grid.
///
/// Chroma sample `(i, j)` is treated as co-sited with luma `(2i, 2j)`, so the mapping from luma
/// coordinate to chroma coordinate is `x * (src_w / dst_w)`. Evaluated at a knot this returns the
/// knot's value exactly, which is what makes the nearest-downsample round trip lossless.
pub fn upsample_plane(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; dst_w * dst_h];
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;

    // Left/top-left co-siting: luma `x` maps to chroma coordinate `x * sx`, with no half-pixel
    // offset. That is what makes even luma coordinates land exactly on a chroma knot, and therefore
    // what makes the nearest-downsample round trip lossless. A centre-aligned mapping (`x*sx + ...`)
    // would offset every sample by a quarter pixel and lose that property.
    for y in 0..dst_h {
        let fy = y as f32 * sy;
        let y0f = fy.floor().min(src_h as f32 - 1.0);
        let y1f = (y0f + 1.0).min(src_h as f32 - 1.0);
        let ty = (fy - y0f).clamp(0.0, 1.0);
        let (y0, y1) = (y0f as usize, y1f as usize);
        let row = y * dst_w;
        for x in 0..dst_w {
            let fx = x as f32 * sx;
            let x0f = fx.floor().min(src_w as f32 - 1.0);
            let x1f = (x0f + 1.0).min(src_w as f32 - 1.0);
            let tx = (fx - x0f).clamp(0.0, 1.0);
            let (x0, x1) = (x0f as usize, x1f as usize);

            let top = src[y0 * src_w + x0] + (src[y0 * src_w + x1] - src[y0 * src_w + x0]) * tx;
            let bot = src[y1 * src_w + x0] + (src[y1 * src_w + x1] - src[y1 * src_w + x0]) * tx;
            out[row + x] = top + (bot - top) * ty;
        }
    }
    out
}

/// Reduces full-resolution chroma back onto the clip's chroma grid.
pub fn downsample_plane(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mode: ChromaMode,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; dst_w * dst_h];
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            // The co-sited source position: chroma sample `x` is co-sited with luma `x * sx`, and
            // covers the `sx` by `sy` block starting there.
            let bx = ((x as f32 * sx).round() as usize).min(src_w - 1);
            let by = ((y as f32 * sy).round() as usize).min(src_h - 1);

            out[y * dst_w + x] = match mode {
                ChromaMode::Nearest => src[by * src_w + bx],
                ChromaMode::Box => {
                    let x_end = ((bx as f32 + sx).round() as usize).clamp(bx + 1, src_w);
                    let y_end = ((by as f32 + sy).round() as usize).clamp(by + 1, src_h);
                    let mut acc = 0.0f32;
                    let mut n = 0usize;
                    for yy in by..y_end {
                        for xx in bx..x_end {
                            acc += src[yy * src_w + xx];
                            n += 1;
                        }
                    }
                    acc / n as f32
                }
            };
        }
    }
    out
}

/// Converts a frame into three full-resolution planar `f32` RGB buffers.
///
/// Grayscale is replicated across R/G/B so the model sees a neutral triple; the caller converts back
/// to a single plane afterwards. Subsampled chroma is upsampled bilinearly.
pub fn frame_to_rgb(frame: &Frame, fmt: &ClipFormat) -> Result<[Vec<f32>; 3]> {
    if fmt.is_rgb() {
        return Ok([
            read_plane(frame, 0, fmt),
            read_plane(frame, 1, fmt),
            read_plane(frame, 2, fmt),
        ]);
    }
    if fmt.is_gray() {
        let y = read_plane(frame, 0, fmt);
        return Ok([y.clone(), y.clone(), y]);
    }

    let y = read_plane(frame, 0, fmt);
    let (cw, ch) = fmt.chroma_dims();
    let raw_u = read_plane(frame, 1, fmt);
    let raw_v = read_plane(frame, 2, fmt);
    let u = if fmt.is_subsampled() {
        upsample_plane(&raw_u, cw, ch, fmt.width, fmt.height)
    } else {
        raw_u
    };
    let v = if fmt.is_subsampled() {
        upsample_plane(&raw_v, cw, ch, fmt.width, fmt.height)
    } else {
        raw_v
    };

    let (kr, kb) = fmt.luma().map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = y.len();
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    // Three parallel planes are indexed by the same position; iterating one of them would obscure
    // that the other two are stepped in lockstep.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let rgb = color::yuv_to_rgb(y[i], u[i], v[i], kr, kb);
        r.push(rgb[0]);
        g.push(rgb[1]);
        b.push(rgb[2]);
    }
    Ok([r, g, b])
}

/// Writes three full-resolution planar `f32` RGB buffers back into a frame.
pub fn rgb_to_frame(
    frame: &mut FrameRefMut,
    fmt: &ClipFormat,
    rgb: [&[f32]; 3],
    chroma: ChromaMode,
) -> Result<()> {
    if fmt.is_rgb() {
        write_plane(frame, 0, fmt, rgb[0]);
        write_plane(frame, 1, fmt, rgb[1]);
        write_plane(frame, 2, fmt, rgb[2]);
        return Ok(());
    }
    if fmt.is_gray() {
        // Collapse to luma so a grayscale clip stays grayscale.
        let (kr, kb) = fmt.luma().unwrap_or((0.2126, 0.0722));
        let kg = 1.0 - kr - kb;
        let out: Vec<f32> = (0..rgb[0].len())
            .map(|i| kr * rgb[0][i] + kg * rgb[1][i] + kb * rgb[2][i])
            .collect();
        write_plane(frame, 0, fmt, &out);
        return Ok(());
    }

    let (kr, kb) = fmt.luma().map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = rgb[0].len();
    let mut y = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    // Three parallel planes are indexed by the same position; iterating one of them would obscure
    // that the other two are stepped in lockstep.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let t = color::rgb_to_yuv(rgb[0][i], rgb[1][i], rgb[2][i], kr, kb);
        y.push(t[0]);
        u.push(t[1]);
        v.push(t[2]);
    }
    write_plane(frame, 0, fmt, &y);

    if fmt.is_subsampled() {
        let (cw, ch) = fmt.chroma_dims();
        let du = downsample_plane(&u, fmt.width, fmt.height, cw, ch, chroma);
        let dv = downsample_plane(&v, fmt.width, fmt.height, cw, ch, chroma);
        write_plane(frame, 1, fmt, &du);
        write_plane(frame, 2, fmt, &dv);
    } else {
        write_plane(frame, 1, fmt, &u);
        write_plane(frame, 2, fmt, &v);
    }
    Ok(())
}

/// Builds the descriptor the GPU native path needs from a `ClipFormat`.
///
/// `None` when the layout is outside what that path implements — currently only a subsampling factor
/// other than 1 or 2, which VapourSynth does not produce anyway. The caller falls back to the planar
/// `f32` route rather than guessing.
#[cfg(feature = "gpu")]
pub fn native_layout(fmt: &ClipFormat) -> Option<colormatch_core::backend::gpu::NativeLayout> {
    use colormatch_core::backend::gpu::{NativeFamily, NativeLayout, NativeSample};
    use colormatch_core::color::plane_scale;

    let sample = match (fmt.sample_type, fmt.bytes) {
        (SampleType::Integer, 1) => NativeSample::U8,
        (SampleType::Integer, 2) => NativeSample::U16,
        (SampleType::Float, 4) => NativeSample::F32,
        _ => return None,
    };
    if !matches!(fmt.sub_w, 1 | 2) || !matches!(fmt.sub_h, 1 | 2) {
        return None;
    }
    let family = if fmt.is_rgb() {
        NativeFamily::Rgb
    } else if fmt.is_gray() {
        NativeFamily::Gray
    } else {
        NativeFamily::Yuv
    };
    let (kr, kb) = match family {
        // RGB has no matrix, and gray collapses with the luma weights; both need *some* coefficients,
        // and the ones `rgb_to_frame` uses as its fallback are the right ones.
        NativeFamily::Rgb | NativeFamily::Gray => (0.2126f32, 0.0722f32),
        NativeFamily::Yuv => fmt.luma().ok()?,
    };
    let bits = fmt.bits as u32;
    let (cw, ch) = fmt.chroma_dims();
    Some(NativeLayout {
        sample,
        family,
        width: fmt.width as u32,
        height: fmt.height as u32,
        cw: cw as u32,
        ch: ch as u32,
        // `sub_w`/`sub_h` on `ClipFormat` are *shift* counts (1 means halve), because
        // `chroma_dims` shifts by them. The shader wants the factor: 1 or 2.
        sub_h: 1u32 << fmt.sub_w,
        sub_v: 1u32 << fmt.sub_h,
        // RGB normalises its three planes as luma; only YUV has a chroma plane kind.
        luma: plane_scale(bits, fmt.full_range, PlaneKind::Luma),
        chroma: plane_scale(
            bits,
            fmt.full_range,
            if family == NativeFamily::Yuv {
                PlaneKind::Chroma
            } else {
                PlaneKind::Luma
            },
        ),
        kr,
        kb,
    })
}

/// Copies one plane's rows into a packed, contiguous buffer.
///
/// VapourSynth rows carry an unspecified stride, so the plane cannot be handed to the GPU as one
/// slice. This is a memcpy per row and no arithmetic — which is the whole point: it replaces
/// normalise, the YUV matrix and three full-frame `f32` allocations.
#[cfg(feature = "gpu")]
fn pack_plane(frame: &Frame, plane: usize, fmt: &ClipFormat, out: &mut [u8]) {
    let h = frame.height(plane);
    let w = frame.width(plane);
    let bytes = fmt.bytes as usize;
    let row_bytes = w * bytes;
    for row in 0..h {
        let dst = &mut out[row * row_bytes..(row + 1) * row_bytes];
        match (fmt.sample_type, fmt.bytes) {
            (SampleType::Float, 4) => {
                let src = frame.plane_row::<f32>(plane, row);
                for (d, &v) in dst.chunks_exact_mut(4).zip(src) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
            }
            (SampleType::Integer, 1) => {
                let src = frame.plane_row::<u8>(plane, row);
                dst.copy_from_slice(src);
            }
            (SampleType::Integer, 2) => {
                let src = frame.plane_row::<u16>(plane, row);
                for (d, &v) in dst.chunks_exact_mut(2).zip(src) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
            }
            (SampleType::Integer, 4) => {
                let src = frame.plane_row::<u32>(plane, row);
                for (d, &v) in dst.chunks_exact_mut(4).zip(src) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
            }
            _ => unreachable!("unsupported sample layout guarded in ClipFormat::from_info"),
        }
    }
}

/// Writes a packed buffer back into one plane, row by row.
#[cfg(feature = "gpu")]
fn unpack_plane(out: &mut FrameRefMut, plane: usize, fmt: &ClipFormat, data: &[u8]) {
    let h = out.height(plane);
    let w = out.width(plane);
    let bytes = fmt.bytes as usize;
    let row_bytes = w * bytes;
    for row in 0..h {
        let src = &data[row * row_bytes..(row + 1) * row_bytes];
        match (fmt.sample_type, fmt.bytes) {
            (SampleType::Float, 4) => {
                let dst = out.plane_row_mut::<f32>(plane, row);
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            (SampleType::Integer, 1) => {
                out.plane_row_mut::<u8>(plane, row).copy_from_slice(src);
            }
            (SampleType::Integer, 2) => {
                let dst = out.plane_row_mut::<u16>(plane, row);
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                    *d = u16::from_le_bytes([c[0], c[1]]);
                }
            }
            (SampleType::Integer, 4) => {
                let dst = out.plane_row_mut::<u32>(plane, row);
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    *d = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            _ => unreachable!("unsupported sample layout guarded in ClipFormat::from_info"),
        }
    }
}

/// Runs the GPU transform straight from the clip's native samples.
///
/// This is the fused route: no `f32` planes are built on the host, and normalisation, the YUV matrix
/// and chroma resampling all happen in the shader.
#[cfg(feature = "gpu")]
pub fn apply_native_gpu(
    frame: &Frame,
    out: &mut FrameRefMut,
    fmt: &ClipFormat,
    model: &colormatch_core::model::Model,
    options: &colormatch_core::apply::ApplyOptions,
    device_id: u32,
) -> Result<()> {
    let layout = native_layout(fmt)
        .ok_or_else(|| anyhow::anyhow!("unsupported layout for the native GPU path"))?;
    let count = if fmt.is_gray() { 1 } else { 3 };

    // Pack the input planes.
    let mut packed: Vec<Vec<u8>> = Vec::with_capacity(3);
    for plane in 0..count {
        let len = frame.width(plane) * frame.height(plane) * fmt.bytes as usize;
        let mut buf = vec![0u8; len];
        pack_plane(frame, plane, fmt, &mut buf);
        packed.push(buf);
    }
    while packed.len() < 3 {
        packed.push(Vec::new());
    }
    let mut produced: [Vec<u8>; 3] = [
        vec![0u8; packed[0].len()],
        vec![0u8; packed[1].len()],
        vec![0u8; packed[2].len()],
    ];
    {
        let [p0, p1, p2] = &mut produced;
        colormatch_core::backend::gpu::apply_gpu_native(
            model,
            [
                (packed[0].as_slice(), p0.as_mut_slice()),
                (packed[1].as_slice(), p1.as_mut_slice()),
                (packed[2].as_slice(), p2.as_mut_slice()),
            ],
            &layout,
            options,
            device_id,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    for (plane, data) in produced.iter().take(count).enumerate() {
        unpack_plane(out, plane, fmt, data);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_format() -> ClipFormat {
        ClipFormat {
            family: ColorFamily::YUV,
            sample_type: SampleType::Integer,
            bits: 10,
            bytes: 2,
            full_range: false,
            matrix: 1,
            planes: 3,
            sub_w: 0,
            sub_h: 0,
            width: 1920,
            height: 1080,
        }
    }

    #[test]
    fn chroma_dims_match_vapoursynth() {
        let base = base_format();
        assert_eq!(base.chroma_dims(), (1920, 1080));
        assert!(!base.is_subsampled());

        let y420 = ClipFormat {
            sub_w: 1,
            sub_h: 1,
            ..base
        };
        assert_eq!(y420.chroma_dims(), (960, 540));
        assert!(y420.is_subsampled());

        let y422 = ClipFormat {
            sub_w: 1,
            sub_h: 0,
            ..base
        };
        assert_eq!(y422.chroma_dims(), (960, 1080));
        assert!(y422.is_subsampled());

        // Odd luma dimensions must not produce a zero-sized chroma plane.
        let odd = ClipFormat {
            sub_w: 1,
            sub_h: 1,
            width: 3,
            height: 3,
            ..base
        };
        assert_eq!(odd.chroma_dims(), (1, 1));
    }

    #[test]
    fn resampling_is_identity_when_dimensions_match() {
        let src: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        assert_eq!(upsample_plane(&src, 4, 3, 4, 3), src);
        assert_eq!(downsample_plane(&src, 4, 3, 4, 3, ChromaMode::Nearest), src);
        assert_eq!(downsample_plane(&src, 4, 3, 4, 3, ChromaMode::Box), src);
    }

    #[test]
    fn nearest_round_trip_is_exact() {
        // This is the property that makes an identity model a true no-op on 4:2:0.
        let (cw, ch) = (8usize, 6usize);
        let src: Vec<f32> = (0..cw * ch)
            .map(|i| ((i * 37) % 101) as f32 / 100.0)
            .collect();

        for (sw, sh) in [(1u8, 1u8), (1, 0), (0, 1)] {
            let (dw, dh) = (cw << sw, ch << sh);
            let up = upsample_plane(&src, cw, ch, dw, dh);
            let down = downsample_plane(&up, dw, dh, cw, ch, ChromaMode::Nearest);
            for (i, (a, b)) in src.iter().zip(down.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "sub {sw}x{sh} index {i}: {a} -> {b} (round trip is not exact)"
                );
            }
        }
    }

    #[test]
    fn upsampling_at_grid_knots_returns_the_knot() {
        // The mechanism behind the exact round trip.
        let (cw, ch) = (4usize, 4usize);
        let src: Vec<f32> = (0..cw * ch).map(|i| i as f32).collect();
        let up = upsample_plane(&src, cw, ch, cw * 2, ch * 2);
        for y in 0..ch {
            for x in 0..cw {
                let got = up[(y * 2) * (cw * 2) + x * 2];
                assert!((got - src[y * cw + x]).abs() < 1e-6, "at {x},{y}: {got}");
            }
        }
    }

    #[test]
    fn upsampling_interpolates_between_knots() {
        let src = vec![0.0f32, 1.0, 0.0, 1.0];
        let up = upsample_plane(&src, 2, 2, 4, 4);
        let mid = up[1];
        assert!(
            (mid - 0.5).abs() < 1e-5,
            "midpoint should be 0.5, got {mid}"
        );
    }

    #[test]
    fn box_downsampling_averages_the_covered_block() {
        let src = vec![0.0f32, 0.0, 1.0, 1.0];
        let down = downsample_plane(&src, 2, 2, 1, 1, ChromaMode::Box);
        assert!((down[0] - 0.5).abs() < 1e-6, "got {}", down[0]);
        let near = downsample_plane(&src, 2, 2, 1, 1, ChromaMode::Nearest);
        assert!((near[0] - 0.0).abs() < 1e-6, "got {}", near[0]);
    }

    #[test]
    fn chroma_mode_parsing() {
        assert_eq!(ChromaMode::parse("", "t").unwrap(), ChromaMode::Nearest);
        assert_eq!(ChromaMode::parse("box", "t").unwrap(), ChromaMode::Box);
        assert_eq!(ChromaMode::parse("Box", "t").unwrap(), ChromaMode::Box);
        assert!(ChromaMode::parse("bicubic", "t").is_err());
    }
}
