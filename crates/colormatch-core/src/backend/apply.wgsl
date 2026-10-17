// Compute shader for the slice chain.
//
// This is a transliteration of `apply_impl` in `crates/colormatch-core/src/apply.rs`, and it has to
// stay one. The plugin's contract is that the reported backend is the one that ran, and a backend
// that quietly disagrees with the reference is worse than no backend at all.
// `tests/gpu_matches_scalar.rs` compares the two on random models and fails on any divergence beyond
// f32 reassociation.
//
// Everything the CPU does is here in the same order: sanitise the input, run the slices, apply the
// luma lock, then the gamut policy. Getting that order wrong is a silent colour error, not a crash.
//
// Model layout in the storage buffer, all f32:
//
//   [0, k * 9)          rotations, row-major 3x3 per slice
//   [axes_base, ...)    axes, `(2 + nodes)` values each: lo, scale, tab[nodes]
//
// Flattened rather than structs on purpose: WGSL's storage-buffer alignment rules for a struct
// containing a fixed-size array are easy to get subtly wrong, and a hand-rolled index is not.

struct Params {
    k: u32,
    nodes: u32,
    // 0 = none, 1 = clip, 2 = soft. Mirrors `colormatch_core::apply::Gamut`.
    gamut: u32,
    // Number of pixels per plane.
    len: u32,
    axes_base: u32,
    luma_lock: f32,
    clamp_lo: f32,
    clamp_hi: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> model: array<f32>;
@group(0) @binding(2) var<storage, read_write> planes: array<f32>;

const F32_MAX: f32 = 3.402823466e38;
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

/// Clamps a finite input into range and replaces non-finite values with 0.
///
/// Mirrors `sanitize`. The non-finite case matters for float clips: a NaN that reaches the tables
/// would turn an entire frame into NaN, so it is mapped to 0 instead.
fn sanitize(v: f32, lo: f32, hi: f32) -> f32 {
    let finite = (v == v) && (abs(v) <= F32_MAX);
    return select(0.0, clamp(v, lo, hi), finite);
}

/// Evaluates one transfer axis at `v`, matching `Axis::eval`.
///
/// The extrapolation outside the grid is not a detail. The first slice's grid is `[0, 1]` and
/// limited-range YUV converts to RGB outside it, so clamping here instead of extrapolating shifted an
/// identity 4:2:0 model by 5% of full scale. The CPU side learned that the hard way; this must not
/// relearn it.
fn eval_axis(base: u32, v: f32) -> f32 {
    let lo = model[base];
    let scale = model[base + 1u];
    let first = model[base + 2u];
    let last = model[base + 1u + params.nodes];

    if (scale == 0.0) {
        return first;
    }
    let idx = (v - lo) * scale;

    // NaN fails every comparison, so it needs its own test to stay deterministic.
    if (!(idx <= 0.0) && !(idx >= 0.0)) {
        return select(first, last, idx > 0.0);
    }
    if (idx <= 0.0) {
        // Unit slope below the grid: `v` is one grid step per unit of `idx`.
        return first + idx / scale;
    }
    let top = f32(params.nodes - 1u);
    if (idx >= top) {
        return last + (idx - top) / scale;
    }
    let i = u32(idx);
    let f = idx - f32(i);
    let a = model[base + 2u + i];
    return a + f * (model[base + 3u + i] - a);
}

/// Soft knee above `hi`, matching `soft_knee`.
fn soft_knee(v: f32, hi: f32) -> f32 {
    if (v <= hi) {
        return v;
    }
    let over = v - hi;
    return hi + over / (1.0 + over);
}

/// Soft knee below `lo`, matching `soft_knee_lo`.
fn soft_knee_lo(v: f32, lo: f32) -> f32 {
    if (v >= lo) {
        return v;
    }
    let over = lo - v;
    return lo - over / (1.0 + over);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }

    let n = params.len;
    let r0 = sanitize(planes[i], params.clamp_lo, params.clamp_hi);
    let g0 = sanitize(planes[n + i], params.clamp_lo, params.clamp_hi);
    let b0 = sanitize(planes[2u * n + i], params.clamp_lo, params.clamp_hi);
    var rgb = vec3<f32>(r0, g0, b0);

    for (var k = 0u; k < params.k; k = k + 1u) {
        let m = k * 9u;
        let u = vec3<f32>(
            model[m] * rgb.x + model[m + 1u] * rgb.y + model[m + 2u] * rgb.z,
            model[m + 3u] * rgb.x + model[m + 4u] * rgb.y + model[m + 5u] * rgb.z,
            model[m + 6u] * rgb.x + model[m + 7u] * rgb.y + model[m + 8u] * rgb.z,
        );

        let stride = 2u + params.nodes;
        let a0 = params.axes_base + (k * 3u) * stride;
        let y = vec3<f32>(
            eval_axis(a0, u.x),
            eval_axis(a0 + stride, u.y),
            eval_axis(a0 + 2u * stride, u.z),
        );

        // Rotate the correction back. `R` is orthonormal, so `Rᵀ` is the inverse.
        let d = y - u;
        rgb = rgb + vec3<f32>(
            model[m] * d.x + model[m + 3u] * d.y + model[m + 6u] * d.z,
            model[m + 1u] * d.x + model[m + 4u] * d.y + model[m + 7u] * d.z,
            model[m + 2u] * d.x + model[m + 5u] * d.y + model[m + 8u] * d.z,
        );
    }

    // Luma lock comes before the gamut step, and shifts all three channels equally. Doing it after,
    // or multiplicatively, changes the result.
    if (params.luma_lock > 0.0) {
        let y0 = dot(vec3<f32>(r0, g0, b0), LUMA);
        let y1 = dot(rgb, LUMA);
        let locked_y = y0 + (y1 - y0) * (1.0 - params.luma_lock);
        rgb = rgb + vec3<f32>(locked_y - y1);
    }

    if (params.gamut == 1u) {
        rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (params.gamut == 2u) {
        rgb = vec3<f32>(
            soft_knee_lo(soft_knee(rgb.x, 1.0), 0.0),
            soft_knee_lo(soft_knee(rgb.y, 1.0), 0.0),
            soft_knee_lo(soft_knee(rgb.z, 1.0), 0.0),
        );
    }

    planes[i] = rgb.x;
    planes[n + i] = rgb.y;
    planes[2u * n + i] = rgb.z;
}


// ============================================================================================
// Native-sample path
//
// The `main` entry point above takes three planar `f32` buffers. This one takes the clip's *native*
// samples — u8, u16 or f32, packed into `u32` words — and does the format conversion, the YUV
// matrix and the chroma resampling on the GPU. That removes the per-frame f32 plane construction
// from the host, which was the largest fixed cost of the GPU path.
//
// It is a second entry point rather than a replacement so the working path stays working: if this
// one is wrong, the f32 path is still there and still verified.
//
// Layout of one plane in the packed buffer, all samples in native order:
//   u8  : four samples per u32, little-endian within the word (sample 0 in the low byte)
//   u16 : two samples per u32, little-endian within the word
//   f32 : one sample per u32, bitcast
// Each plane starts at a word offset, so a plane's samples never straddle a word boundary.
// ============================================================================================

struct NativeParams {
    k: u32,
    nodes: u32,
    gamut: u32,
    // Luma pixels, which is the dispatch size: one invocation per luma pixel.
    width: u32,
    height: u32,
    axes_base: u32,
    luma_lock: f32,
    clamp_lo: f32,
    clamp_hi: f32,

    cw: u32,
    ch: u32,
    // 0 = rgb, 1 = gray, 2 = yuv
    family: u32,
    // 0 = u8, 1 = u16, 2 = f32
    sample_type: u32,
    sub_h: u32,
    sub_v: u32,
    _pad0: u32,

    in0: u32, in1: u32, in2: u32, _pad1: u32,
    out0: u32, out1: u32, out2: u32, _pad2: u32,

    // normalise: normalised = clamp(raw * n_scale + n_offset, n_clamp_lo, n_clamp_hi)
    n_luma_scale: f32, n_luma_offset: f32, n_luma_lo: f32, n_luma_hi: f32,
    n_chroma_scale: f32, n_chroma_offset: f32, n_chroma_lo: f32, n_chroma_hi: f32,

    // denormalise: raw = round(clamp(v * d_scale + d_offset, 0, d_peak))
    d_luma_scale: f32, d_luma_offset: f32, d_luma_peak: f32, _pad3: f32,
    d_chroma_scale: f32, d_chroma_offset: f32, d_chroma_peak: f32, _pad4: f32,

    kr: f32, kb: f32, _pad5: u32, _pad6: u32,
};

@group(1) @binding(0) var<uniform> nparams: NativeParams;
@group(1) @binding(1) var<storage, read> nin: array<u32>;
@group(1) @binding(2) var<storage, read_write> nout: array<atomic<u32>>;

/// Loads one native sample as its raw numeric value.
fn load_native(base: u32, idx: u32) -> f32 {
    if (nparams.sample_type == 0u) {
        let w = nin[base + (idx >> 2u)];
        return f32((w >> ((idx & 3u) * 8u)) & 0xFFu);
    }
    if (nparams.sample_type == 1u) {
        let w = nin[base + (idx >> 1u)];
        return f32((w >> ((idx & 1u) * 16u)) & 0xFFFFu);
    }
    return bitcast<f32>(nin[base + idx]);
}

/// Writes one native sample from an already-denormalised, rounded value.
///
/// u8 and u16 pack several samples per word, so a plain store would race with the neighbouring
/// invocation. The output buffer is cleared first and each sample owns distinct bits, so an atomic
/// OR is both race-free and cheaper than a read-modify-write.
fn store_native(base: u32, idx: u32, v: f32) {
    if (nparams.sample_type == 2u) {
        atomicStore(&nout[base + idx], bitcast<u32>(v));
        return;
    }
    let bits = u32(v);
    if (nparams.sample_type == 0u) {
        let shift = (idx & 3u) * 8u;
        atomicOr(&nout[base + (idx >> 2u)], (bits & 0xFFu) << shift);
    } else {
        let shift = (idx & 1u) * 16u;
        atomicOr(&nout[base + (idx >> 1u)], (bits & 0xFFFFu) << shift);
    }
}

/// Bilinear chroma fetch at luma position `(x, y)`, mirroring `upsample_plane`.
///
/// The mapping is left/top-left co-sited — luma `x` maps to chroma `x * cw / width` — which is what
/// makes an even luma coordinate land exactly on a chroma knot and the nearest-downsample round trip
/// lossless. A centre-aligned mapping would offset every sample by a quarter pixel and break it.
fn chroma_at(base: u32, x: u32, y: u32) -> f32 {
    let sx = f32(nparams.cw) / f32(nparams.width);
    let sy = f32(nparams.ch) / f32(nparams.height);
    let cw = f32(nparams.cw);
    let ch = f32(nparams.ch);

    let fy = f32(y) * sy;
    let y0f = min(floor(fy), ch - 1.0);
    let y1f = min(y0f + 1.0, ch - 1.0);
    let ty = clamp(fy - y0f, 0.0, 1.0);

    let fx = f32(x) * sx;
    let x0f = min(floor(fx), cw - 1.0);
    let x1f = min(x0f + 1.0, cw - 1.0);
    let tx = clamp(fx - x0f, 0.0, 1.0);

    let x0 = u32(x0f);
    let x1 = u32(x1f);
    let y0 = u32(y0f);
    let y1 = u32(y1f);
    let w = nparams.cw;

    let tl = load_native(base, y0 * w + x0);
    let tr = load_native(base, y0 * w + x1);
    let bl = load_native(base, y1 * w + x0);
    let br = load_native(base, y1 * w + x1);
    let top = tl + (tr - tl) * tx;
    let bot = bl + (br - bl) * tx;
    return top + (bot - top) * ty;
}

fn normalise_luma(raw: f32) -> f32 {
    return clamp(raw * nparams.n_luma_scale + nparams.n_luma_offset,
                 nparams.n_luma_lo, nparams.n_luma_hi);
}
fn normalise_chroma(raw: f32) -> f32 {
    return clamp(raw * nparams.n_chroma_scale + nparams.n_chroma_offset,
                 nparams.n_chroma_lo, nparams.n_chroma_hi);
}
fn denormalise(v: f32, scale: f32, offset: f32, peak: f32) -> f32 {
    return round(clamp(v * scale + offset, 0.0, peak));
}

@compute @workgroup_size(64)
fn main_native(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = nparams.width * nparams.height;
    if (i >= total) {
        return;
    }
    let x = i % nparams.width;
    let y = i / nparams.width;

    // --- load and normalise ---
    var raw_y = 0.0;
    var raw_u = 0.0;
    var raw_v = 0.0;
    var rgb = vec3<f32>(0.0);

    if (nparams.family == 0u) {
        // RGB: three full-resolution planes.
        let r0 = normalise_luma(load_native(nparams.in0, i));
        let g0 = normalise_luma(load_native(nparams.in1, i));
        let b0 = normalise_luma(load_native(nparams.in2, i));
        rgb = vec3<f32>(r0, g0, b0);
    } else if (nparams.family == 1u) {
        // Gray: one plane, replicated. `rgb_to_frame` collapses it back with the same weights.
        raw_y = normalise_luma(load_native(nparams.in0, i));
        rgb = vec3<f32>(raw_y);
    } else {
        raw_y = normalise_luma(load_native(nparams.in0, i));
        let cu = normalise_chroma(chroma_at(nparams.in1, x, y));
        let cv = normalise_chroma(chroma_at(nparams.in2, x, y));
        raw_u = cu;
        raw_v = cv;
        // Mirror `yuv_to_rgb`.
        let kg = 1.0 - nparams.kr - nparams.kb;
        rgb = vec3<f32>(
            raw_y + 2.0 * (1.0 - nparams.kr) * cv,
            raw_y - (2.0 * (1.0 - nparams.kr) * nparams.kr / kg) * cv
                  - (2.0 * (1.0 - nparams.kb) * nparams.kb / kg) * cu,
            raw_y + 2.0 * (1.0 - nparams.kb) * cu,
        );
    }

    // --- sanitise, exactly as the CPU does before the chain ---
    let r0 = sanitize(rgb.x, nparams.clamp_lo, nparams.clamp_hi);
    let g0 = sanitize(rgb.y, nparams.clamp_lo, nparams.clamp_hi);
    let b0 = sanitize(rgb.z, nparams.clamp_lo, nparams.clamp_hi);
    rgb = vec3<f32>(r0, g0, b0);

    // --- the slice chain, identical to `main` ---
    for (var k = 0u; k < nparams.k; k = k + 1u) {
        let m = k * 9u;
        let u = vec3<f32>(
            model[m] * rgb.x + model[m + 1u] * rgb.y + model[m + 2u] * rgb.z,
            model[m + 3u] * rgb.x + model[m + 4u] * rgb.y + model[m + 5u] * rgb.z,
            model[m + 6u] * rgb.x + model[m + 7u] * rgb.y + model[m + 8u] * rgb.z,
        );
        let stride = 2u + nparams.nodes;
        let a0 = nparams.axes_base + (k * 3u) * stride;
        let t = vec3<f32>(
            eval_axis(a0, u.x),
            eval_axis(a0 + stride, u.y),
            eval_axis(a0 + 2u * stride, u.z),
        );
        let d = t - u;
        rgb = rgb + vec3<f32>(
            model[m] * d.x + model[m + 3u] * d.y + model[m + 6u] * d.z,
            model[m + 1u] * d.x + model[m + 4u] * d.y + model[m + 7u] * d.z,
            model[m + 2u] * d.x + model[m + 5u] * d.y + model[m + 8u] * d.z,
        );
    }

    if (nparams.luma_lock > 0.0) {
        let y0 = dot(vec3<f32>(r0, g0, b0), LUMA);
        let y1 = dot(rgb, LUMA);
        let locked_y = y0 + (y1 - y0) * (1.0 - nparams.luma_lock);
        rgb = rgb + vec3<f32>(locked_y - y1);
    }

    if (nparams.gamut == 1u) {
        rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (nparams.gamut == 2u) {
        rgb = vec3<f32>(
            soft_knee_lo(soft_knee(rgb.x, 1.0), 0.0),
            soft_knee_lo(soft_knee(rgb.y, 1.0), 0.0),
            soft_knee_lo(soft_knee(rgb.z, 1.0), 0.0),
        );
    }

    // --- store ---
    if (nparams.family == 0u) {
        store_native(nparams.out0, i, denormalise(rgb.x, nparams.d_luma_scale, nparams.d_luma_offset, nparams.d_luma_peak));
        store_native(nparams.out1, i, denormalise(rgb.y, nparams.d_luma_scale, nparams.d_luma_offset, nparams.d_luma_peak));
        store_native(nparams.out2, i, denormalise(rgb.z, nparams.d_luma_scale, nparams.d_luma_offset, nparams.d_luma_peak));
        return;
    }
    if (nparams.family == 1u) {
        // Collapse with the same weights `rgb_to_frame` uses for gray.
        let kg = 1.0 - nparams.kr - nparams.kb;
        let lum = nparams.kr * rgb.x + kg * rgb.y + nparams.kb * rgb.z;
        store_native(nparams.out0, i, denormalise(lum, nparams.d_luma_scale, nparams.d_luma_offset, nparams.d_luma_peak));
        return;
    }

    // YUV: mirror `rgb_to_yuv`, then write luma, and chroma only from the co-sited invocations.
    let kg = 1.0 - nparams.kr - nparams.kb;
    let ny = nparams.kr * rgb.x + kg * rgb.y + nparams.kb * rgb.z;
    let nu = (rgb.z - ny) / (2.0 * (1.0 - nparams.kb));
    let nv = (rgb.x - ny) / (2.0 * (1.0 - nparams.kr));

    store_native(nparams.out0, i, denormalise(ny, nparams.d_luma_scale, nparams.d_luma_offset, nparams.d_luma_peak));

    // Chroma sample `(x / sub_h, y / sub_v)` is co-sited with luma `(x, y)`, so exactly the
    // invocations at multiples of the subsample factors write chroma. That is the nearest
    // downsample, and it is why the round trip is lossless.
    if ((x % nparams.sub_h) == 0u && (y % nparams.sub_v) == 0u) {
        let ci = (y / nparams.sub_v) * nparams.cw + (x / nparams.sub_h);
        store_native(nparams.out1, ci, denormalise(nu, nparams.d_chroma_scale, nparams.d_chroma_offset, nparams.d_chroma_peak));
        store_native(nparams.out2, ci, denormalise(nv, nparams.d_chroma_scale, nparams.d_chroma_offset, nparams.d_chroma_peak));
    }
}
