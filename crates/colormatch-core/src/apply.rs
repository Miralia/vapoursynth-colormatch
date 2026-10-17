//! Applying a model to pixel planes.
//!
//! Two methods are provided:
//!
//! * [`apply_native`] — evaluates the slice chain directly. Exact, but cost scales with `K`.
//! * [`crate::lut::bake_lut`] + [`crate::lut::apply_lut_tetrahedral`] — bakes the chain into a 3D
//!   LUT. Cost is independent of `K`, at the price of interpolation error.
//!
//! # Layout and loop order
//!
//! Planes are processed as separate contiguous `f32` buffers (structure-of-arrays). The slice loop
//! is the *inner* loop over a tile of pixels, so the image is read and written once and the `K`
//! small tables stay resident in L1/L2. The cost is dominated by the table gathers rather than by
//! the 3×3 matrix products — the arithmetic is a few multiply-adds per channel while a gather is a
//! dependent memory access — which is why the tables are kept small and indexed directly.

use crate::model::Model;
use wide::f32x8;

/// How out-of-gamut results are handled.
///
/// The default is [`Gamut::None`], and that is a correctness decision rather than a permissive one.
/// The transform works in an RGB space derived from the input by a matrix, and for limited-range
/// YUV the legal RGB range is *wider* than `[0, 1]`: `Y=235, U=V=128` is RGB `(1, 1, 1)`, but
/// `Y=235` with a non-neutral chroma maps outside the cube, and video explicitly reserves that
/// headroom for undershoot and overshoot. Clamping to `[0, 1]` therefore throws away legal signal —
/// measured on a limited-range 4:2:0 clip, a hard clamp changed samples by 31% of full scale even
/// for an *identity* model. The encoder's own clamp to the sample range is the right place to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Gamut {
    /// Leave out-of-range values alone. The encoder clamps to the sample range.
    #[default]
    None = 0,
    /// Hard clamp to `[0, 1]`. Cheap, but shifts hue in clipped highlights and destroys YUV headroom.
    Clip = 1,
    /// Soft knee: compress values above `1.0` (and below `0.0`) instead of clipping, preserving
    /// hue relationships at the cost of a slight luminance compression.
    Soft = 2,
}

/// Options controlling a native apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApplyOptions {
    /// Out-of-gamut handling.
    pub gamut: Gamut,
    /// How strongly to restore the original luma after the transform, in `[0, 1]`.
    ///
    /// `0.0` disables the correction; `1.0` fully restores the source luma (useful to keep an
    /// exposure shift from the reference look out of the result).
    pub luma_lock: f32,
    /// Upper clamp applied before the transform, matching the builder's clipping convention.
    pub clamp_hi: f32,
    /// Lower clamp applied before the transform.
    pub clamp_lo: f32,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            // `Gamut::None`, matching the enum's own `#[default]`, the plugin's default and the
            // documented one. This said `Clip` while all three said `None`, so a library caller
            // taking the struct default got a hard clamp that the plugin would never apply — and
            // for limited-range YUV that clamp discards legal headroom.
            gamut: Gamut::None,
            luma_lock: 0.0,
            clamp_lo: -1.732_050_8,
            clamp_hi: 1.732_050_8,
        }
    }
}

/// Rec.709 luma weights, used by [`ApplyOptions::luma_lock`].
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Clamps a finite input into range and replaces non-finite values with 0.
#[inline]
fn sanitize(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() { v.clamp(lo, hi) } else { 0.0 }
}

#[inline]
fn soft_knee(v: f32, hi: f32) -> f32 {
    if v <= hi {
        return v;
    }
    // Asymptotically approach `hi + 1` from below, C¹-continuous at `hi`.
    let over = v - hi;
    hi + over / (1.0 + over)
}

#[inline]
fn soft_knee_lo(v: f32, lo: f32) -> f32 {
    if v >= lo {
        return v;
    }
    let over = lo - v;
    lo - over / (1.0 + over)
}

/// Applies `model` to three planar `f32` buffers in place.
///
/// The buffers must have the same length and are interpreted as RGB in the model's working space.
/// Values outside `[options.clamp_lo, options.clamp_hi]` are clamped before evaluation.
///
/// `luma_lock` and the gamut mode are compile-time constants here. Both were previously tested
/// inside the per-pixel loop, which cost roughly 2x: the loop is memory- and branch-bound, and two
/// unpredictable branches per pixel are exactly what stops it from vectorising.
pub fn apply_native(model: &Model, planes: [&mut [f32]; 3], options: &ApplyOptions) {
    let lock = options.luma_lock > 0.0;
    match (lock, options.gamut) {
        (false, Gamut::None) => apply_impl::<false, 0>(model, planes, options),
        (false, Gamut::Clip) => apply_impl::<false, 1>(model, planes, options),
        (false, Gamut::Soft) => apply_impl::<false, 2>(model, planes, options),
        (true, Gamut::None) => apply_impl::<true, 0>(model, planes, options),
        (true, Gamut::Clip) => apply_impl::<true, 1>(model, planes, options),
        (true, Gamut::Soft) => apply_impl::<true, 2>(model, planes, options),
    }
}

fn apply_impl<const LOCK: bool, const GAMUT: u8>(
    model: &Model,
    planes: [&mut [f32]; 3],
    options: &ApplyOptions,
) {
    let n = planes[0].len();
    debug_assert!(planes[1].len() == n && planes[2].len() == n);

    if model.is_empty() {
        return;
    }

    // Tile so the working set stays in L1 while the K tables stay in L2.
    const TILE: usize = 4096;

    let (clamp_lo, clamp_hi) = (options.clamp_lo, options.clamp_hi);
    let luma_lock = options.luma_lock;
    let slices = &model.slices;

    let [pr, pg, pb] = planes;
    for base in (0..n).step_by(TILE) {
        let end = (base + TILE).min(n);
        for i in base..end {
            // Non-finite inputs are mapped to 0 rather than allowed to propagate: a NaN in a
            // float clip must not turn an entire frame into NaN.
            let r0 = sanitize(pr[i], clamp_lo, clamp_hi);
            let g0 = sanitize(pg[i], clamp_lo, clamp_hi);
            let b0 = sanitize(pb[i], clamp_lo, clamp_hi);
            let (mut r, mut g, mut b) = (r0, g0, b0);

            for s in slices {
                let m = &s.rot;
                let u0 = m[0][0] * r + m[0][1] * g + m[0][2] * b;
                let u1 = m[1][0] * r + m[1][1] * g + m[1][2] * b;
                let u2 = m[2][0] * r + m[2][1] * g + m[2][2] * b;

                let d0 = s.axes[0].eval(u0) - u0;
                let d1 = s.axes[1].eval(u1) - u1;
                let d2 = s.axes[2].eval(u2) - u2;

                // Rotate the correction back: Rᵀ·d.
                r += m[0][0] * d0 + m[1][0] * d1 + m[2][0] * d2;
                g += m[0][1] * d0 + m[1][1] * d1 + m[2][1] * d2;
                b += m[0][2] * d0 + m[1][2] * d1 + m[2][2] * d2;
            }

            if LOCK {
                let y0 = LUMA[0] * r0 + LUMA[1] * g0 + LUMA[2] * b0;
                let y1 = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
                let target = y0 + (y1 - y0) * (1.0 - luma_lock);
                let shift = target - y1;
                r += shift;
                g += shift;
                b += shift;
            }

            // A `u8` const generic rather than a `Gamut` one: const generics only accept integral
            // types, `bool` and `char`. The match still folds away at compile time, so the branch
            // costs nothing and the loop stays vectorisable.
            let (r, g, b) = match GAMUT {
                1 => (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)),
                2 => (
                    soft_knee_lo(soft_knee(r, 1.0), 0.0),
                    soft_knee_lo(soft_knee(g, 1.0), 0.0),
                    soft_knee_lo(soft_knee(b, 1.0), 0.0),
                ),
                _ => (r, g, b),
            };

            pr[i] = r;
            pg[i] = g;
            pb[i] = b;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Wide (portable SIMD) path
// ---------------------------------------------------------------------------------------------

/// Width of the vector used by [`apply_native_wide`], in lanes.
pub const WIDE_LANES: usize = 8;

/// Whether the wide path can run here. `wide` resolves to NEON on aarch64 and AVX/SSE elsewhere, so
/// this is a compile-time yes on every target the plugin ships to; the check exists so callers can
/// report the backend honestly.
#[inline]
pub const fn wide_available() -> bool {
    cfg!(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "x86",
    ))
}

/// The wide path's inner work for one slice: rotate, evaluate, rotate back.
///
/// The rotations are vector operations. The three table evaluations are not — there is no gather
/// instruction on aarch64 and none that `wide` exposes portably — so they are done per lane after a
/// store to a small stack array. The loop still comes out well ahead because the rotations, the
/// correction, and the loop's bookkeeping all widen.
#[inline(always)]
fn wide_slice(s: &crate::model::Slice, r: f32x8, g: f32x8, b: f32x8) -> (f32x8, f32x8, f32x8) {
    let m = &s.rot;
    // Left-to-right association, matching the scalar expression exactly. Both are plain multiply-add
    // chains with no FMA contraction, so the two agree bit for bit; the test below is what holds
    // that true.
    let u0 = r * m[0][0] + g * m[0][1] + b * m[0][2];
    let u1 = r * m[1][0] + g * m[1][1] + b * m[1][2];
    let u2 = r * m[2][0] + g * m[2][1] + b * m[2][2];

    let (mut a0, mut a1, mut a2) = (u0.to_array(), u1.to_array(), u2.to_array());
    for lane in 0..WIDE_LANES {
        a0[lane] = s.axes[0].eval(a0[lane]) - a0[lane];
        a1[lane] = s.axes[1].eval(a1[lane]) - a1[lane];
        a2[lane] = s.axes[2].eval(a2[lane]) - a2[lane];
    }
    let (d0, d1, d2) = (f32x8::from(a0), f32x8::from(a1), f32x8::from(a2));

    // The inner sum, then the addition to the accumulator -- matching the scalar path's
    // association exactly. Writing it as `r + d0*m00 + d1*m10 + d2*m20` folds `r` in first, which
    // rounds in a different place and costs a ULP. The scalar path is the reference, so the vector
    // path has to reproduce its rounding order, not merely an equivalent-looking expression.
    (
        r + (d0 * m[0][0] + d1 * m[1][0] + d2 * m[2][0]),
        g + (d0 * m[0][1] + d1 * m[1][1] + d2 * m[2][1]),
        b + (d0 * m[0][2] + d1 * m[1][2] + d2 * m[2][2]),
    )
}

/// Applies `model` using portable SIMD, eight pixels at a time.
///
/// Produces results bit-identical to [`apply_native`]'s scalar loop for every input: the vector
/// width changes how many pixels are in flight, not the order of any single pixel's arithmetic, and
/// each lane evaluates its own table lookup with the same scalar [`crate::model::Axis::eval`]. That
/// is what lets the backend be swapped by preference with no effect on output.
pub fn apply_native_wide(model: &Model, planes: [&mut [f32]; 3], options: &ApplyOptions) {
    let lock = options.luma_lock > 0.0;
    match (lock, options.gamut) {
        (false, Gamut::None) => apply_wide_impl::<false, 0>(model, planes, options),
        (false, Gamut::Clip) => apply_wide_impl::<false, 1>(model, planes, options),
        (false, Gamut::Soft) => apply_wide_impl::<false, 2>(model, planes, options),
        (true, Gamut::None) => apply_wide_impl::<true, 0>(model, planes, options),
        (true, Gamut::Clip) => apply_wide_impl::<true, 1>(model, planes, options),
        (true, Gamut::Soft) => apply_wide_impl::<true, 2>(model, planes, options),
    }
}

fn apply_wide_impl<const LOCK: bool, const GAMUT: u8>(
    model: &Model,
    planes: [&mut [f32]; 3],
    options: &ApplyOptions,
) {
    let n = planes[0].len();
    debug_assert!(planes[1].len() == n && planes[2].len() == n);
    if model.is_empty() {
        return;
    }

    let (clamp_lo, clamp_hi) = (options.clamp_lo, options.clamp_hi);
    let luma_lock = options.luma_lock;
    let slices = &model.slices;
    let [pr, pg, pb] = planes;

    let vector_end = n - (n % WIDE_LANES);

    let mut i = 0usize;
    while i < vector_end {
        let (mut rv, mut gv, mut bv) = (
            f32x8::from(&pr[i..i + WIDE_LANES]),
            f32x8::from(&pg[i..i + WIDE_LANES]),
            f32x8::from(&pb[i..i + WIDE_LANES]),
        );

        // Sanitising is per-lane: the non-finite mapping is a comparison, not arithmetic.
        let (mut r0, mut g0, mut b0) = (rv.to_array(), gv.to_array(), bv.to_array());
        for lane in 0..WIDE_LANES {
            r0[lane] = sanitize(r0[lane], clamp_lo, clamp_hi);
            g0[lane] = sanitize(g0[lane], clamp_lo, clamp_hi);
            b0[lane] = sanitize(b0[lane], clamp_lo, clamp_hi);
        }
        let (or_, og_, ob_) = (r0, g0, b0);
        rv = f32x8::from(r0);
        gv = f32x8::from(g0);
        bv = f32x8::from(b0);

        for s in slices {
            let (nr, ng, nb) = wide_slice(s, rv, gv, bv);
            rv = nr;
            gv = ng;
            bv = nb;
        }

        if LOCK {
            let orv = f32x8::from(or_);
            let ogv = f32x8::from(og_);
            let obv = f32x8::from(ob_);
            let y0 = orv * LUMA[0] + ogv * LUMA[1] + obv * LUMA[2];
            let y1 = rv * LUMA[0] + gv * LUMA[1] + bv * LUMA[2];
            let target = y0 + (y1 - y0) * (1.0 - luma_lock);
            let shift = target - y1;
            rv += shift;
            gv += shift;
            bv += shift;
        }

        let (rr, gg, bb) = match GAMUT {
            1 => (
                rv.fast_max(f32x8::ZERO).fast_min(f32x8::ONE),
                gv.fast_max(f32x8::ZERO).fast_min(f32x8::ONE),
                bv.fast_max(f32x8::ZERO).fast_min(f32x8::ONE),
            ),
            2 => {
                let mut ra = rv.to_array();
                let mut ga = gv.to_array();
                let mut ba = bv.to_array();
                for lane in 0..WIDE_LANES {
                    ra[lane] = soft_knee_lo(soft_knee(ra[lane], 1.0), 0.0);
                    ga[lane] = soft_knee_lo(soft_knee(ga[lane], 1.0), 0.0);
                    ba[lane] = soft_knee_lo(soft_knee(ba[lane], 1.0), 0.0);
                }
                (f32x8::from(ra), f32x8::from(ga), f32x8::from(ba))
            }
            _ => (rv, gv, bv),
        };

        pr[i..i + WIDE_LANES].copy_from_slice(&rr.to_array());
        pg[i..i + WIDE_LANES].copy_from_slice(&gg.to_array());
        pb[i..i + WIDE_LANES].copy_from_slice(&bb.to_array());
        i += WIDE_LANES;
    }

    // Tail: fewer than WIDE_LANES pixels left.
    while i < n {
        let r0 = sanitize(pr[i], clamp_lo, clamp_hi);
        let g0 = sanitize(pg[i], clamp_lo, clamp_hi);
        let b0 = sanitize(pb[i], clamp_lo, clamp_hi);
        let (mut r, mut g, mut b) = (r0, g0, b0);
        for s in slices {
            let m = &s.rot;
            let u0 = m[0][0] * r + m[0][1] * g + m[0][2] * b;
            let u1 = m[1][0] * r + m[1][1] * g + m[1][2] * b;
            let u2 = m[2][0] * r + m[2][1] * g + m[2][2] * b;
            let d0 = s.axes[0].eval(u0) - u0;
            let d1 = s.axes[1].eval(u1) - u1;
            let d2 = s.axes[2].eval(u2) - u2;
            r += m[0][0] * d0 + m[1][0] * d1 + m[2][0] * d2;
            g += m[0][1] * d0 + m[1][1] * d1 + m[2][1] * d2;
            b += m[0][2] * d0 + m[1][2] * d1 + m[2][2] * d2;
        }
        if LOCK {
            let y0 = LUMA[0] * r0 + LUMA[1] * g0 + LUMA[2] * b0;
            let y1 = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
            let target = y0 + (y1 - y0) * (1.0 - luma_lock);
            let shift = target - y1;
            r += shift;
            g += shift;
            b += shift;
        }
        let (r, g, b) = match GAMUT {
            1 => (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)),
            2 => (
                soft_knee_lo(soft_knee(r, 1.0), 0.0),
                soft_knee_lo(soft_knee(g, 1.0), 0.0),
                soft_knee_lo(soft_knee(b, 1.0), 0.0),
            ),
            _ => (r, g, b),
        };
        pr[i] = r;
        pg[i] = g;
        pb[i] = b;
        i += 1;
    }
}

/// Applies the slice chain to a single RGB triple, without clamping or gamut handling.
///
/// This is the primitive that the slice evaluation is defined in terms of; it is used by the
/// gain estimator and by tests, and mirrors the inner loop of [`apply_native`].
#[inline]
pub fn apply_point(model: &Model, px: [f32; 3]) -> [f32; 3] {
    let (mut r, mut g, mut b) = (px[0], px[1], px[2]);
    for s in &model.slices {
        let m = &s.rot;
        let u0 = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let u1 = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let u2 = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        let d0 = s.axes[0].eval(u0) - u0;
        let d1 = s.axes[1].eval(u1) - u1;
        let d2 = s.axes[2].eval(u2) - u2;
        r += m[0][0] * d0 + m[1][0] * d1 + m[2][0] * d2;
        g += m[0][1] * d0 + m[1][1] * d1 + m[2][1] * d2;
        b += m[0][2] * d0 + m[1][2] * d1 + m[2][2] * d2;
    }
    [r, g, b]
}

/// Convenience wrapper that applies the model to interleaved RGB triples.
///
/// Intended for tests, the CLI and the NumPy reference comparison; the plugin uses the planar
/// form to avoid a deinterleave pass.
pub fn apply_native_interleaved(model: &Model, px: &mut [[f32; 3]], options: &ApplyOptions) {
    let n = px.len();
    let mut r: Vec<f32> = px.iter().map(|p| p[0]).collect();
    let mut g: Vec<f32> = px.iter().map(|p| p[1]).collect();
    let mut b: Vec<f32> = px.iter().map(|p| p[2]).collect();
    apply_native(model, [&mut r, &mut g, &mut b], options);
    for i in 0..n {
        px[i] = [r[i], g[i], b[i]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Axis, Slice, TABLE_NODES};

    fn identity_axis() -> Axis {
        Axis::identity(-1.732_050_8, 1.732_050_8)
    }

    #[test]
    fn empty_model_is_a_noop() {
        let model = Model::identity(TABLE_NODES);
        let mut px = vec![[0.2f32, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let before = px.clone();
        apply_native_interleaved(&model, &mut px, &ApplyOptions::default());
        assert_eq!(px, before);
    }

    #[test]
    fn identity_slice_is_a_noop() {
        let mut model = Model::identity(TABLE_NODES);
        model.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [identity_axis(), identity_axis(), identity_axis()],
        });
        let mut px = vec![[0.2f32, 0.5, 0.9], [0.31, 0.42, 0.53]];
        let before = px.clone();
        apply_native_interleaved(&model, &mut px, &ApplyOptions::default());
        for (a, b) in px.iter().zip(before.iter()) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-6, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn identity_rotation_with_gain_only_affects_one_channel() {
        // A single slice whose first axis multiplies by 0.5 while the others are identity.
        let mut model = Model::identity(TABLE_NODES);
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        let axes = [
            Axis::from_fn(lo, hi, |x| x * 0.5),
            Axis::identity(lo, hi),
            Axis::identity(lo, hi),
        ];
        model.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes,
        });

        let mut px = vec![[0.4f32, 0.4, 0.4]];
        apply_native_interleaved(&model, &mut px, &ApplyOptions::default());
        assert!((px[0][0] - 0.2).abs() < 1e-5, "got {:?}", px[0]);
        assert!((px[0][1] - 0.4).abs() < 1e-6);
        assert!((px[0][2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn rotation_moves_energy_between_channels() {
        // 90° rotation about Z maps R -> G. With identity axes the slice is still a no-op,
        // so use a gain on the first axis to make the rotation observable.
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        let mut model = Model::identity(TABLE_NODES);
        model.slices.push(Slice {
            rot: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::from_fn(lo, hi, |x| x * 0.5),
                Axis::identity(lo, hi),
                Axis::identity(lo, hi),
            ],
        });
        // u0 = -g, so halving u0 means g' = g - 0.5*g = 0.5*g; R and B unchanged.
        let mut px = vec![[0.1f32, 0.8, 0.3]];
        apply_native_interleaved(&model, &mut px, &ApplyOptions::default());
        assert!((px[0][1] - 0.4).abs() < 1e-5, "got {:?}", px[0]);
        assert!((px[0][0] - 0.1).abs() < 1e-6, "got {:?}", px[0]);
        assert!((px[0][2] - 0.3).abs() < 1e-6, "got {:?}", px[0]);
    }

    #[test]
    fn clip_and_soft_gamut_differ_out_of_range() {
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        let mut model = Model::identity(TABLE_NODES);
        model.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::from_fn(lo, hi, |x| x * 3.0),
                Axis::identity(lo, hi),
                Axis::identity(lo, hi),
            ],
        });
        let mk = || vec![[0.8f32, 0.5, 0.5]];

        let mut a = mk();
        apply_native_interleaved(
            &model,
            &mut a,
            &ApplyOptions {
                gamut: Gamut::Clip,
                ..Default::default()
            },
        );
        assert!((a[0][0] - 1.0).abs() < 1e-6);

        let mut b = mk();
        apply_native_interleaved(
            &model,
            &mut b,
            &ApplyOptions {
                gamut: Gamut::Soft,
                ..Default::default()
            },
        );
        assert!(
            b[0][0] > 1.0,
            "soft knee should overshoot past 1.0: {:?}",
            b[0]
        );
        assert!(b[0][0] < 2.0, "soft knee should stay bounded: {:?}", b[0]);
    }

    #[test]
    fn luma_lock_restores_source_luma() {
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        let mut model = Model::identity(TABLE_NODES);
        model.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::from_fn(lo, hi, |x| x * 0.6),
                Axis::from_fn(lo, hi, |x| x * 0.6),
                Axis::from_fn(lo, hi, |x| x * 0.6),
            ],
        });
        let src = [0.7f32, 0.7, 0.7];
        let luma = |p: [f32; 3]| LUMA[0] * p[0] + LUMA[1] * p[1] + LUMA[2] * p[2];

        let mut free = vec![src];
        apply_native_interleaved(&model, &mut free, &ApplyOptions::default());
        assert!(luma(free[0]) < 0.5, "expected a darkening without lock");

        let mut locked = vec![src];
        apply_native_interleaved(
            &model,
            &mut locked,
            &ApplyOptions {
                luma_lock: 1.0,
                ..Default::default()
            },
        );
        assert!(
            (luma(locked[0]) - luma(src)).abs() < 1e-4,
            "luma not restored: {:?}",
            locked[0]
        );
    }

    #[test]
    fn nan_input_does_not_propagate_nan_output() {
        let mut model = Model::identity(TABLE_NODES);
        model.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [identity_axis(), identity_axis(), identity_axis()],
        });
        let mut px = vec![[f32::NAN, 0.5, 0.5], [f32::INFINITY, 0.5, 0.5]];
        apply_native_interleaved(&model, &mut px, &ApplyOptions::default());
        for p in &px {
            for c in p {
                assert!(!c.is_nan(), "NaN leaked: {p:?}");
            }
        }
    }
}

#[cfg(test)]
mod wide_tests {
    use super::*;
    use crate::build::{BuildOptions, Samples, build_model};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
        }
    }

    fn model(k: usize) -> Model {
        let mut r = Rng(0x51D);
        let a: Vec<[f32; 3]> = (0..30_000)
            .map(|_| [r.next(), r.next(), r.next()])
            .collect();
        let b: Vec<[f32; 3]> = a
            .iter()
            .map(|p| {
                [
                    (p[0] * 0.8 + 0.1).clamp(0.0, 1.0),
                    (p[1] * 0.9 + 0.05).clamp(0.0, 1.0),
                    (p[2] * 0.7 + 0.2).clamp(0.0, 1.0),
                ]
            })
            .collect();
        build_model(
            &Samples::new(a, 1),
            &Samples::new(b, 1),
            &BuildOptions {
                directions: k,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// A triple of planes holding values inside the grid, outside it, and non-finite.
    fn planes(n: usize, seed: u64) -> [Vec<f32>; 3] {
        let mut r = Rng(seed);
        let mut mk = move |i: usize| -> f32 {
            match i % 7 {
                0 => r.next(),
                1 => r.next() * 4.0 - 2.0,
                2 => f32::NEG_INFINITY,
                3 => f32::INFINITY,
                4 => f32::NAN,
                5 => 0.0,
                _ => 1.0,
            }
        };
        [
            (0..n).map(&mut mk).collect(),
            (0..n).map(&mut mk).collect(),
            (0..n).map(&mut mk).collect(),
        ]
    }

    /// The whole point of the wide path: it must not change a single output bit.
    ///
    /// Covers every combination of the compile-time switches, four slice counts, and lengths that
    /// are and are not multiples of the vector width, so the scalar tail is exercised too. Bit
    /// equality rather than a tolerance, because the two paths are required to be interchangeable —
    /// a tolerance here would hide exactly the kind of divergence this backend exists to avoid.
    #[test]
    fn wide_matches_scalar_bit_for_bit() {
        for k in [1usize, 3, 8, 32] {
            let m = model(k);
            for luma_lock in [0.0f32, 0.5, 1.0] {
                for gamut in [Gamut::None, Gamut::Clip, Gamut::Soft] {
                    let opts = ApplyOptions {
                        luma_lock,
                        gamut,
                        ..Default::default()
                    };
                    for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 1000] {
                        // Three distinct bindings, not array indexing: `&mut a[0], &mut a[1]` does
                        // not borrow-disjoint through an index expression.
                        let [ar, ag, ab] = planes(n, 0xA11CE);
                        let (mut wr, mut wg, mut wb) = (ar.clone(), ag.clone(), ab.clone());
                        let (mut ar, mut ag, mut ab) = (ar, ag, ab);
                        apply_native(&m, [&mut ar, &mut ag, &mut ab], &opts);
                        apply_native_wide(&m, [&mut wr, &mut wg, &mut wb], &opts);
                        let pairs = [(&ar, &wr), (&ag, &wg), (&ab, &wb)];
                        for (plane, (scalar, wide)) in pairs.into_iter().enumerate() {
                            for (i, (x, y)) in scalar.iter().zip(wide).enumerate() {
                                assert_eq!(
                                    x.to_bits(),
                                    y.to_bits(),
                                    "K={k} lock={luma_lock} gamut={gamut:?} n={n}                                      plane={plane} idx={i}: scalar {x} vs wide {y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wide_handles_an_empty_model() {
        let m = model(4);
        let mut m = m.clone();
        m.slices.clear();
        assert!(m.is_empty());
        let (mut a, mut b, mut c) = (vec![0.5f32; 16], vec![0.5f32; 16], vec![0.5f32; 16]);
        apply_native_wide(&m, [&mut a, &mut b, &mut c], &ApplyOptions::default());
        assert_eq!(a, vec![0.5f32; 16], "an empty model must be an exact no-op");
    }

    #[test]
    fn wide_is_available_on_the_targets_we_ship() {
        // `wide` resolves to NEON on aarch64 and AVX/SSE on x86_64, so the answer is yes on every
        // platform the plugin builds for.
        assert!(wide_available());
        assert!(WIDE_LANES.is_power_of_two());
    }
}

#[cfg(test)]
mod default_tests {
    use super::*;

    /// The struct default has to agree with the enum default, the plugin's default and the
    /// documented default. It did not: this said `Clip` while all three said `None`.
    #[test]
    fn apply_options_default_gamut_is_none() {
        assert_eq!(ApplyOptions::default().gamut, Gamut::None);
        assert_eq!(Gamut::default(), Gamut::None);
    }

    /// The clamp bounds are the cube's diagonal, which is the widest value a projection of a unit
    /// cube pixel can reach. Narrowing them would silently clamp legal limited-range headroom.
    #[test]
    fn clamp_bounds_span_the_unit_cube_projection() {
        let d = ApplyOptions::default();
        assert!((d.clamp_hi - 1.732_050_8).abs() < 1e-6, "{}", d.clamp_hi);
        assert!((d.clamp_lo + 1.732_050_8).abs() < 1e-6, "{}", d.clamp_lo);
    }
}
