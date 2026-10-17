//! Model building: pooled sampling, sliced-Wasserstein sweep, transfer-table fitting.
//!
//! The transform is the fixed point of a sequential sliced-Wasserstein gradient flow. At each step
//! `k` we project both clouds onto the axes of a rotation `R_k`, replace the projected marginals of
//! the running accumulator with the reference's, and rotate the correction back:
//!
//! ```text
//! X ← X + (T_k(X·R_k) − X·R_k)·R_kᵀ
//! ```
//!
//! The reference cloud `Y` is never modified, which is what makes this a fixed-point iteration rather
//! than a single affine fit.
//!
//! `T_k` approximates `F_ref⁻¹ ∘ F_test` on a uniform grid, from linearly splatted histograms. The
//! splat makes the map continuous and monotone by construction, so it needs no repair pass; it is a
//! structural choice rather than an accuracy win, since the CDF smoothing applied anyway is what
//! removes the staircase. `Builder::DrDre` reproduces the original's percentile-knot fit instead.
//!
//! The full derivation, the measurements behind those statements and the reasoning for each of the
//! five fixups (splat, smoothing, exact inversion, band trimming, unit-slope extrapolation) are in
//! `docs/algorithm.md`. `examples/measure.rs` reproduces every figure quoted there.

use crate::model::{Axis, FORMAT_VERSION, Model, ModelStats, Slice, TABLE_NODES};

/// `sqrt(3)`: the unit RGB cube's diagonal length, and therefore the bound on `|a·p|` for a unit
/// vector `a` and `p` in `[0, 1]³`. Used as the rotated axes' grid range, matching `ColorMatch.m`.
///
/// Any coordinate of a point in `[0,1]³` after an orthonormal rotation lies within this bound,
/// which is why rotated axes use `[−√3, +√3]` as their grid range.
pub const CUBE_DIAG: f32 = 1.732_050_8;

/// Errors from model building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// No samples were provided on one side.
    NoSamples,
    /// `directions` was zero.
    NoDirections,
    /// `bins` was too small to resolve the distributions.
    BinsTooSmall(usize),
    /// `bins` exceeded the table's fixed node count.
    ///
    /// The model format stores a fixed [`TABLE_NODES`]-entry table per axis, so a finer histogram
    /// than that has nowhere to go. This is rejected rather than truncated, and the check lives here
    /// because the table's size is a property of the format, not of any one caller.
    BinsTooLarge {
        /// The requested bin count.
        bins: usize,
        /// The largest bin count the format can hold, `TABLE_NODES - 1`.
        max: usize,
    },
    /// The two sides have incompatible sample counts for the requested operation.
    SampleCountMismatch {
        /// Samples on the test side.
        test: usize,
        /// Samples on the reference side.
        reference: usize,
    },
    /// An explicit axes list had the wrong length or a zero-length direction.
    BadAxes(String),
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoSamples => write!(f, "no samples provided"),
            Self::NoDirections => write!(f, "`directions` must be at least 1"),
            Self::BinsTooSmall(b) => write!(f, "`bins` must be at least 16, got {b}"),
            Self::BinsTooLarge { bins, max } => write!(
                f,
                "`bins` must be at most {max} (the model format stores a fixed {}-node table), \
                 got {bins}",
                TABLE_NODES
            ),
            Self::SampleCountMismatch { test, reference } => {
                write!(
                    f,
                    "sample count mismatch: test={test} reference={reference}"
                )
            }
            Self::BadAxes(m) => write!(f, "bad axes: {m}"),
        }
    }
}

impl std::error::Error for BuildError {}

/// How the projection directions are chosen.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AxesScheme {
    /// Deterministic, well-spread directions from a Fibonacci sphere. No optimiser, no RNG.
    #[default]
    Fibonacci,
    /// DrDre's `generateNonParallelAxes`: a repulsion objective minimised from a random start.
    ///
    /// Because the original seeds `fminunc` with `rand`, its output is not reproducible. This
    /// variant is deterministic given `seed`, but is *not* bit-identical to MATLAB's.
    DrDre,
    /// Explicit direction vectors, one per slice. Required for reproducible tests.
    Explicit(Vec<[f32; 3]>),
}

/// Options for [`build_model`].
#[derive(Debug, Clone, PartialEq)]
pub struct BuildOptions {
    /// Number of directions (`K`). 32 is the measured sweet spot.
    pub directions: usize,
    /// Histogram bins per axis (`B`), at most [`TABLE_NODES`] - 1.
    ///
    /// The node count is fixed by the model format, so a larger `B` has nowhere to go and is
    /// rejected. See the measurement in `examples/measure.rs` for where it stops mattering.
    pub bins: usize,
    /// Direction selection.
    pub axes: AxesScheme,
    /// Reserved: extra passes over the direction list. 1 reproduces the original.
    pub passes: usize,
    /// Step size for the gradient flow. The original uses a full step, `1.0`.
    pub step: f32,
    /// Seed for the `DrDre` axes scheme.
    pub seed: u64,
    /// How samples are accumulated into histograms. The default is what the algorithm is designed
    /// around; `Hard` exists for the comparison test.
    pub binning: Binning,
    /// How the transfer table is fitted. `DrDre` reproduces the original's structure so the two can
    /// be compared; the default is the one this project is built around.
    pub builder: Builder,
    /// Passes of the `[1, 2, 1]/4` CDF smoother. [`CDF_SMOOTH_PASSES`] is the default and what the
    /// pipeline is designed around; `0` exists so the effect can be measured rather than asserted.
    pub cdf_smoothing: usize,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            directions: 32,
            bins: 256,
            axes: AxesScheme::Fibonacci,
            passes: 1,
            step: 1.0,
            seed: 0x5EED,
            binning: Binning::default(),
            builder: Builder::default(),
            cdf_smoothing: CDF_SMOOTH_PASSES,
        }
    }
}

/// A pooled RGB sample cloud.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Samples {
    /// Sample pixels in the model's working RGB space, nominally in `[0, 1]`.
    pub rgb: Vec<[f32; 3]>,
    /// How many source frames contributed.
    pub frames: u32,
}

impl Samples {
    /// Creates a sample cloud from interleaved RGB.
    pub fn new(rgb: Vec<[f32; 3]>, frames: u32) -> Self {
        Self { rgb, frames }
    }

    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.rgb.len()
    }

    /// Returns `true` when no pixels were collected.
    pub fn is_empty(&self) -> bool {
        self.rgb.is_empty()
    }

    /// Merges another cloud into this one (used to pool multiple input clips).
    pub fn extend(&mut self, other: &Samples) {
        self.rgb.extend_from_slice(&other.rgb);
        self.frames = self.frames.saturating_add(other.frames);
    }
}

/// Generates `n` well-spread unit directions using a Fibonacci sphere.
pub fn fibonacci_directions(n: usize) -> Vec<[f32; 3]> {
    let golden = std::f64::consts::PI * (1.0 + 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let i = i as f64 + 0.5;
            let phi = (1.0 - 2.0 * i / n as f64).acos();
            let theta = golden * i;
            [
                (theta.cos() * phi.sin()) as f32,
                (theta.sin() * phi.sin()) as f32,
                phi.cos() as f32,
            ]
        })
        .collect()
}

/// Builds an orthonormal frame whose first axis is `d`, with the remaining two axes completing a
/// right-handed basis.
///
/// Unlike DrDre's code, which feeds a unit vector into `rot()` as if it were Euler angles, this
/// produces a proper rotation matrix. The two behave equivalently in practice, but this one is
/// guaranteed orthonormal by construction.
pub fn frame_from_direction(d: [f32; 3]) -> [[f32; 3]; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let e1 = if len > 1e-12 {
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    };
    let helper = if e1[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    // e2 = normalize(helper × e1)
    let mut e2 = [
        helper[1] * e1[2] - helper[2] * e1[1],
        helper[2] * e1[0] - helper[0] * e1[2],
        helper[0] * e1[1] - helper[1] * e1[0],
    ];
    let n2 = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2])
        .sqrt()
        .max(1e-12);
    e2 = [e2[0] / n2, e2[1] / n2, e2[2] / n2];
    // e3 = e1 × e2
    let e3 = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    [e1, e2, e3]
}

/// Builds the rotation list for a build.
pub fn build_rotations(options: &BuildOptions) -> Result<Vec<[[f32; 3]; 3]>, BuildError> {
    if options.directions == 0 {
        return Err(BuildError::NoDirections);
    }
    match &options.axes {
        AxesScheme::Explicit(dirs) => {
            if dirs.len() != options.directions {
                return Err(BuildError::BadAxes(format!(
                    "expected {} directions, got {}",
                    options.directions,
                    dirs.len()
                )));
            }
            Ok(dirs.iter().copied().map(frame_from_direction).collect())
        }
        AxesScheme::Fibonacci | AxesScheme::DrDre => {
            // Both schemes start with the identity frame so that the first slice matches the
            // original's `[0,1]`-ranged, per-channel histogram equalisation.
            let dirs = fibonacci_directions(options.directions);
            let mut out = Vec::with_capacity(options.directions);
            out.push([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            for d in dirs.iter().skip(1) {
                out.push(frame_from_direction(*d));
            }
            Ok(out)
        }
    }
}

/// Linear-splat histogram over `[lo, hi]` with `bins` intervals (`bins + 1` nodes).
///
/// Each sample adds `1−f` and `f` to adjacent bins, which is equivalent to linearly interpolating
/// the empirical CDF.
pub fn splat_histogram(values: &[f32], lo: f32, hi: f32, bins: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; bins + 1];
    let scale = bins as f32 / (hi - lo);
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        // Clamp in *index* space rather than by subtracting an epsilon from the upper bound:
        // `256.0f32 - 1e-6` rounds back to exactly `256.0`, which would index one past the end.
        let x = ((v - lo) * scale).clamp(0.0, bins as f32);
        let i = (x as usize).min(bins - 1);
        let f = x - i as f32;
        h[i] += 1.0 - f;
        h[i + 1] += f;
    }
    h
}

/// Passes of a `[1, 2, 1] / 4` kernel applied to a splatted histogram before inversion.
///
/// # Why this is needed
///
/// The linear splat already makes the CDF piecewise linear, but it does not remove *atoms*: when
/// many samples share a value, the CDF still steps between grid nodes. Inverting a stepped CDF is
/// ill-conditioned — the map is unconstrained across the step — and the resulting table can be
/// locally several times steeper than the identity. Because the sweep composes slices, that excess
/// slope multiplies: measured on a synthetic cloud with 4x-duplicated values (which is exactly what
/// chroma upsampling produces), a self-match showed a worst-case gain of 3.0 at `K=1` rising to 4.6
/// at `K=4`, against 1.0 for the same cloud without duplicates.
///
/// Two passes of the kernel turn those steps into short ramps and bound the inversion's slope. The
/// cost in fidelity is negligible for smooth data — it is a low-pass on the histogram's *derivative*,
/// not on the distribution itself — and it is the standard regularisation for a finite-sample
/// quantile estimate.
/// Passes of the `[1, 2, 1]/4` CDF smoother applied by default.
///
/// The splat removes the staircase from the CDF's interior but not its atoms, and inverting a stepped
/// CDF is unconstrained. Smoothing turns steps into ramps and bounds the slope; the effect is largest
/// on coarsely quantised input, where it roughly halves the worst-case gain. `0` exists so that can be
/// measured rather than asserted — see section 7 of `examples/measure.rs`.
pub const CDF_SMOOTH_PASSES: usize = 2;

/// Applies `passes` of a `[1, 2, 1] / 4` kernel in place, preserving the total.
fn smooth_histogram(h: &mut [f32], passes: usize) {
    if h.len() < 3 {
        return;
    }
    let mut scratch = vec![0.0f32; h.len()];
    for _ in 0..passes {
        scratch.copy_from_slice(h);
        for i in 1..h.len() - 1 {
            h[i] = 0.25 * scratch[i - 1] + 0.5 * scratch[i] + 0.25 * scratch[i + 1];
        }
    }
}

/// Normalised CDF at the `bins + 1` uniform grid nodes.
pub fn cdf_from_histogram(h: &[f32]) -> Vec<f32> {
    let mut c = Vec::with_capacity(h.len());
    let mut acc = 0.0f32;
    for &v in h {
        acc += v;
        c.push(acc);
    }
    let total = acc.max(1e-12);
    for v in c.iter_mut() {
        *v /= total;
    }
    c
}

/// How the per-axis transfer table is fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Builder {
    /// Uniform grid, linear-splatted CDFs, probability-band trimming, identity tail ramps.
    ///
    /// The default. See the module documentation for why each of those is there.
    #[default]
    Splat,
    /// DrDre's ColorMatch, reproduced structurally: percentile-paired knots at 0.1% steps, a
    /// monotonicity repair pass, and `±sqrt(3)` sentinel knots.
    ///
    /// This exists to make the comparison measurable rather than asserted. The two builders are
    /// compared in `examples/measure.rs`; the reason to prefer the default is in the numbers, not in
    /// an opinion about which looks more modern.
    ///
    /// Two details are approximated rather than reproduced, and both are disclosed here because the
    /// name would otherwise overclaim:
    ///
    /// * The original runs a MATLAB `smoothingspline` over the knots with `SmoothingParam` 0.99 —
    ///   near-interpolating, but not exactly. A monotone cubic Hermite (PCHIP) stands in for it. It
    ///   is monotone by construction, which the original's spline is not, and that is the one
    ///   behavioural difference in the interior.
    /// * `interp1(..., 'cubic')` differs from PCHIP in its end conditions and slope estimate, so the
    ///   curves are not bit-identical even before the spline.
    ///
    /// What *is* faithful is the structure that the design argument turns on: non-uniform
    /// percentile knots, no probability trimming, the `±sqrt(3)` sentinels, and a repair pass that
    /// deletes decreasing pairs instead of preventing them.
    DrDre,
}

/// How samples are accumulated into a histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Binning {
    /// Linear splat: each sample contributes `1 - f` and `f` to adjacent bins, which is equivalent
    /// to interpolating the empirical CDF. Continuous, so the transfer map is monotone by
    /// construction.
    #[default]
    Splat,
    /// Hard binning: each sample contributes wholly to one bin. Kept so that the comparison in
    /// `splat_beats_hard_binning` is reproducible from the shipped code rather than being a number
    /// frozen in a comment.
    Hard,
}

/// Accumulates a histogram by hard binning: every sample lands wholly in one bin.
///
/// This is the obvious implementation, and it is the one the comparison test measures against. It
/// produces a staircase CDF whose inverse is discontinuous, so the transfer map is not monotone by
/// construction and needs a repair pass — which is exactly the `interp1` fixup `ColorMatch.m` has to
/// run on every slice.
pub fn hard_histogram(values: &[f32], lo: f32, hi: f32, bins: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; bins + 1];
    let scale = bins as f32 / (hi - lo);
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        let x = ((v - lo) * scale).clamp(0.0, bins as f32);
        let i = (x as usize).min(bins - 1);
        h[i] += 1.0;
    }
    h
}

/// Accumulates a histogram with the given scheme.
fn histogram(values: &[f32], lo: f32, hi: f32, bins: usize, binning: Binning) -> Vec<f32> {
    match binning {
        Binning::Splat => splat_histogram(values, lo, hi, bins),
        Binning::Hard => hard_histogram(values, lo, hi, bins),
    }
}

/// Fraction of probability trimmed from each end of both distributions before matching.
///
/// # Why trimming is not optional
///
/// A quantile map is only well conditioned where both distributions actually have mass. In the
/// extreme tails a single sample decides the answer, so a difference of one part in 10^7 between the
/// two CDFs translates into a transfer table that jumps by whole units. That is not hypothetical:
/// with the tails defined as "first non-zero bin", the sequential sweep accumulates a drift of
/// ~1e-5 per slice, and by the tenth slice the accumulator's tail no longer lines up with the
/// reference's, producing a table whose slope is thousands of times the identity.
///
/// Trimming to the central `1 - 2e-4` of both distributions bounds the interior slope by roughly the
/// ratio of the two band widths, and the tails become identity ramps -- so out-of-distribution
/// content passes through unchanged instead of being crushed.
const PROB_TRIM: f32 = 1e-4;

/// Index range covering the central `1 - 2 * PROB_TRIM` of a CDF.
fn cdf_band(c: &[f32], nodes: usize) -> (usize, usize) {
    debug_assert!(
        nodes >= 2,
        "a CDF needs at least two nodes to be invertible"
    );
    // The upper bound is `nodes - 2`, not `nodes - 1`: the inversion reads `c[k + 1]` and
    // `grid[k + 1]`, so a band ending on the last node would index one past the end.
    let last_usable = nodes - 2;
    let lo_i = c
        .iter()
        .position(|&v| v >= PROB_TRIM)
        .unwrap_or(0)
        .min(last_usable);
    let hi_i = c
        .iter()
        .rposition(|&v| v <= 1.0 - PROB_TRIM)
        .unwrap_or(last_usable)
        .min(last_usable);
    (lo_i, hi_i.max(lo_i))
}

/// Which end of the table a tail fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// Nodes below the band: the identity at `grid[0]`, meeting the interior at the band edge.
    Left,
    /// Nodes above the band: the interior at the band edge, meeting the identity at the last node.
    Right,
}

/// Fills `tab[from..to]` with a straight line joining the identity to the fitted interior.
///
/// The two ends are pinned: the outer end takes the identity's value, the inner end takes the
/// interior's. Nothing else is constrained, deliberately -- see the note at the call site for why
/// matching the interior's slope here was measured and rejected.
fn fill_tail(tab: &mut [f32], grid: &[f32], from: usize, to: usize, tail: Tail) {
    if from >= to {
        return;
    }
    let nodes = tab.len().min(grid.len());
    let (x0, y0, x1, y1) = match tail {
        Tail::Left => (grid[0], grid[0], grid[to], tab[to]),
        Tail::Right => {
            let edge = from - 1;
            let last = nodes - 1;
            (grid[edge], tab[edge], grid[last], grid[last])
        }
    };

    let span = x1 - x0;
    for (j, slot) in tab.iter_mut().enumerate().take(to).skip(from) {
        *slot = if span > 0.0 {
            y0 + (grid[j] - x0) / span * (y1 - y0)
        } else {
            y1
        };
    }
}

/// Percentiles of `values` at `n` equally spaced probability levels from 0 to 100, inclusive.
///
/// Matches MATLAB's `prctile` default: linear interpolation between order statistics. The original
/// uses `0:0.1:100`, i.e. 1001 knots.
fn percentiles(values: &[f32], n: usize) -> Vec<f32> {
    if values.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let last = sorted.len() - 1;
    (0..n)
        .map(|i| {
            let p = if n == 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            };
            let x = p * last as f32;
            let i0 = (x.floor() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            let f = x - i0 as f32;
            sorted[i0] + f * (sorted[i1] - sorted[i0])
        })
        .collect()
}

/// Monotone cubic Hermite (PCHIP) evaluation of `(xs, ys)` at `x`, with linear extrapolation.
///
/// `xs` must be strictly increasing. PCHIP is used rather than a natural cubic because a natural
/// cubic can overshoot, and an overshooting transfer function is not monotone — which is a worse
/// failure than a slightly different curve. The Fritsch-Carlson limiter is what prevents it.
fn pchip_eval(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let n = xs.len();
    if n == 0 {
        return x;
    }
    if n == 1 {
        return ys[0];
    }
    if x <= xs[0] {
        let h = xs[1] - xs[0];
        return if h > 0.0 {
            ys[0] + (x - xs[0]) / h * (ys[1] - ys[0])
        } else {
            ys[0]
        };
    }
    if x >= xs[n - 1] {
        let h = xs[n - 1] - xs[n - 2];
        return if h > 0.0 {
            ys[n - 1] + (x - xs[n - 1]) / h * (ys[n - 1] - ys[n - 2])
        } else {
            ys[n - 1]
        };
    }

    // Locate the interval.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let h = xs[lo + 1] - xs[lo];
    if h <= 0.0 {
        return ys[lo];
    }
    let secant = (ys[lo + 1] - ys[lo]) / h;

    // Fritsch-Carlson: a one-sided difference is the slope, zeroed at an extremum, then limited so
    // the cubic cannot overshoot.
    let slope = |a: f32, b: f32| -> f32 {
        if a * b <= 0.0 {
            0.0
        } else {
            let m = (a + b) * 0.5;
            let limit = 3.0 * a.abs().min(b.abs());
            m.clamp(-limit, limit)
        }
    };
    let m_lo = if lo == 0 {
        secant
    } else {
        let prev = (ys[lo + 1] - ys[lo - 1]) / (xs[lo + 1] - xs[lo - 1]);
        slope(secant, prev)
    };
    let m_hi = if lo + 2 >= n {
        secant
    } else {
        let next = (ys[lo + 2] - ys[lo]) / (xs[lo + 2] - xs[lo]);
        slope(next, secant)
    };
    // Re-limit the pair together, which is the part that actually guarantees monotonicity.
    let (mut m_lo, mut m_hi) = (m_lo, m_hi);
    if secant == 0.0 {
        m_lo = 0.0;
        m_hi = 0.0;
    } else {
        let (a, b) = (m_lo / secant, m_hi / secant);
        let magnitude = a * a + b * b;
        if magnitude > 9.0 {
            let tau = 3.0 / magnitude.sqrt();
            m_lo = tau * a * secant;
            m_hi = tau * b * secant;
        }
    }

    let t = (x - xs[lo]) / h;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * ys[lo]
        + (t3 - 2.0 * t2 + t) * h * m_lo
        + (-2.0 * t3 + 3.0 * t2) * ys[lo + 1]
        + (t3 - t2) * h * m_hi
}

/// Fits one axis the way DrDre's ColorMatch does.
///
/// The structure, in the order `ColorMatch.m` does it:
///
/// 1. Clamp both clouds to the grid range — `[0, 1]` for the first slice, `±sqrt(3)` after.
/// 2. Take percentiles at `0:0.1:100` of both, and pair them: `(q_test(p), q_ref(p))`. In the
///    continuous limit that is `F_ref^-1(F_test(x))`, the same map the default builder computes.
/// 3. Dedupe on the *test* percentiles, keeping the first occurrence.
/// 4. Repair monotonicity by **deleting** pairs where the knots decrease, repeatedly.
/// 5. Add `±sqrt(3)` sentinel knots at both ends.
/// 6. Evaluate the knot curve at each uniform grid node.
///
/// Steps 3 and 4 are why this needs a repair pass at all: percentile knots from a finite sample can
/// repeat (step 3 keeps one) and can invert (step 4 deletes). The default builder avoids both by
/// construction, which is the point of comparing them.
fn fit_axis_drdre(test: &[f32], reference: &[f32], lo: f32, hi: f32, bins: usize) -> Axis {
    let nodes = bins + 1;
    let t: Vec<f32> = test.iter().map(|v| v.clamp(lo, hi)).collect();
    let r: Vec<f32> = reference.iter().map(|v| v.clamp(lo, hi)).collect();

    // 1001 percentile knots, as `0:0.1:100` gives.
    let tq = percentiles(&t, 1001);
    let rq = percentiles(&r, 1001);

    // Pair, dedupe on the test side, and repair monotonicity by deletion.
    let mut xs: Vec<f32> = Vec::with_capacity(tq.len());
    let mut ys: Vec<f32> = Vec::with_capacity(tq.len());
    for (x, y) in tq.iter().zip(rq.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if let Some(last) = xs.last()
            && *x <= *last
        {
            continue;
        }
        xs.push(*x);
        ys.push(*y);
    }
    // The original deletes decreasing pairs in a loop rather than filtering once, because removing
    // one pair can expose another inversion. Iterate until stable.
    loop {
        let mut removed = false;
        let mut i = 1;
        while i < ys.len() {
            if ys[i] < ys[i - 1] {
                xs.remove(i);
                ys.remove(i);
                removed = true;
            } else {
                i += 1;
            }
        }
        if !removed {
            break;
        }
    }

    if xs.len() < 2 {
        // Degenerate: no usable knots. Fall back to the identity so the axis is at least well
        // defined and monotone rather than undefined.
        return Axis::identity(lo, hi);
    }

    // Sentinel knots, exactly as `c1_0 = [-sqrt(3) c1_0 sqrt(3)]` does: at `±sqrt(3)` for *every*
    // slice, including the first, whose grid is `[0, 1]`. Using the grid edges instead would pin the
    // curve to the identity at both ends, which the original does not do, and would make this
    // comparison flattering rather than fair.
    let mut kx = Vec::with_capacity(xs.len() + 2);
    let mut ky = Vec::with_capacity(ys.len() + 2);
    kx.push(-CUBE_DIAG);
    ky.push(-CUBE_DIAG);
    kx.extend_from_slice(&xs);
    ky.extend_from_slice(&ys);
    kx.push(CUBE_DIAG);
    ky.push(CUBE_DIAG);

    let mut tab = [0.0f32; TABLE_NODES];
    let scale = bins as f32 / (hi - lo);
    for (j, slot) in tab.iter_mut().enumerate().take(nodes) {
        let x = lo + j as f32 / scale;
        *slot = pchip_eval(&kx, &ky, x).clamp(lo, hi);
    }
    // The sentinels make the ends exact, but a monotone guard costs nothing and protects against a
    // pathological input surviving the repair.
    for j in 1..nodes {
        if tab[j] < tab[j - 1] {
            tab[j] = tab[j - 1];
        }
    }

    Axis { lo, scale, tab }
}

/// Inverts `reference` at the probability levels `test` reaches, walking both together.
///
/// # Why not `strictly_increasing` + binary search
///
/// The obvious implementation makes the reference CDF strictly increasing by adding `i * eps` and
/// then binary-searches it. That perturbs `cb` away from `ca` by a *position-dependent* amount, and
/// wherever the CDF is flat the search lands one node early — a systematic error of one grid step
/// per flat node, on every slice. Because the sweep composes slices, that error does not average
/// out: measured on a 4:2:0 self-match, where the two CDFs are identical and the answer is exactly
/// the identity, the worst-case gain climbed from 1.0 to 36 across 32 slices.
///
/// Walking both CDFs with a monotone pointer needs no perturbation and is exact on the case that
/// matters: when `ca == cb`, advancing while `cb[k + 1] <= ca[j]` lands on `k = j` with `t = 0`, so
/// `tab[j] = grid[j]` for every node, including flat runs. Monotonicity holds because `k` only
/// advances and, within a fixed `k`, `t` rises with `ca[j]`.
#[allow(clippy::too_many_arguments)]
fn invert_into_grid(
    ca: &[f32],
    cb: &[f32],
    grid: &[f32],
    r_lo: usize,
    r_hi: usize,
    t_lo: usize,
    t_hi: usize,
    tab: &mut [f32],
) {
    let mut k = r_lo;
    for j in t_lo..=t_hi {
        while k < r_hi && cb[k + 1] <= ca[j] {
            k += 1;
        }
        let span = cb[k + 1] - cb[k];
        let t = if span > 0.0 {
            ((ca[j] - cb[k]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        tab[j] = grid[k] + t * (grid[k + 1] - grid[k]);
    }
}

/// Fits one axis' transfer function as `F_ref⁻¹ ∘ F_test` on the uniform grid.
///
/// The interior is inverted over each side's central probability band; the two tails are linear
/// ramps from the identity at the grid extremes to the interior's end values. This mirrors what
/// DrDre's implementation achieves by appending `±√3` sentinel knots, but the band is derived from
/// the data rather than hard-coded.
/// The fit settings that travel together, so [`fit_axis`] takes one coherent set rather than five
/// loose knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FitSettings {
    bins: usize,
    binning: Binning,
    builder: Builder,
    cdf_smoothing: usize,
}

fn fit_axis(test: &[f32], reference: &[f32], lo: f32, hi: f32, settings: FitSettings) -> Axis {
    let FitSettings {
        bins,
        binning,
        builder,
        cdf_smoothing,
    } = settings;
    if builder == Builder::DrDre {
        return fit_axis_drdre(test, reference, lo, hi, bins);
    }
    let nodes = bins + 1;
    let grid: Vec<f32> = (0..nodes)
        .map(|j| lo + (hi - lo) * j as f32 / bins as f32)
        .collect();
    let mut ha = histogram(test, lo, hi, bins, binning);
    let mut hb = histogram(reference, lo, hi, bins, binning);
    smooth_histogram(&mut ha, cdf_smoothing);
    smooth_histogram(&mut hb, cdf_smoothing);
    let ca = cdf_from_histogram(&ha);
    let cb = cdf_from_histogram(&hb);

    let (t_lo, t_hi) = cdf_band(&ca, nodes);
    let (r_lo, r_hi) = cdf_band(&cb, nodes);

    let mut tab = [0.0f32; TABLE_NODES];
    // Invert the reference CDF over its own band, so a probability at the band edge maps to the
    // reference's band edge rather than to the grid's extreme.
    invert_into_grid(&ca, &cb, &grid, r_lo, r_hi, t_lo, t_hi, &mut tab);

    // Tails: the identity at the grid end, meeting the fitted interior at the band edge.
    //
    // Outside the band the transform has no data, so it falls back to the identity -- that is what
    // keeps out-of-distribution content untouched instead of crushed, and it is a stated property of
    // this implementation. A straight line between the two ends is the minimal transition that keeps
    // it.
    //
    // The obvious improvement is a cubic Hermite that also matches the interior's *slope* at the band
    // edge, removing the slope discontinuity a 3D LUT cannot resolve. It was implemented, measured and
    // rejected: it improves the LUT bake but costs tail fidelity, because a curve starting at the
    // interior's slope deviates further from the identity mid-tail than a straight line does -- and
    // leaving unfitted content alone is the property the tails exist to provide.
    //
    // The variant is not in the tree, so the figures from that measurement are not reproducible here
    // and are deliberately not quoted. The decision stands on the reasoning: the kink is a limitation
    // of the LUT path, not something the native path should pay for.
    fill_tail(tab.as_mut_slice(), &grid, 0, t_lo, Tail::Left);
    fill_tail(tab.as_mut_slice(), &grid, t_hi + 1, nodes, Tail::Right);

    // Monotone guard against pathological inputs.
    for i in 1..tab.len() {
        if tab[i] < tab[i - 1] {
            tab[i] = tab[i - 1];
        }
    }

    let scale = bins as f32 / (hi - lo);
    Axis { lo, scale, tab }
}

/// Applies one slice's correction to a cloud in place.
fn apply_slice(cloud: &mut [[f32; 3]], slice: &Slice) {
    let m = &slice.rot;
    for px in cloud.iter_mut() {
        let (r, g, b) = (px[0], px[1], px[2]);
        let u0 = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        let u1 = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        let u2 = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        let d0 = slice.axes[0].eval(u0) - u0;
        let d1 = slice.axes[1].eval(u1) - u1;
        let d2 = slice.axes[2].eval(u2) - u2;
        px[0] = r + m[0][0] * d0 + m[1][0] * d1 + m[2][0] * d2;
        px[1] = g + m[0][1] * d0 + m[1][1] * d1 + m[2][1] * d2;
        px[2] = b + m[0][2] * d0 + m[1][2] * d1 + m[2][2] * d2;
    }
}

/// Mean sliced-Wasserstein-1 distance between two clouds, averaged over `n` directions.
pub fn sliced_w1(a: &[[f32; 3]], b: &[[f32; 3]], n: usize) -> f32 {
    if a.is_empty() || b.is_empty() || n == 0 {
        return 0.0;
    }
    let dirs = fibonacci_directions(n);
    let mut total = 0.0f32;
    let mut scratch_a: Vec<f32> = Vec::with_capacity(a.len());
    let mut scratch_b: Vec<f32> = Vec::with_capacity(b.len());
    for d in dirs {
        scratch_a.clear();
        scratch_b.clear();
        scratch_a.extend(a.iter().map(|p| p[0] * d[0] + p[1] * d[1] + p[2] * d[2]));
        scratch_b.extend(b.iter().map(|p| p[0] * d[0] + p[1] * d[1] + p[2] * d[2]));
        scratch_a.sort_by(f32::total_cmp);
        scratch_b.sort_by(f32::total_cmp);
        let m = scratch_a.len().min(scratch_b.len());
        let mut acc = 0.0f64;
        for i in 0..m {
            let ia = i * scratch_a.len() / m;
            let ib = i * scratch_b.len() / m;
            acc += (scratch_a[ia] - scratch_b[ib]).abs() as f64;
        }
        total += (acc / m as f64) as f32;
    }
    total / n as f32
}

/// Estimates the largest local gain (max row-sum of |J|) of the model on a cloud.
pub fn estimate_max_gain(model: &Model, cloud: &[[f32; 3]], stride: usize) -> f32 {
    if model.is_empty() || cloud.is_empty() {
        return 1.0;
    }
    const D: f32 = 2e-4;
    let mut worst = 0.0f32;
    let mut probe: Vec<[f32; 3]> = Vec::with_capacity(3);
    for (idx, p) in cloud.iter().enumerate() {
        if idx % stride.max(1) != 0 {
            continue;
        }
        let base = crate::apply::apply_point(model, *p);
        for c in 0..3 {
            let mut q = *p;
            q[c] += D;
            probe.clear();
            let shifted = crate::apply::apply_point(model, q);
            for k in 0..3 {
                worst = worst.max((shifted[k] - base[k]).abs() / D);
            }
        }
    }
    worst
}

/// Builds a model from a pooled test cloud and a pooled reference cloud.
///
/// `test` is consumed as the running accumulator; `reference` is read-only.
pub fn build_model(
    test: &Samples,
    reference: &Samples,
    options: &BuildOptions,
) -> Result<Model, BuildError> {
    // The table is a fixed-size array in the model format, so `bins` cannot exceed its node count.
    // Checked here because the format's size is not a caller's business.
    let max_bins = TABLE_NODES - 1;
    if options.bins > max_bins {
        return Err(BuildError::BinsTooLarge {
            bins: options.bins,
            max: max_bins,
        });
    }
    if test.is_empty() || reference.is_empty() {
        return Err(BuildError::NoSamples);
    }
    if options.directions == 0 {
        return Err(BuildError::NoDirections);
    }
    if options.bins < 16 {
        return Err(BuildError::BinsTooSmall(options.bins));
    }

    let started = std::time::Instant::now();
    let rotations = build_rotations(options)?;
    let bins = options.bins;
    let step = options.step;
    let passes = options.passes.max(1);

    let mut cloud = test.rgb.clone();
    // The sliced-W1 numbers are informational, so they are measured on a subsample with fewer
    // directions: sorting the full 260k-pixel cloud 64 times dominated the whole build, and 16
    // directions over <=20k pixels is statistically ample for a report.
    let probe = |c: &[[f32; 3]]| -> Vec<[f32; 3]> {
        if c.len() <= 20_000 {
            c.to_vec()
        } else {
            let step = c.len() / 20_000;
            c.iter().step_by(step).copied().collect()
        }
    };
    let test_probe = probe(&test.rgb);
    let ref_probe = probe(&reference.rgb);
    let w1_before = sliced_w1(&test_probe, &ref_probe, 16);

    let mut slices: Vec<Slice> = Vec::with_capacity(rotations.len() * passes);
    let mut buf_test: Vec<f32> = Vec::with_capacity(cloud.len());
    let mut buf_ref: Vec<f32> = Vec::with_capacity(reference.rgb.len());

    for _ in 0..passes {
        for (k, rot) in rotations.iter().enumerate() {
            // The first slice keeps the original's `[0,1]` convention; rotated axes use ±√3.
            let (lo, hi) = if k == 0 {
                (0.0, 1.0)
            } else {
                (-CUBE_DIAG, CUBE_DIAG)
            };

            buf_test.clear();
            buf_ref.clear();
            buf_test.extend(
                cloud
                    .iter()
                    .map(|p| p[0] * rot[0][0] + p[1] * rot[0][1] + p[2] * rot[0][2]),
            );
            buf_ref.extend(
                reference
                    .rgb
                    .iter()
                    .map(|p| p[0] * rot[0][0] + p[1] * rot[0][1] + p[2] * rot[0][2]),
            );
            let ax0 = fit_axis(
                &buf_test,
                &buf_ref,
                lo,
                hi,
                FitSettings {
                    bins,
                    binning: options.binning,
                    builder: options.builder,
                    cdf_smoothing: options.cdf_smoothing,
                },
            );

            buf_test.clear();
            buf_ref.clear();
            buf_test.extend(
                cloud
                    .iter()
                    .map(|p| p[0] * rot[1][0] + p[1] * rot[1][1] + p[2] * rot[1][2]),
            );
            buf_ref.extend(
                reference
                    .rgb
                    .iter()
                    .map(|p| p[0] * rot[1][0] + p[1] * rot[1][1] + p[2] * rot[1][2]),
            );
            let ax1 = fit_axis(
                &buf_test,
                &buf_ref,
                lo,
                hi,
                FitSettings {
                    bins,
                    binning: options.binning,
                    builder: options.builder,
                    cdf_smoothing: options.cdf_smoothing,
                },
            );

            buf_test.clear();
            buf_ref.clear();
            buf_test.extend(
                cloud
                    .iter()
                    .map(|p| p[0] * rot[2][0] + p[1] * rot[2][1] + p[2] * rot[2][2]),
            );
            buf_ref.extend(
                reference
                    .rgb
                    .iter()
                    .map(|p| p[0] * rot[2][0] + p[1] * rot[2][1] + p[2] * rot[2][2]),
            );
            let ax2 = fit_axis(
                &buf_test,
                &buf_ref,
                lo,
                hi,
                FitSettings {
                    bins,
                    binning: options.binning,
                    builder: options.builder,
                    cdf_smoothing: options.cdf_smoothing,
                },
            );

            let slice = Slice {
                rot: *rot,
                axes: [ax0, ax1, ax2],
            };
            if (step - 1.0).abs() < f32::EPSILON {
                apply_slice(&mut cloud, &slice);
            } else {
                // Damped step: interpolate the correction toward the current state.
                let before = cloud.clone();
                apply_slice(&mut cloud, &slice);
                for (c, b) in cloud.iter_mut().zip(before.iter()) {
                    c[0] = b[0] + (c[0] - b[0]) * step;
                    c[1] = b[1] + (c[1] - b[1]) * step;
                    c[2] = b[2] + (c[2] - b[2]) * step;
                }
            }
            slices.push(slice);
        }
    }

    let w1_after = sliced_w1(&probe(&cloud), &ref_probe, 16);
    let mut model = Model {
        version: FORMAT_VERSION,
        nodes: TABLE_NODES as u32,
        test_space: crate::color::SpaceInfo::WORKING,
        ref_space: crate::color::SpaceInfo::WORKING,
        stats: ModelStats {
            samples: (test.len() + reference.len()) as u64,
            frames: test.frames,
            w1_before,
            w1_after,
            max_gain: 1.0,
            build_ms: 0.0,
            directions: slices.len() as u32,
        },
        slices,
    };
    // Gain estimation is a finite-difference probe; subsample so it stays cheap on big clouds.
    let stride = (model.len() / 3).max(1);
    model.stats.max_gain = estimate_max_gain(&model, &cloud, stride);
    model.stats.build_ms = started.elapsed().as_secs_f32() * 1000.0;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so tests never depend on the `rand` crate.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32) / (u32::MAX as f32)
        }
        fn gauss(&mut self) -> f32 {
            // Sum of uniforms: good enough for synthetic test clouds.
            let s: f32 = (0..4).map(|_| self.next_f32()).sum();
            (s - 2.0) * 1.5
        }
    }

    fn synthetic(n: usize, seed: u64) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut r = Rng(seed);
        let mut test = Vec::with_capacity(n);
        for _ in 0..n {
            test.push([
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
            ]);
        }
        // A cross-channel look: 3x3 mixing plus per-channel gamma. Cannot be matched by three
        // independent 1-D LUTs, which is exactly why rotated directions are needed.
        let m = [
            [0.86f32, 0.10, 0.04],
            [0.06, 0.90, 0.04],
            [0.02, 0.12, 0.86],
        ];
        let gamma = [1.15f32, 1.00, 0.92];
        let lift = [0.01f32, -0.01, 0.02];
        let reference = test
            .iter()
            .map(|p| {
                let mut y = [0.0f32; 3];
                for i in 0..3 {
                    y[i] = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + lift[i];
                }
                [
                    y[0].clamp(0.0, 1.0).powf(gamma[0]),
                    y[1].clamp(0.0, 1.0).powf(gamma[1]),
                    y[2].clamp(0.0, 1.0).powf(gamma[2]),
                ]
            })
            .collect();
        (test, reference)
    }

    #[test]
    fn fibonacci_directions_are_unit_vectors() {
        for n in [1usize, 2, 8, 32, 64] {
            for d in fibonacci_directions(n) {
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-5, "n={n} len={len}");
            }
        }
    }

    #[test]
    fn frames_are_orthonormal_and_right_handed() {
        for d in fibonacci_directions(64) {
            let f = frame_from_direction(d);
            // Rows orthonormal.
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f32 = (0..3).map(|k| f[i][k] * f[j][k]).sum();
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expect).abs() < 1e-5, "d={d:?} dot={dot}");
                }
            }
            // First row is the direction.
            assert!((f[0][0] - d[0]).abs() < 1e-5);
            let slice = Slice {
                rot: f,
                axes: [
                    Axis::identity(-CUBE_DIAG, CUBE_DIAG),
                    Axis::identity(-CUBE_DIAG, CUBE_DIAG),
                    Axis::identity(-CUBE_DIAG, CUBE_DIAG),
                ],
            };
            assert!(slice.orthonormality_error() < 1e-5);
            assert!(
                (slice.determinant() - 1.0).abs() < 1e-4,
                "det={}",
                slice.determinant()
            );
        }
    }

    #[test]
    fn splat_histogram_preserves_total_mass() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let h = splat_histogram(&values, 0.0, 1.0, 256);
        let total: f32 = h.iter().sum();
        assert!((total - 1000.0).abs() < 1e-2, "total={total}");
        assert_eq!(h.len(), 257);
    }

    #[test]
    fn percentiles_match_linear_interpolation_between_order_statistics() {
        let v: Vec<f32> = (0..=10).map(|i| i as f32).collect();
        let p = percentiles(&v, 11);
        assert_eq!(p.len(), 11);
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[5] - 5.0).abs() < 1e-6);
        assert!((p[10] - 10.0).abs() < 1e-6);

        // Midpoints interpolate rather than snap.
        let mid = percentiles(&v, 21);
        assert!((mid[1] - 0.5).abs() < 1e-6, "{}", mid[1]);

        // Degenerate inputs must not panic.
        assert!(percentiles(&[], 11).is_empty());
        assert!(percentiles(&[1.0], 0).is_empty());
        assert_eq!(percentiles(&[f32::NAN, f32::NAN], 5).len(), 0);
        assert_eq!(percentiles(&[2.0], 3), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn pchip_is_monotone_and_passes_through_its_knots() {
        let xs = vec![0.0f32, 0.1, 0.5, 0.9, 1.0];
        let ys = vec![0.0f32, 0.9, 0.92, 0.95, 1.0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert!((pchip_eval(&xs, &ys, *x) - y).abs() < 1e-4, "at {x}");
        }
        // Monotone across the whole span, including between knots.
        let mut prev = f32::MIN;
        let mut x = -0.5f32;
        while x <= 1.5 {
            let y = pchip_eval(&xs, &ys, x);
            assert!(y >= prev - 1e-5, "not monotone at {x}: {prev} -> {y}");
            prev = y;
            x += 0.001;
        }
        // Extrapolates linearly rather than clamping.
        assert!(pchip_eval(&xs, &ys, -0.5) < 0.0);
        assert!(pchip_eval(&xs, &ys, 1.5) > 1.0);
        // Degenerate inputs.
        assert_eq!(pchip_eval(&[], &[], 0.5), 0.5);
        assert_eq!(pchip_eval(&[1.0], &[0.25], 0.5), 0.25);
    }

    #[test]
    fn drdre_builder_is_monotone_and_survives_degenerate_input() {
        let mut r = Rng(0xD12D);
        let test: Vec<f32> = (0..5_000).map(|_| r.next_f32()).collect();
        let reference: Vec<f32> = test.iter().map(|v| v * 0.8 + 0.1).collect();
        for (lo, hi) in [(0.0f32, 1.0f32), (-CUBE_DIAG, CUBE_DIAG)] {
            let axis = fit_axis_drdre(&test, &reference, lo, hi, 256);
            assert!(axis.is_monotone(1e-4), "not monotone on [{lo}, {hi}]");
            assert!(axis.tab.iter().all(|v| v.is_finite()));
        }

        // Every sample identical: there is nothing to fit, and the identity is the only safe answer.
        let flat = vec![0.5f32; 1_000];
        let axis = fit_axis_drdre(&flat, &flat, 0.0, 1.0, 256);
        assert!(axis.is_monotone(1e-4));
        for j in [0usize, 1, 128, 255, 256] {
            assert!((axis.tab[j] - j as f32 / 256.0).abs() < 1e-4, "node {j}");
        }

        // All non-finite: same.
        let nan = vec![f32::NAN; 100];
        assert!(fit_axis_drdre(&nan, &nan, 0.0, 1.0, 64).is_monotone(1e-4));

        // Empty input.
        assert!(fit_axis_drdre(&[], &[], 0.0, 1.0, 64).is_monotone(1e-4));
    }

    #[test]
    fn drdre_self_match_is_near_the_identity() {
        // Percentile-paired knots from the same cloud lie on the identity, so the fitted axis should
        // reproduce it. The residual is the builder's own error, and it is the one case where the
        // right answer is known exactly.
        let mut r = Rng(0x5EED);
        let cloud: Vec<f32> = (0..20_000).map(|_| r.next_f32()).collect();
        let axis = fit_axis_drdre(&cloud, &cloud, 0.0, 1.0, 256);
        let mut worst = 0.0f32;
        for j in 0..257 {
            let x = j as f32 / 256.0;
            worst = worst.max((axis.tab[j] - x).abs());
        }
        assert!(worst < 5e-3, "self-match residual {worst} is too large");
    }

    #[test]
    fn the_two_builders_differ_but_both_reduce_the_distance() {
        let mut r = Rng(0xB0B0);
        let test: Vec<[f32; 3]> = (0..20_000)
            .map(|_| [r.next_f32(), r.next_f32(), r.next_f32()])
            .collect();
        let reference: Vec<[f32; 3]> = test
            .iter()
            .map(|p| {
                [
                    (p[0] * 0.8 + 0.1).clamp(0.0, 1.0),
                    p[1].clamp(0.0, 1.0),
                    (p[2] * 1.1).clamp(0.0, 1.0),
                ]
            })
            .collect();

        let splat = build_model(
            &Samples::new(test.clone(), 1),
            &Samples::new(reference.clone(), 1),
            &BuildOptions {
                directions: 16,
                builder: Builder::Splat,
                ..Default::default()
            },
        )
        .unwrap();
        let drdre = build_model(
            &Samples::new(test, 1),
            &Samples::new(reference, 1),
            &BuildOptions {
                directions: 16,
                builder: Builder::DrDre,
                ..Default::default()
            },
        )
        .unwrap();

        for (name, m) in [("splat", &splat), ("drdre", &drdre)] {
            assert!(
                m.stats.w1_after < m.stats.w1_before * 0.5,
                "{name}: did not reduce the distance ({} -> {})",
                m.stats.w1_before,
                m.stats.w1_after
            );
            assert!(
                m.stats.max_gain.is_finite() && m.stats.max_gain > 0.0,
                "{name}"
            );
            for slice in &m.slices {
                for axis in &slice.axes {
                    assert!(axis.is_monotone(1e-4), "{name}: non-monotone table");
                }
            }
        }
        assert_ne!(
            splat.slices, drdre.slices,
            "the two builders must not produce the same model"
        );
    }

    #[test]
    fn the_tails_fall_back_to_the_identity() {
        // The stated purpose of the tails: content the model was not fitted on passes through
        // untouched. This is the property a smoother tail would trade away -- a cubic matching the
        // interior's slope deviates further from the identity mid-tail, and measured 2.6x worse on
        // held-out scenes -- so it is worth asserting rather than assuming.
        let mut r = Rng(0x7A11);
        // A deliberately narrow reference cloud, so most of the grid is tail.
        let reference: Vec<[f32; 3]> = (0..6_000)
            .map(|_| {
                [
                    0.45 + 0.02 * r.next_f32(),
                    0.45 + 0.02 * r.next_f32(),
                    0.45 + 0.02 * r.next_f32(),
                ]
            })
            .collect();
        let test: Vec<[f32; 3]> = (0..6_000)
            .map(|_| {
                [
                    0.45 + 0.02 * r.next_f32(),
                    0.45 + 0.02 * r.next_f32(),
                    0.45 + 0.02 * r.next_f32(),
                ]
            })
            .collect();
        let m = build_model(
            &Samples::new(test, 1),
            &Samples::new(reference, 1),
            &BuildOptions::default(),
        )
        .unwrap();

        // Far outside the fitted band, the transform must be close to the identity.
        for &v in &[0.02f32, 0.05, 0.95, 0.98] {
            let out = crate::apply::apply_point(&m, [v, v, v]);
            let drift = (out[0] - v).abs();
            assert!(drift < 0.02, "at {v}: moved to {} (drift {drift})", out[0]);
        }

        // And every axis stays monotone, which the tails must not break.
        for slice in &m.slices {
            for axis in &slice.axes {
                assert!(axis.is_monotone(1e-4), "tail broke monotonicity");
            }
        }
    }

    #[test]
    fn pooling_frames_beats_fitting_one() {
        // The headline quality claim: one model built from many frames generalises better than one
        // built from a single frame, and does not overfit the frames it saw. Kept as a test, not just
        // a figure in the docs, because it is the reason the wrapper pools at all.
        let look = |x: [f32; 3]| -> [f32; 3] {
            let m = [
                [0.86f32, 0.10, 0.04],
                [0.06, 0.90, 0.04],
                [0.02, 0.12, 0.86],
            ];
            let mut o = [0.0f32; 3];
            for c in 0..3 {
                let v = m[c][0] * x[0] + m[c][1] * x[1] + m[c][2] * x[2];
                o[c] = v.clamp(0.0, 1.0);
            }
            o
        };
        let mut r = Rng(0xA11CE);
        let scenes: Vec<Vec<[f32; 3]>> = (0..6)
            .map(|i| {
                let exposure = 0.6 + 0.5 * i as f32 / 5.0;
                (0..4_000)
                    .map(|_| {
                        let mut p = [0.0f32; 3];
                        for c in p.iter_mut() {
                            *c = ((0.3 + 0.25 * (r.next_f32() * 2.0 - 1.0)) * exposure)
                                .clamp(0.0, 1.0);
                        }
                        p
                    })
                    .collect()
            })
            .collect();
        let held_out: Vec<Vec<[f32; 3]>> = (0..3)
            .map(|i| {
                let exposure = 0.7 + 0.4 * i as f32 / 2.0;
                (0..4_000)
                    .map(|_| {
                        let mut p = [0.0f32; 3];
                        for c in p.iter_mut() {
                            *c = ((0.3 + 0.25 * (r.next_f32() * 2.0 - 1.0)) * exposure)
                                .clamp(0.0, 1.0);
                        }
                        p
                    })
                    .collect()
            })
            .collect();

        let single = build_model(
            &Samples::new(scenes[0].clone(), 1),
            &Samples::new(scenes[0].iter().copied().map(look).collect(), 1),
            &BuildOptions::default(),
        )
        .unwrap();
        let pooled_test: Vec<[f32; 3]> = scenes.iter().flatten().copied().collect();
        let pooled = build_model(
            &Samples::new(pooled_test.clone(), 6),
            &Samples::new(pooled_test.iter().copied().map(look).collect(), 6),
            &BuildOptions::default(),
        )
        .unwrap();

        let mean_error = |model: &Model, frames: &[Vec<[f32; 3]>]| -> f32 {
            let mut total = 0.0f32;
            for f in frames {
                let reference: Vec<[f32; 3]> = f.iter().copied().map(look).collect();
                let moved: Vec<[f32; 3]> = f
                    .iter()
                    .map(|&p| {
                        let o = crate::apply::apply_point(model, p);
                        [
                            o[0].clamp(0.0, 1.0),
                            o[1].clamp(0.0, 1.0),
                            o[2].clamp(0.0, 1.0),
                        ]
                    })
                    .collect();
                total += sliced_w1(&moved, &reference, 16);
            }
            total / frames.len() as f32
        };

        let single_seen = mean_error(&single, &scenes);
        let pooled_seen = mean_error(&pooled, &scenes);
        let single_unseen = mean_error(&single, &held_out);
        let pooled_unseen = mean_error(&pooled, &held_out);

        assert!(
            pooled_seen < single_seen * 0.75,
            "pooling should clearly beat one frame on the frames it saw: {pooled_seen} vs {single_seen}"
        );
        assert!(
            pooled_unseen < single_unseen * 0.75,
            "pooling should beat one frame on held-out scenes too: {pooled_unseen} vs {single_unseen}"
        );
        // No overfitting: the held-out error must not be much worse than the build-set error.
        assert!(
            pooled_unseen < pooled_seen * 1.5,
            "pooled model overfits: seen {pooled_seen}, unseen {pooled_unseen}"
        );
    }

    #[test]
    fn every_supported_bin_count_fits_without_panicking() {
        // `bins` is user-facing and documented as saturating at 64, so 64 has to actually work. The
        // band used to be able to end on the last CDF node, and the inversion then read `c[k + 1]`
        // one past the end.
        let mut r = Rng(0x5EED_1234);
        let cloud: Vec<[f32; 3]> = (0..40_000)
            .map(|_| [r.next_f32(), r.next_f32(), r.next_f32()])
            .collect();
        for bins in [16usize, 17, 32, 64, 65, 128, 255, 256] {
            for k in [1usize, 4, 32] {
                let m = build_model(
                    &Samples::new(cloud.clone(), 1),
                    &Samples::new(cloud.clone(), 1),
                    &BuildOptions {
                        directions: k,
                        bins,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|e| panic!("bins={bins} K={k}: {e}"));
                assert!(
                    m.stats.w1_after < 1e-4,
                    "bins={bins} K={k}: w1_after={}",
                    m.stats.w1_after
                );
                for s in &m.slices {
                    for a in &s.axes {
                        assert!(a.is_monotone(1e-4), "bins={bins} K={k}: table not monotone");
                    }
                }
            }
        }
    }

    #[test]
    fn an_over_fine_histogram_is_rejected_not_truncated() {
        // The model format stores a fixed 257-node table, so `bins > 256` has nowhere to go. It must
        // be an error: silently clamping would make `bins=512` and `bins=256` produce the same model
        // while reporting different settings.
        let mut r = Rng(7);
        let cloud: Vec<[f32; 3]> = (0..4_000)
            .map(|_| [r.next_f32(), r.next_f32(), r.next_f32()])
            .collect();
        for bins in [257usize, 512, 4096] {
            let e = build_model(
                &Samples::new(cloud.clone(), 1),
                &Samples::new(cloud.clone(), 1),
                &BuildOptions {
                    bins,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(
                matches!(e, BuildError::BinsTooLarge { bins: b, max: 256 } if b == bins),
                "bins={bins}: got {e}"
            );
        }
        // 256 itself must still work.
        build_model(
            &Samples::new(cloud.clone(), 1),
            &Samples::new(cloud, 1),
            &BuildOptions {
                bins: 256,
                ..Default::default()
            },
        )
        .expect("bins=256 is the format's limit and must be accepted");
    }

    #[test]
    fn smoothing_preserves_total_and_bounds_slope() {
        let mut h = vec![0.0f32; 33];
        h[16] = 100.0; // a pure atom
        let before: f32 = h.iter().sum();
        smooth_histogram(&mut h, CDF_SMOOTH_PASSES);
        let after: f32 = h.iter().sum();
        assert!(
            (before - after).abs() < 1e-3,
            "mass not preserved: {before} -> {after}"
        );

        // The atom must be spread over several bins so that the CDF no longer steps.
        let nonzero = h.iter().filter(|v| **v > 0.0).count();
        assert!(nonzero >= 3, "atom was not spread: {nonzero} non-zero bins");

        // Every step is now bounded, so the inverted map's slope is bounded too.
        let max_step = h
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < before * 0.5,
            "a step remains: {max_step} vs total {before}"
        );
    }

    #[test]
    fn duplicated_values_do_not_blow_up_the_gain() {
        // This is the shape chroma upsampling produces: each source sample repeated four times.
        let mut r = Rng(0xC0FFEE);
        let mut cloud = Vec::new();
        for _ in 0..5_000 {
            let p = [
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
                (0.3 + 0.28 * r.gauss()).clamp(0.0, 1.0),
            ];
            for _ in 0..4 {
                cloud.push(p);
            }
        }
        for k in [1usize, 4, 16, 32] {
            let m = build_model(
                &Samples::new(cloud.clone(), 1),
                &Samples::new(cloud.clone(), 1),
                &BuildOptions {
                    directions: k,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                m.stats.max_gain < 3.0,
                "K={k}: a self-match on duplicated values should stay near the identity, got gain {}",
                m.stats.max_gain
            );
            assert!(
                m.stats.w1_after < 1e-4,
                "K={k}: w1_after={}",
                m.stats.w1_after
            );
        }
    }

    #[test]
    fn splat_histogram_ignores_non_finite() {
        let h = splat_histogram(&[f32::NAN, f32::INFINITY, 0.5], 0.0, 1.0, 256);
        let total: f32 = h.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "total={total}");
    }

    #[test]
    fn cdf_is_monotone_and_normalised() {
        let values: Vec<f32> = (0..500).map(|i| (i as f32 / 500.0).powi(2)).collect();
        let c = cdf_from_histogram(&splat_histogram(&values, 0.0, 1.0, 128));
        assert!((c[c.len() - 1] - 1.0).abs() < 1e-6);
        assert!(c.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn build_matches_a_cross_channel_look() {
        let (test, reference) = synthetic(60_000, 0x1234_5678);
        let opts = BuildOptions {
            directions: 32,
            ..Default::default()
        };
        let model = build_model(
            &Samples::new(test.clone(), 8),
            &Samples::new(reference.clone(), 8),
            &opts,
        )
        .expect("build");

        assert_eq!(model.len(), 32);
        let before = model.stats.w1_before;
        let after = model.stats.w1_after;
        assert!(after < before, "w1 did not improve: {before} -> {after}");
        let reduction = 1.0 - after / before;
        assert!(
            reduction > 0.95,
            "expected >=95% reduction, got {:.1}%",
            reduction * 100.0
        );

        // The applied cloud must actually move toward the reference.
        let mut applied = test.clone();
        for s in &model.slices {
            apply_slice(&mut applied, s);
        }
        let check = sliced_w1(&applied, &reference, 64);
        assert!(
            check < before * 0.1,
            "applied cloud still far: {check} vs {before}"
        );
    }

    #[test]
    fn more_directions_match_better() {
        let (test, reference) = synthetic(20_000, 0xABCD);
        let mut prev = f32::INFINITY;
        for k in [1usize, 8, 32] {
            let opts = BuildOptions {
                directions: k,
                ..Default::default()
            };
            let m = build_model(
                &Samples::new(test.clone(), 1),
                &Samples::new(reference.clone(), 1),
                &opts,
            )
            .unwrap();
            assert!(
                m.stats.w1_after <= prev + 1e-4,
                "K={k} regressed: {} vs {prev}",
                m.stats.w1_after
            );
            prev = m.stats.w1_after;
        }
        // K=1 alone cannot fix a cross-channel look.
        let one = build_model(
            &Samples::new(test.clone(), 1),
            &Samples::new(reference.clone(), 1),
            &BuildOptions {
                directions: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            one.stats.w1_after > 0.005,
            "K=1 unexpectedly solved it: {}",
            one.stats.w1_after
        );
    }

    #[test]
    fn tables_are_monotone_and_orthonormal() {
        let (test, reference) = synthetic(10_000, 7);
        let m = build_model(
            &Samples::new(test, 1),
            &Samples::new(reference, 1),
            &BuildOptions::default(),
        )
        .unwrap();
        for (i, s) in m.slices.iter().enumerate() {
            assert!(s.orthonormality_error() < 1e-4, "slice {i}");
            for (a, ax) in s.axes.iter().enumerate() {
                assert!(ax.is_monotone(0.0), "slice {i} axis {a} not monotone");
                assert_eq!(ax.nodes(), TABLE_NODES);
            }
        }
    }

    #[test]
    fn empty_or_degenerate_inputs_are_rejected() {
        let opts = BuildOptions::default();
        assert_eq!(
            build_model(&Samples::default(), &Samples::new(vec![[0.0; 3]], 1), &opts),
            Err(BuildError::NoSamples)
        );
        assert_eq!(
            build_model(&Samples::new(vec![[0.0; 3]], 1), &Samples::default(), &opts),
            Err(BuildError::NoSamples)
        );
        let s = Samples::new(vec![[0.1, 0.2, 0.3]], 1);
        assert_eq!(
            build_model(
                &s,
                &s,
                &BuildOptions {
                    directions: 0,
                    ..Default::default()
                }
            ),
            Err(BuildError::NoDirections)
        );
        assert_eq!(
            build_model(
                &s,
                &s,
                &BuildOptions {
                    bins: 4,
                    ..Default::default()
                }
            ),
            Err(BuildError::BinsTooSmall(4))
        );
    }

    #[test]
    fn explicit_axes_are_honoured_and_reproducible() {
        let (test, reference) = synthetic(5_000, 99);
        let dirs: Vec<[f32; 3]> = (0..4)
            .map(|i| {
                let a = i as f32 * 0.4;
                [a.cos(), a.sin(), 0.3]
            })
            .collect();
        let opts = BuildOptions {
            directions: 4,
            axes: AxesScheme::Explicit(dirs),
            ..Default::default()
        };
        let a = build_model(
            &Samples::new(test.clone(), 1),
            &Samples::new(reference.clone(), 1),
            &opts,
        )
        .unwrap();
        let b = build_model(&Samples::new(test, 1), &Samples::new(reference, 1), &opts).unwrap();
        // `build_ms` is a duration and therefore differs between runs; everything else must match
        // exactly, including every transfer-table entry.
        assert_eq!(
            a.slices, b.slices,
            "explicit axes must be fully reproducible"
        );
        assert_eq!(a.stats.samples, b.stats.samples);
        assert_eq!(a.stats.w1_before, b.stats.w1_before);
        assert_eq!(a.stats.w1_after, b.stats.w1_after);
        assert_eq!(a.stats.max_gain, b.stats.max_gain);

        let bad = BuildOptions {
            directions: 8,
            axes: AxesScheme::Explicit(vec![[1.0, 0.0, 0.0]; 2]),
            ..Default::default()
        };
        assert!(matches!(
            build_model(
                &Samples::new(vec![[0.1; 3]], 1),
                &Samples::new(vec![[0.2; 3]], 1),
                &bad
            ),
            Err(BuildError::BadAxes(_))
        ));
    }

    #[test]
    fn identity_reference_yields_near_identity_model() {
        let (test, _) = synthetic(20_000, 4242);
        let opts = BuildOptions {
            directions: 16,
            ..Default::default()
        };
        let m = build_model(
            &Samples::new(test.clone(), 1),
            &Samples::new(test.clone(), 1),
            &opts,
        )
        .unwrap();
        assert!(
            m.stats.w1_before < 1e-6,
            "identical clouds should have zero distance"
        );
        assert!(
            m.stats.w1_after < 1e-4,
            "an identity reference must leave the cloud alone, got w1_after={}",
            m.stats.w1_after
        );
        // Applying it must not move pixels appreciably.
        //
        // The residual is a discretisation floor, not a bug. This cloud is clamped to `[0, 1]`, so
        // its distribution carries atoms at both ends; with 257 table nodes and 16 composed slices
        // the measured worst case is ~4 LSB at 8-bit while the mean stays below 0.02 LSB. The floor
        // grows with the slice count (see `tests/identity_floor.rs` for the measured curve).
        let mut cloud = test[..2000].to_vec();
        let before = cloud.clone();
        for s in &m.slices {
            apply_slice(&mut cloud, s);
        }
        let worst = before
            .iter()
            .zip(cloud.iter())
            .map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max))
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "identity match moved pixels by {worst}");
    }

    #[test]
    fn model_survives_serialization_after_build() {
        let (test, reference) = synthetic(5_000, 31);
        let m = build_model(
            &Samples::new(test, 1),
            &Samples::new(reference, 1),
            &BuildOptions {
                directions: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let back = Model::from_bytes(&m.to_bytes()).expect("round trip");
        assert_eq!(back, m);
    }

    #[test]
    fn build_is_deterministic() {
        let (test, reference) = synthetic(8_000, 5150);
        let opts = BuildOptions {
            directions: 12,
            ..Default::default()
        };
        let a = build_model(
            &Samples::new(test.clone(), 1),
            &Samples::new(reference.clone(), 1),
            &opts,
        )
        .unwrap();
        let b = build_model(&Samples::new(test, 1), &Samples::new(reference, 1), &opts).unwrap();
        assert_eq!(a.slices, b.slices);
        assert_eq!(a.stats.w1_after, b.stats.w1_after);
        assert_eq!(a.stats.max_gain, b.stats.max_gain);
    }
}
