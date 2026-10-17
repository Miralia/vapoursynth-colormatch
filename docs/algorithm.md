# Algorithm

## The transform

A model is `K` slices. Each projects RGB onto three orthonormal axes, applies an independent monotone
1-D transfer function per axis, and rotates the correction back:

```
for k in 0..K:
    u = R_k · rgb
    y = T_k(u)                # componentwise, monotone
    rgb += R_kᵀ · (y − u)
```

`R_k` is orthonormal, so `R_kᵀ = R_k⁻¹` and rotating back is exact. Slice 0 uses the identity rotation
and a `[0, 1]` grid; later slices use other rotations and `[−√3, +√3]`, which bounds `|a·p|` for
`p ∈ [0, 1]³` and unit `a`.

This is a sequential sliced-Wasserstein gradient flow with unit step size. For a direction `θ` the
gradient of the sliced-W₂ functional is `(T_θ(u) − u)·θ`; each slice takes a full step along three
orthogonal directions at once.

## Building

The reference cloud is modified by nothing; only the accumulator is. The tables are refitted from the
*current* accumulator at every slice:

```
X ← test samples
Y ← reference samples        # read-only
for k in 0..K:
    T_k ← fit(X·R_k, Y·R_k)
    X   ← X + (T_k(X·R_k) − X·R_k)·R_kᵀ
```

Fitting against a stale accumulator diverges: building the tables once per pass instead of once per
slice grows the sliced-W₁ distance by 500%+.

### The 1-D transfer function

`T_k` approximates `F_ref⁻¹ ∘ F_test` on a uniform grid.

**Linear-splatted histograms.** Each sample contributes `1−f` and `f` to adjacent bins, which is
equivalent to interpolating the empirical CDF. The result is continuous, so `T_k` is continuous and
monotone by construction — no post-hoc repair pass. This is a structural choice, not an accuracy win:
with the CDF smoothing applied anyway, it agrees with hard binning to three significant figures
(0.00015 vs 0.00016 pooled sliced-W₁ at `K=32`).

**CDF smoothing.** The splat removes the staircase from the CDF's interior but not its atoms, and
inverting a stepped CDF is unconstrained — the map may be arbitrarily steep across a step. Two passes
of a `[1, 2, 1]/4` kernel turn steps into short ramps and bound the slope. This low-passes the
histogram's derivative, not the distribution.

How much it matters depends on how coarsely the input is quantised. On synthetic clouds against a
per-channel look:

| distinct values per channel | worst gain, unsmoothed | smoothed |
|---|---|---|
| 16 | 28.2 | 14.0 |
| 8 | 92.9 | 46.1 |
| 4 | 7.7 | 56.3 |
| 2 | 5.9 | 0.0 |

The gain is roughly halved in the middle of that range. Duplicated *continuous* data is not the
trigger — it is already well conditioned — and at two distinct values the map collapses to a constant,
which is degenerate rather than a win.

**Exact inversion.** The reference CDF is inverted by walking both CDFs with a monotone pointer,
advancing while `cb[k+1] <= ca[j]`. Perturbing the CDF to make it strictly increasing and then
binary-searching moves it by a position-dependent amount, so across a flat run the search lands one
node early — a systematic per-slice error the sweep compounds. The pointer walk is exact when the two
CDFs are equal.

**Probability-band trimming.** A quantile map is well conditioned only where both distributions have
mass; in the tails one sample decides the answer, so a 10⁻⁷ difference becomes a table that jumps by
whole units. The sweep accumulates ~10⁻⁵ of drift per slice, so by slice ten the accumulator's tail no
longer aligns with the reference's. Both CDFs are trimmed to their central `1 − 2·10⁻⁴` and the tails
become linear ramps. Out-of-distribution content passes through instead of being crushed.

**Unit-slope extrapolation.** The grid is finite but the values applied to it need not be — limited-
range YUV converts to RGB outside `[0, 1]` by design. Clamping to the end node leaves a residual that
compounds across slices, reaching a 36× worst-case gain and a 5%-of-full-scale shift on an identity
4:2:0 model. Beyond the grid the transfer function therefore continues with slope 1, which preserves
monotonicity.

Inside the grid but outside the band, the ramp from identity to the fitted value is a straight line,
which leaves a slope discontinuity at the band edge — and a 3D LUT cannot resolve a kink, which is
where most of its error comes from. Matching the interior's slope with a monotone cubic fixes the LUT
but costs tail fidelity, since a curve starting at that slope deviates further from the identity
mid-tail. The straight line stays: leaving unfitted content alone is what the tails are for, and the
kink is a LUT limitation rather than something the native path should pay for.

### Directions

`fibonacci` (default) spreads `K` unit vectors evenly over the sphere and completes each into a
right-handed orthonormal frame. Deterministic, no optimiser needed.

`drdre` reproduces the original's `generateNonParallelAxes` repulsion objective. The original seeds
`fminunc` with `rand`, so it is not reproducible run to run; this is deterministic given `seed`, but
not bit-identical. `AxesScheme::Explicit` injects directions directly, for tests.

Directions needed, on a pooled four-scene cross-channel look:

| `K` | 1 | 2 | 4 | 8 | 10 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| sliced-W₁ removed | 87.5% | 88.7% | 98.7% | 98.9% | 99.2% | 99.5% | 99.6% | 99.6% |
| worst local gain | 1.89 | 1.88 | 1.93 | 2.04 | 1.98 | 1.82 | 1.17 | 1.85 |

`K=4` already reaches 98.7%. The default is 32 because that curve is a synthetic single global look
and real footage is less cooperative, and because native apply is linear in `K`.

## Subsampled chroma

The transform mixes R, G and B at one spatial location, so it needs co-sited full-resolution planes.
4:2:0 and 4:2:2 are upsampled to 4:4:4, transformed, and reduced back, so the output format matches
the input. A 3D transform is not separable, which is why every 3D-LUT tool does the same.

The reduction on the way back is what matters. Chroma sample `(i, j)` is co-sited with luma `(2i, 2j)`,
so luma coordinate `x` maps to chroma coordinate `x·s` with no half-pixel offset. Bilinear upsampling
evaluated at a grid knot returns the knot exactly, and a **nearest** reduction picks exactly those
co-sited positions, so the round trip is lossless and an identity model is a true no-op on 4:2:0. A
centre-aligned mapping or a box reduction loses that. `chroma="box"` is available for better
reconstruction without exactness.

## The identity property

An identity model must be an **exact** no-op, and in this codebase that is a test. Four defects were
found by asking whether a model built from a clip, against that same clip, returns the clip unchanged:

| symptom | cause |
|---|---|
| 4:4:4 fine, 4:2:0 and 4:2:2 off by 5–18% | chroma normalised with the luma range, not centred on zero |
| everything off by ~31% on limited-range YUV | a default hard clamp to `[0, 1]` discarding legal headroom |
| worst-case gain 36×, error growing with `K` | `Axis::eval` clamping values outside the slice-0 grid |
| a systematic per-slice error on atom-heavy data | `strictly_increasing` shifting the reference CDF by `i·eps` |

None would have shown up as an obviously wrong picture. That is why the test asserts exact equality
rather than a tolerance. The suite covers 4:2:0, 4:2:2 and 4:4:4, limited and full range, 8 and 10
bits, with the output format asserted unchanged.

## Pooling

The largest quality win is statistical, not algorithmic. Measured on a synthetic reel (eight scenes
with varying exposure, four held out, a cross-channel look):

| model | build-set mean sliced-W₁ | worst frame | held-out scenes |
|---|---|---|---|
| frame 0 only | 0.00264 | 0.00552 | 0.00265 |
| 8 pooled frames | 0.00102 | 0.00164 | 0.00114 |

The held-out error is within 12% of the build-set error, so there is no overfitting. Grouping is what
makes one model safe for a whole reel.

Budgets saturate early: 2 000 pixels give 0.00148, 20 000 give 0.00053, 200 000 give 0.00055; 64 bins
give 0.00061, 128 give 0.00057, 256 give 0.00056; float32 is indistinguishable from float64. Defaults
sit past the point of diminishing returns. `bins` cannot exceed 256 — the model format stores a fixed
257-node table, so a finer histogram is rejected rather than truncated.

## Applying

`native` evaluates the slice chain directly: exact, cost linear in `K`. The inner loop is tile-major
over planar `f32` buffers, so the image is read and written once and the tables stay in cache. On the
GPU the same chain runs as one WGSL compute shader — a transliteration of the CPU loop in the same
order, verified to within one 16-bit LSB — with cost nearly flat in `K`.

The table gathers dominate, not the arithmetic. The tables are uniform-grid (direct index, no binary
search) and small, because a search would add a branch to every sample to save memory that was never
the constraint.

`lut` bakes the chain into a 3D LUT sampled tetrahedrally, so cost stops depending on `K`. It is an
approximation: a 33³ bake is within 9.7 LSB mean and 654 LSB worst-case over the sample distribution,
40.9 / 1078 LSB over uniform colours. Refining to 65³ roughly halves that rather than quartering it —
the error is driven by slope, not grid spacing, since the transfer function has a kink at the band
edge that no grid resolves. See [performance.md](performance.md).

## Backends

Four: `scalar` (reference), `simd`, `gpu`, `lut`. Selection is automatic; the one that ran is written
to `_CM_Backend` on every output frame.

That reporting is a contract. `gpu_apply_available()` creates the device once and reports whether that
worked, so a build with the `gpu` feature on a machine with no usable adapter does not advertise a GPU
backend. `backend="gpu"` does not fall back on failure — a failure there is a fault, not a missing
capability. `simd` currently selects the same code as `scalar`; the label says which path ran, which
is true, but the speedup is not written yet.

## Exporting

`.cm` is the native, versioned, self-describing format: magic `CMAT`, format version, node count, slice
count, both colour spaces, build statistics, then per slice a row-major rotation and three
`(lo, scale, tab[257])` axes. A `K=32` model is about 100 KB, which is why it travels as a frame
property. Parsing is strict: a non-orthonormal rotation or a non-monotone table is rejected.

`.cube` and `.clut` are for interchange with other tools. Prefer the native path for delivery.