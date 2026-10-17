# Cost model

What scales with what, and where the time goes. No timings: absolute figures belong to the host, not to
this software.

## What the work is

Each slice is three table gathers and a 3×3 matrix product; a frame costs `K` of them per pixel.

| quantity | effect |
|---|---|
| slice count `K` | linear on both CPU paths, nearly flat on the GPU and LUT paths |
| frame resolution | linear, every backend |
| bit depth | none (everything is processed as `f32`) |
| chroma subsampling | a second pass over the frame for 4:2:0 and 4:2:2 |
| model size | ~100 KB at `K=32`, so it fits in a frame property |

## Why gathers dominate

The 3×3 product is three multiply-adds per channel; a gather is a dependent memory access whose address
comes from the data. Hence uniform-grid tables (direct index, no search) and 257 nodes per axis, stored
as an inline fixed-size array so the access needs no runtime bounds check.

This is also what bounds the CPU SIMD path. Eight pixels go through the rotations at once, but each
lane still needs its own three table lookups per slice, so roughly half the work stays scalar and the
speedup plateaus there. The GPU is flat in `K` for the same reason in reverse: a gather is cheap
there.

## Why the GPU helps, and where it stops

A gather that costs a CPU a cache miss is a shared-memory load on a GPU, so the frame's memory traffic
rather than `K` sets the pace. On the CPU the same work is linear in `K`, so the gap widens with `K` —
which is why `auto` switches to the GPU only above a threshold.

Measured against a passthrough of the same clip, the shape is:

| configuration | cost over passthrough |
|---|---|
| scalar, `K=1` | small |
| scalar, `K=32` | about 30x the `K=1` figure |
| SIMD, `K=32` | roughly a third of scalar |
| GPU, `K=32` | roughly twice the `K=1` figure |
| LUT 33³ | nearly flat in `K`, and about the same as the GPU |

The CPU's fixed cost — conversion, allocation, frame I/O — is close to nothing: extrapolating the
scalar slope back to `K=1` accounts for essentially the whole measurement, so the slice chain *is* the
cost. On the GPU the balance inverts: the chain is so cheap there that the fixed cost is roughly a
third of the total at `K=32`, and that fixed cost is what a GPU port cannot remove from a
upload-and-download design.

## LUT: a different trade, not a faster one

A bake replaces `K` gathers with one tetrahedral interpolation, so cost stops depending on `K`. The LUT
is a resampling and cannot represent the transform exactly.

The error is dominated by the slope discontinuity where the fitted interior meets the identity tail
ramp, not by grid spacing — so refining the grid helps slowly. Smoothing the ramp fixes the LUT but was
rejected for the tail-fidelity cost; see [algorithm.md](algorithm.md).

The LUT path also clamps its input to the unit cube. For limited-range YUV, where legal RGB extends
beyond `[0, 1]`, that is a real limitation, which is why `native` is the default.

## Remaining cost, in priority order

1. **Fuse the format conversion into the transform, on the CPU path.** The GPU side is done: native
   samples upload directly and the shader does normalisation, the YUV matrix and chroma resampling,
   which removed roughly a fifth of the GPU path's cost at `K=32`. On the CPU the same change buys
   very little, because the conversion is already close to free — the slice chain hides it.
2. **Vectorise the gathers.** They are roughly half the native path's cost and do not vectorise,
   because neither NEON nor the portable abstraction offers a gather. Restructuring so a lane group
   shares one table walk — sorting lanes by index, or using a small per-slice histogram — is the only
   way past this, and it is a rewrite of the inner loop rather than a flag. This is the largest
   remaining win on the CPU path, where the slice chain is the whole cost.
3. **Fewer slices without losing quality.** Directions are chosen without regard to the data, so
   selecting ones that reduce the objective could reach the same quality at lower `K`, which is the CPU
   path's linear factor.
4. **Drop the download half of the GPU round trip.** VapourSynth 4.3's Vulkan interop (`ffGPUOutput`)
   needs a Vulkan-enabled VapourSynth and driver on the target, which is not guaranteed.

## Reproducing the algorithmic figures

```bash
cargo run --release -p colormatch-core --example measure
```

Prints the sliced-W₁ distances, worst-case gains, LUT errors and identity-floor residuals quoted in
the README and [algorithm.md](algorithm.md). Those are properties of the algorithm and reproduce
anywhere.

If you profile your own material: `-t` in `vspipe` is `--timecodes FILE`, **not** `--threads`. It
writes a timecode file and leaves concurrency at its default, without failing. Use `-r`/`--requests`.