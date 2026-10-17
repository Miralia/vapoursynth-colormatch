# vapoursynth-colormatch

Rotated-marginal sliced-optimal-transport colour matching for VapourSynth.

Builds **one** global per-pixel colour transform from many frames of a reference, then applies it to a
whole reel. Cannot bleed or flicker, and exports as a LUT.

```python
import vs_colormatch as vscm

out = vscm.match(clip, ref)
```

Requires `vapoursynth>=74`. `vsjetpack` is optional.

## Install

```bash
pip install vs-colormatch
```

The wheel ships both the wrapper and the compiled plugin. The plugin is placed at
`vapoursynth/plugins/colormatch/` in site-packages, so `core.cm` exists once installed — provided
`vs-colormatch` and `vapoursynth` share a site-packages. Call `vscm.load_plugin()` to cover the case
where they do not; it is a no-op otherwise.

```python
import vs_colormatch as vscm
vscm.load_plugin()
```

## Usage

### One line

```python
out = vscm.match(clip, ref)
```

`clip` and `ref` need not match in length, resolution or format. The reference is read only for its
colour distribution. Subsampled input keeps its format.

### Build once, reuse

```python
model = vscm.build(
    test=[clip.std.Trim(100, 199), clip.std.Trim(300, 399)],
    ref =[ref.std.Trim(1100, 1199), ref.std.Trim(1300, 1399)],
    directions=32,
    cache=True,
)
out = vscm.apply(clip, model)
```

Both sides are pooled all-to-all: every `test` clip feeds the source distribution, every `ref` clip the
target. `vscm.trim(clip, 100, 200)` and `vscm.crop(clip, x=(300, 400))` are conveniences.

### Per-scene and per-range models

```python
# Let vsjetpack's scene detector find the segments.
out = vscm.match_groups(clip, ref, threshold=0.1)

# Or give the ranges.
out = vscm.match_groups(clip, ref, ranges=[(0, 1240), (1240, 2890), (2890, 4102)])

# Or specify which clips build each model.
models = vscm.build_groups(
    test=[[clip_a, clip_b], [clip_c]],
    ref =[[ref_a, ref_b], [ref_c]],
    directions=32,
    group_kwargs=[{}, {"directions": 64}],
)
out = vscm.apply_groups(clip, models, [(0, 1240), (1240, 4102)])
```

Frames not covered by any range pass through unchanged. Output keeps the input's width, height, format
and length. `verbose=True` reports the range-to-model mapping and the frame count left alone.

A single `ref` clip is cut per range, so it must be at least as long as `clip` and frame-aligned with
it. A list of references keeps the two sides independent.

Scene detection needs `vsjetpack`; otherwise pass `ranges=`.

### Backends

```python
from vs_colormatch import Backend

vscm.match(clip, ref, backend="auto")            # default
vscm.match(clip, ref, backend=Backend.autoselect())
vscm.match(clip, ref, backend=Backend.LUT(33))   # baked 3D LUT, cost independent of K
```

The backend and method that actually ran are written to `_CM_Backend` / `_CM_Method` on every output
frame, so a fallback is never silent. `backend="gpu"` reports a failure rather than falling back.

### Export

```python
model.save("look.cm")                   # reusable model
model.export_lut("look.cube", size=33)  # for Resolve, Premiere, other tools
model = vscm.load("look.cm")            # reload without a clip
```

A `.cm` model can also be loaded inside a graph: `core.cm.BuildModel(model_file="look.cm")`.

### Cache

Keys cover the build arguments, clip geometry and an algorithm tag; they do not look at pixel content.
Two different clips of the same resolution, format and length therefore share a key — pass
`cache_tag=` with something content-identifying when that can happen.

Model files land in `<script dir>/.vsjet/vs_colormatch/`. Override with `cache_dir=`,
`$VSCOLORMATCH_CACHE_DIR`, or `[tool.vs_colormatch] cache_dir = "..."` in `pyproject.toml` /
`vsjet.toml`.

## Comparison with DrDre ColorMatch

| | [DrDre ColorMatch](https://originaltrilogy.com/topic/Color-matching-and-prediction-color-correction-tool-v13-released/id/18128/) (2015, MATLAB) | this |
|---|---|---|
| Sampling | one image pair | many frames, many clips, pooled |
| Chroma | 4:4:4 only | 4:2:0 / 4:2:2, output format preserved |
| Tables | quantile knots + smoothing spline | uniform grid from splatted CDFs |
| Application | MATLAB, or a 33³ `.cube` | native float in VapourSynth |
| Backends | — | CPU scalar / CPU SIMD / GPU (wgpu) / LUT, auto-selected |
| Reference | same size as the test | any length, resolution or format |
| Gamut | clamps to `[0, 1]` | no clamp by default |

`compat="drdre"` builds with the original's method for comparison. At `K=32`:

| | sliced-W₁ after | worst gain | 33³ bake error | self-match residual |
|---|---|---|---|---|
| default | 0.00015 (99.6%) | 1.17 | 1078 LSB | 3.9e-7 |
| `compat="drdre"` | 0.00016 (99.5%) | 2.33 | 653 LSB | 3.1e-6 |

The default is the more robust of the two, not the more accurate.

`cargo run --release -p colormatch-core --example measure` reproduces every figure in these documents.

## Status

**1.0.0 — not published.** Working and tested:

* `cm.BuildModel`, `cm.Apply`, `cm.Devices`, `cm.ExportLUT`
* CPU scalar, CPU SIMD (eight pixels per iteration, bit-identical to scalar), GPU compute
  (wgpu + WGSL) and LUT backends, with the chosen one reported per frame
* 4:2:0 / 4:2:2 / 4:4:4, limited and full range, 8–16-bit integer and 32-bit float
* `.cm` model format (versioned), `.cube` / `.clut` export, caching
* output is bit-identical to the input in size, format and length, and an identity model is an exact
  no-op on every supported format
* the GPU path uploads the clip's **native** samples and converts in the shader, so no `f32` planes are
  built on the host; it agrees with the CPU path exactly on every layout it implements
* the CPU scalar and SIMD backends are bit-identical to each other, so the backend `auto` picks cannot
  change a single output pixel
* 88 Rust tests, 76 wrapper tests, an integration suite driven through VapourSynth

Known gaps:

1. `match_groups` without explicit `ranges` needs `vsjetpack` for `sc_detect`.
2. The table gathers are about half the native path's cost and do not vectorise, so the CPU backends
   have a ceiling below the GPU's — see [docs/performance.md](docs/performance.md).

## Building from source

```bash
cargo build --release -p colormatch-vs     # target/release/libcolormatch.{dylib,so} or colormatch.dll
cargo test                                  # Rust unit + integration tests
just test-integration                       # VapourSynth integration suite
just test-python                            # wrapper tests
```

Requires Rust 1.88+ (edition 2024), a C toolchain, and VapourSynth for the integration tests. `just`
and `zig` are needed only for cross-platform packaging.

### VapourSynth trap

`std.ModifyFrame` passes the frame index to its callback as the keyword argument `n`, so the first
parameter must be named exactly `n`:

```python
def fill(n, frame):     # not idx, not count
    frame = frame.copy()
    np.asarray(frame[0])[:] = ...
    return frame

clip = core.std.ModifyFrame(blank, blank, fill)
```

The graph builds fine either way; the error appears only when a frame is requested, attributed to
whatever filter asked for it.

## Layout

```
crates/colormatch-core/   pure algorithm; no VapourSynth dependency
crates/colormatch-vs/     the VapourSynth plugin (4 filters)
crates/colormatch-cli/    model inspection and LUT export without VapourSynth
python/src/vs_colormatch/ the wrapper: ranges, normalisation, caching, backend policy
tests/                    Rust integration tests + the VapourSynth integration suite
docs/                     algorithm notes and cost model
```

## Credits

A reimplementation of **Menno Dreischor's ColorMatch** (2015). See [NOTICE](NOTICE).

## License

MIT. See [LICENSE](LICENSE).