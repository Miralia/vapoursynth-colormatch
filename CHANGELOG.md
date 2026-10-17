# Changelog

**Nothing has been published.** No release artifacts, no distribution — the entries are undated source
markers, not release records.

[Keep a Changelog](https://keepachangelog.com/en/1.1.0) format,
[semver](https://semver.org/spec/v2.0.0.html) with the pre-1.0 reading: minor for capability, patch for
fixes, no bare 1.0 while the feature set is incomplete. `Cargo.toml` is authoritative;
`python/gen_version.py` derives the Python version and `tests/python/test_version.py` fails on drift.

## [1.0.0] — unreleased

The advertised feature set is complete: every backend named in the documentation is implemented and
interchangeable, so the version no longer claims less than the software does. Two features were
outstanding when the pre-1.0 gate was written — the GPU compute path, and `Backend::Simd` selecting
the scalar code under a different name. Both are done.

### Added

- **The GPU path converts formats in the shader.** `apply_gpu_native` uploads the clip's native
  samples — u8 four per word, u16 two per word, f32 bitcast — and does normalisation, the YUV matrix,
  bilinear chroma upsampling and the nearest reduction on the device. No `f32` planes are built on the
  host, so the per-frame conversion and its allocations are gone. Measured against the planar route it
  is about 1.9x faster at `K=32`.

  It is bit-exact against the CPU path, not merely close: every supported layout compares at 0. The
  path is used whenever the backend is GPU and the layout is one the shader implements; `chroma="box"`
  and partial plane selection still take the planar route.

- Edge-case coverage for the whole wrapper surface: degenerate geometries (1×1, 1×N, N×1, prime
  widths), numeric bounds on `directions`/`bins`/`samples`/`weights`, single-frame clips through the
  scene detector, partial plane selection, scene ranges with gaps and with overlaps, malformed model
  files, non-finite input, and backend naming. Rejections are asserted to *name the offending
  argument*, not merely to fail.

### Fixed

- **`ApplyOptions::default()` returned `Gamut::Clip`** while `Gamut`'s own `#[default]`, the plugin's
  default and the documentation all said `Gamut::None`. A library caller taking the struct default got
  a hard clamp the plugin would never apply, and for limited-range YUV that clamp discards legal
  headroom. Found because an identity-model test failed by ~0.19 of full scale.

- **The shader's chroma subsampling factors were computed from shift counts.** `ClipFormat` stores
  `sub_w`/`sub_h` as shifts (the value `chroma_dims` shifts by), and they were passed to the shader as
  if they were factors — so for 4:2:0 the shader saw a factor of 1, treated every pixel as co-sited,
  and wrote the chroma planes out of bounds. The symptom was a constant on the chroma planes while
  luma stayed exact.

### Changed

- `test_stays_pre_1_0` is replaced by `test_major_version_is_earned`, which states what 1.0 has to
  mean rather than forbidding it. The gate did its job: it kept the version honest until the claim was
  true.

## [0.6.0] — unreleased

### Added

- **A real CPU SIMD backend.** `Backend::Simd` previously selected the same code as `Scalar`; it now
  runs eight pixels per iteration through the portable `wide` abstraction (NEON on aarch64, AVX/SSE on
  x86_64). The crate denies `unsafe_code`, so intrinsics were not an option and a portable SIMD crate
  is what makes this possible at all.

  The output is **bit-identical** to the scalar path, not merely close. Each lane evaluates its own
  table lookup with the same scalar code, and the vector rotations reproduce the scalar expression's
  association exactly — `r + (d0*m00 + d1*m10 + d2*m20)` rather than `r + d0*m00 + d1*m10 + d2*m20`,
  which folds `r` in first and rounds in a different place. That distinction cost one ULP and was
  caught by a bit-equality test rather than by a tolerance. Bit equality matters here because
  otherwise a user's grade would depend on which CPU backend `auto` happened to choose.

  In-process the gain runs from 1.7x at `K=1` to 3.7x at `K=32`, and is independent of frame size — it
  is a property of the loop, not of the cache. It plateaus because the table gathers, about half the
  work, do not vectorise: neither NEON nor the portable abstraction offers a gather instruction. Every
  figure is the best of seven runs on one machine and should be read as a shape rather than a
  guarantee; the ordering (wide ahead at every `K`) is what held across repeats.

### Changed

- `docs/performance.md` no longer lists explicit SIMD as outstanding, and states the gather ceiling
  that bounds the CPU path instead.

## [0.5.3] — unreleased

### Fixed

- The test suite wrote `.cm` cache files into the source tree: `match()` caches by default and the local
  cache resolves relative to the script directory, which under pytest is the test module's. A session
  fixture points it at a temporary directory.

### Changed

- Documentation rewritten for brevity. `docs/performance.md` is a cost model — what scales with what —
  rather than a table of durations.

## [0.5.2] — unreleased

### Fixed

- `device_id` was validated then ignored: `Backend.GPU(2)` chose adapter 0. It now indexes the same
  ranked adapter list `cm.Devices` returns, and an index with no adapter behind it falls back to SIMD
  and says so in `_CM_Backend`.
- `trim` clamped an out-of-bounds `end` while erroring on an out-of-bounds `start`. Both are errors now.
- The wrapper rejected backend names the plugin accepts (`scalar`, `wgpu`, and any casing or
  whitespace).
- `Backend.LUT(32)` failed at filter-creation time; sizes are now checked at construction.
- `apply(clip, 42)` stringified the argument and reported an unreadable model file. It is a `TypeError`.

## [0.5.1] — unreleased

### Fixed

- A build mixing input families failed or used the wrong colour space. The plugin took one
  `matrix`/`range` for the whole call and the wrapper passed them only when every input agreed, so a
  YUV test clip against an RGB reference failed outright, and two YUV clips of different ranges were
  both treated as limited. `cm.BuildModel` now takes `matrices` and `ranges` arrays, one entry per
  input; `-1` means unspecified.
- `compat` was not reachable from the wrapper, and was missing from the cache key.

## [0.5.0] — unreleased

### Added

- `compat="drdre"`: the original's table fit — percentile-paired knots at 0.1% steps, monotonicity
  repaired by deleting decreasing pairs, `±√3` sentinels, a monotone cubic through the knots. Two
  details are approximated rather than reproduced: MATLAB's `smoothingspline` at `SmoothingParam` 0.99
  is replaced by PCHIP, and `interp1(..., 'cubic')` differs from PCHIP in its end conditions.

### Changed

- Documentation no longer claims the default builder is more accurate. Measured at `K=32`, the original
  bakes into a LUT better (653 vs 1078 LSB) while the default holds a smaller worst-case gain and a
  smaller self-match residual. The two are comparable; the default is the more robust.
- An example did not compile in a default build — it was gated with an inner `#![cfg]`, so without the
  `gpu` feature it had no `main`.

## [0.4.0] — unreleased

### Added

- Per-scene and per-range models: `build_groups`, `apply_groups`, `match_groups`, `detect_scenes`,
  `SceneGroup`. The plugin is untouched — this is graph assembly around `cm.BuildModel`/`cm.Apply`.
- Frames not covered by any range pass through unchanged. Each group gets its own cache entry, suffixed
  even when the caller supplies `cache_tag`, because the key does not cover which clips were used.
- `detect_scenes` wraps `vstools.sc_detect`; a single-frame clip is returned directly, since the
  detector cannot shift a clip with no neighbour.

## [0.3.0] — unreleased

### Added

- GPU compute backend: a WGSL shader and a `wgpu` pipeline, one shader compiling to Metal, Vulkan and
  DX12. A transliteration of the CPU loop in the same order, agreeing within one 16-bit LSB.
- `gpu_apply_available()` is a real device check, so a build with the feature on and no usable adapter
  does not advertise a GPU backend.

### Changed

- `backend="auto"` selects GPU when available and `K >= 8`.
- The `gpu` feature is on by default for the plugin; the core crate still builds without `wgpu`.

## [0.2.1] — unreleased

### Fixed

- The plugin reported version 0.1 to VapourSynth for every release, because the export macro hard-codes
  `1` to `configPlugin`. The entry point is now hand-written and derives the version from the crate.

### Rejected

- Matching the interior's slope at the band edge with a monotone cubic fixes the LUT bake but costs
  tail fidelity. Reverted, with the reasoning recorded at the call site.

## [0.2.0] — unreleased

Subsampled YUV support, and five correctness fixes that came out of testing it. Each had the same
signature — an identity model failing to be a no-op.

### Added

- 4:2:0 and 4:2:2 in `cm.Apply` and `cm.BuildModel`: chroma is upsampled, transformed and reduced back,
  so the output format is identical to the input's. Previously subsampled input could not be applied.
- `cm.BuildModel(model_file=...)` reloads a serialized model without a clip. `test`/`ref` are optional
  and mutually exclusive with it.
- `chroma` on `cm.Apply` and the wrapper's `apply`/`match`; `convert_subsampled` on
  `normalize_inputs`/`build`/`match`, off by default.
- `vs_colormatch.cmfile`, a strict `.cm` header reader, so cached models report real provenance.
- Python version generated from the Cargo workspace version, with drift tests.

### Fixed

- `bins` above 256 crashed and any `bins` other than 256 could panic: the right-tail ramp iterated all
  257 entries instead of stopping at the node count. `bins` was accepted up to 4096.
- Chroma was normalised with the luma range and not centred on zero, so every YUV→RGB conversion was
  wrong — neutral chroma normalised to 0.5 instead of 0.0.
- `Axis::eval` clamped instead of extrapolating outside its grid, compounding across slices into a 36×
  worst-case gain and a 5%-of-full-scale shift on an identity 4:2:0 model.
- The CDF inversion landed one node early wherever the CDF was flat: `strictly_increasing` added `i*eps`
  and the search over that perturbed array returns the wrong node. Replaced with a two-pointer walk.
- The default gamut policy was a hard clamp to `[0, 1]`, discarding the headroom limited-range video
  reserves for undershoot and overshoot. Default is now `gamut="none"`.
- `apply_lut_tetrahedral` indexed past the end of the LUT for an input of exactly 1.0 where
  `denom >= 64`, because `64.0f32 - 1e-6 == 64.0f32` and the clamp was a no-op.
- A stepped CDF inverted without constraint on atom-heavy input. Two passes of a `[1,2,1]/4` kernel on
  the splatted histograms bound the slope (worst-case gain roughly halved on coarsely quantised
  input).
- A frame-evaluation failure was reported as a missing `_Matrix` property; a missing `_Range` was an
  error under `strict` rather than falling back to VapourSynth's documented default.
- Cached models reported fabricated zero statistics; the `.cm` header carried them all along.
- A filter with no inputs never receives `arAllFramesReady`, so `ModelOnlyFilter` must produce its frame
  in `get_frame_initial`.

### Fixed — packaging

The wheel had never been built, and building it found four defects that would all have shipped:

- It could not be built at all: the version came from `versioningit`, which needs a git tag. It now
  comes from `_version.py`, generated from `Cargo.toml`.
- `py3-none-any` on a wheel with a native library — pip would have installed a macOS dylib on Linux. The
  tag is derived from the Rust target, so it survives cross-compilation, and `Root-Is-Purelib` is false.
- A duplicate plugin at `vapoursynth/wheel/vapoursynth/plugins/...`, from a build hook with one too many
  `parent` calls.
- `readme` and `license-files` pointed outside the project directory, which hatchling refuses.

`just test-wheel` builds, installs into a throwaway venv and asserts the tag, layout and that `core.cm`
runs.

### Fixed — caching

- The key omitted `seed`, `weights` and any algorithm identity, so a model built by one version could be
  reused under another's different semantics. There is now an `ALGORITHM_TAG`.
- `match` accepted `**kwargs` and forwarded none: `vscm.match(clip, ref, bins=64)` ignored `bins`. Build
  arguments are forwarded (a typo is a `TypeError`) and apply arguments go through `apply_kwargs`.
- `load_plugin` finds the plugin the wheel shipped next to the wrapper.

### Changed

- Documentation corrected against re-measurement. Three claims were wrong: splatting was described as a
  decisive accuracy win over hard binning (they measure the same to three significant figures); the
  direction sweep was said to need `K=32` (the current builder removes 98.7% at `K=4`); and the 33³ LUT
  error was quoted as 3.2 LSB (it is ~654 LSB worst-case, dominated by the band-edge slope
  discontinuity).
- `examples/measure.rs` re-measures every documented figure, so invalidating one shows up as a diff.
- `normalize_inputs` no longer converts subsampled YUV to 4:4:4 by default, since the plugin resamples
  chroma internally and the pre-conversion cost an extra pass.

## [0.1.0] — unreleased

First version.

- `cm.BuildModel`, `cm.Apply`, `cm.Devices`, `cm.ExportLUT`.
- Pooled multi-frame sampling with per-clip weights and deterministic frame selection.
- Linear-splatted histograms, probability-band trimming, uniform-grid tables, per-slice refitting.
- CPU scalar and SIMD backends; GPU adapter enumeration with automatic selection.
- Versioned `.cm` model format, `.cube` and `.clut` export, caching in a vs-jetpack compatible `.vsjet`
  tree.
- CLI for inspecting models and exporting LUTs without VapourSynth.
- Python wrapper: `match`, `build`, `apply`, `trim`, `crop`, `Model`, backend policy, normalisation
  reporting.