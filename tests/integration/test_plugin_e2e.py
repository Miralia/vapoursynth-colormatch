"""End-to-end tests for the `cm` VapourSynth plugin.

Run with:  python3 tests/integration/test_plugin_e2e.py
The plugin path can be overridden with CM_PLUGIN.
"""
import os
import sys
import tempfile
import numpy as np
import vapoursynth as vs

PLUGIN = os.environ.get(
    "CM_PLUGIN",
    os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "libcolormatch.dylib"),
)

core = vs.core
core.std.LoadPlugin(path=os.path.abspath(PLUGIN))

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def clip_from_arrays(arrays, fmt):
    """Builds a clip from a list of (h, w, 3) arrays."""
    h, w, _ = arrays[0].shape
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=len(arrays))
    src = blank
    arrs = arrays

    def _set(n, f):
        f = f.copy()
        for p in range(3):
            np.asarray(f[p])[:] = arrs[n][:, :, p]
        return f

    return core.std.ModifyFrame(blank, src, _set)


def make_scene(seed, h=64, w=96, scene=0.5):
    rng = np.random.default_rng(seed)
    base = 0.30 + 0.28 * rng.standard_normal((h, w, 3))
    base = np.clip(base * (0.75 + 0.5 * scene), 0.0, 1.0)
    return base


def grade(x):
    """A cross-channel look: mixing plus per-channel gamma. Not a per-channel LUT."""
    m = np.array([[0.86, 0.10, 0.04], [0.06, 0.90, 0.04], [0.02, 0.12, 0.86]])
    y = np.clip(x.reshape(-1, 3) @ m.T + np.array([0.01, -0.01, 0.02]), 0.0, 1.0)
    y = y ** np.array([1.15, 1.00, 0.92])
    return y.reshape(x.shape)


def to_u16(a):
    return np.clip(a * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def read_clip(clip, n):
    f = clip.get_frame(n)
    return np.stack([np.asarray(f[p]).astype(np.float64) / 65535.0 for p in range(3)], axis=-1)


def read_plane(clip, n, p):
    """One plane, normalised by the clip's own peak. Planes need not share a shape (4:2:0)."""
    peak = float((1 << clip.format.bits_per_sample) - 1)
    return np.asarray(clip.get_frame(n)[p]).astype(np.float64) / peak


def frame_max_diff(a, b):
    return float(np.abs(a - b).max())


print("== 1. Devices ==")
dev = core.cm.Devices()
df = dev.get_frame(0)
check("Devices format is Gray8", df.format.name == "Gray8", df.format.name)
check("Devices size is 1x1", (df.width, df.height) == (1, 1))
check("Devices reports a version", str(df.props["_CM_Version"]) != "")
check("Devices reports backends", "scalar" in df.props["_CM_Backends"].decode())
print(f"       adapters={df.props['_CM_AdapterCount']} backends={df.props['_CM_Backends']} gpu={df.props['_CM_GpuEnabled']}")

print("== 2. BuildModel + Apply on RGB48 ==")
test_arrs = [make_scene(100 + i, scene=i / 8) for i in range(8)]
ref_arrs = [grade(a) for a in test_arrs]
test_clip = clip_from_arrays([to_u16(a) for a in test_arrs], vs.RGB48)
ref_clip = clip_from_arrays([to_u16(a) for a in ref_arrs], vs.RGB48)

model = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=32, samples=262144)
mf = model.get_frame(0)
stats = list(mf.props["_CM_Stats"])
print(f"       model bytes={mf.props['_CM_ModelBytes']} samples={int(stats[0])} frames={int(stats[1])}")
print(f"       w1 {stats[2]:.6f} -> {stats[3]:.6f}  max_gain={stats[4]:.2f}  build={stats[5]:.1f}ms  dirs={int(stats[6])}")
check("model clip is 1x1 Gray8 length 1",
      (mf.format.name, mf.width, mf.height, model.num_frames) == ("Gray8", 1, 1, 1))
check("model blob is present", mf.props["_CM_ModelBytes"] > 100_000)
check("w1 improved by >=95%", stats[3] < stats[2] * 0.05, f"{stats[2]} -> {stats[3]}")

out = core.cm.Apply(clip=test_clip, model=model)
check("Apply preserves format", out.format.id == test_clip.format.id)
check("Apply preserves size", (out.width, out.height) == (test_clip.width, test_clip.height))
check("Apply preserves length", out.num_frames == test_clip.num_frames)
of = out.get_frame(0)
check("Apply reports the backend", of.props.get("_CM_Backend", b"").decode() != "")
check("Apply reports the method", of.props.get("_CM_Method", b"").decode() == "native",
      of.props.get("_CM_Method"))
print(f"       backend={of.props['_CM_Backend']} method={of.props['_CM_Method']}")

before, after = [], []
for n in range(len(test_arrs)):
    src = read_clip(test_clip, n)
    dst = read_clip(out, n)
    tgt = read_clip(ref_clip, n)
    before.append(np.abs(src - tgt).mean())
    after.append(np.abs(dst - tgt).mean())
print(f"       mean |test-ref| {np.mean(before):.5f} -> {np.mean(after):.5f}")
check("Apply moves the clip toward the reference", np.mean(after) < np.mean(before) * 0.35)

print("== 3. Identity reference leaves the clip alone ==")
ident = core.cm.BuildModel(test=[test_clip], ref=[test_clip], directions=16)
ident_out = core.cm.Apply(clip=test_clip, model=ident)
worst = 0.0
for n in range(3):
    worst = max(worst, frame_max_diff(read_clip(ident_out, n), read_clip(test_clip, n)))
print(f"       worst per-channel move = {worst * 65535:.1f} / 65535")
check("identity model is a near-noop (<8 LSB at 16-bit)", worst * 65535 < 8.0)

print("== 4. model_file round-trip ==")
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "look.cm")
    cached = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=8, cache=path)
    cached.get_frame(0)  # the filter is lazy: the build happens on first frame request
    check("cache file written", os.path.exists(path) and os.path.getsize(path) > 10_000,
          f"{os.path.getsize(path) if os.path.exists(path) else 'missing'}")
    from_file = core.cm.Apply(clip=test_clip, model_file=path)
    check("Apply from file matches Apply from clip",
          frame_max_diff(read_clip(from_file, 0), read_clip(out, 0)) < 0.5)

    cube = os.path.join(td, "look.cube")
    core.cm.ExportLUT(clip=test_clip, model=model, path=cube, size=33)
    lines = open(cube).read().splitlines()
    header = [l for l in lines if l.startswith("LUT_3D_SIZE")]
    check("cube header", header == ["LUT_3D_SIZE 33"], str(header))
    rows = [l for l in lines if l and not l[0].isalpha()]
    check("cube row count", len(rows) == 33 ** 3, str(len(rows)))

print("== 5. Error paths ==")
def expect_error(name, fn, needle):
    try:
        fn()
        check(name, False, "no error raised")
    except Exception as e:
        msg = str(e)
        check(name, needle.lower() in msg.lower(), f"message was: {msg[:160]}")

yuv420 = core.std.BlankClip(width=64, height=64, format=vs.YUV420P10, length=4)
yuv444 = core.std.BlankClip(width=64, height=64, format=vs.YUV444P10, length=4)
# Subsampled input is supported natively; section 7 checks that it round-trips exactly. What is
# still refused is a YUV clip with no resolvable matrix.
expect_error("YUV420 without matrix rejected by Apply",
             lambda: core.cm.Apply(clip=yuv420, model=model), "matrix")
expect_error("YUV420 without matrix rejected by BuildModel",
             lambda: core.cm.BuildModel(test=[yuv420], ref=[yuv420]), "matrix")
check("YUV420 accepted with an explicit matrix",
      core.cm.Apply(clip=yuv420, model=model, matrix=1, range=0) is not None)
# The plugin refuses to guess a matrix for YUV; the wrapper reads `_Matrix` and passes it.
expect_error("YUV444 without matrix rejected by BuildModel",
             lambda: core.cm.BuildModel(test=[yuv444], ref=[yuv444]), "matrix")
try:
    core.cm.BuildModel(test=[yuv444], ref=[yuv444], matrix=1).get_frame(0)
    check("YUV444 with explicit matrix accepted", True)
except Exception as e:
    check("YUV444 with explicit matrix accepted", False, str(e)[:160])
expect_error("missing model rejected",
             lambda: core.cm.Apply(clip=test_clip), "model")
expect_error("both model and model_file rejected",
             lambda: core.cm.Apply(clip=test_clip, model=model, model_file="x.cm"), "not both")
expect_error("bad backend rejected",
             lambda: core.cm.Apply(clip=test_clip, model=model, backend="tensorrt"), "backend")
expect_error("bad lut_size rejected",
             lambda: core.cm.Apply(clip=test_clip, model=model, lut_size=32), "lut_size")
expect_error("empty test array rejected",
             lambda: core.cm.BuildModel(test=[], ref=[ref_clip]), "test")
expect_error("bad compat value rejected",
             lambda: core.cm.BuildModel(test=[test_clip], ref=[ref_clip], compat="v13"), "compat")

print("== 6. LUT method ==")
lut_out = core.cm.Apply(clip=test_clip, model=model, method="lut", lut_size=33)
lf = lut_out.get_frame(0)
check("lut method reported", lf.props["_CM_Method"].decode() == "lut:33", lf.props["_CM_Method"])
for size in (17, 33, 65):
    lo = core.cm.Apply(clip=test_clip, model=model, method="lut", lut_size=size)
    a = read_clip(lut_out if size == 33 else lo, 0)
    diffs = np.concatenate([np.abs(read_clip(out, 0)[p] - a[p]).ravel() for p in range(3)])
    print(f"       {size}^3: max={diffs.max() * 65535:7.1f}  mean={diffs.mean() * 65535:6.3f}  "
          f"p99.9={np.percentile(diffs, 99.9) * 65535:7.1f}  (16-bit LSB)")
# A 3D LUT is a resampling of the transform, so its error scales with the model's local slope times
# the grid spacing -- it is not a rounding artefact. A high-gain model (this one has gain > 2) is
# therefore the worst case for the LUT path, and 2% of full scale at 33^3 is the honest bound.
# The threshold catches a broken LUT path without pretending the path is lossless.
d = frame_max_diff(read_clip(lut_out, 0), read_clip(out, 0))
check("lut33 within 3% of full scale of native", d < 0.03, f"{d * 100:.2f}%")
d17 = frame_max_diff(read_clip(core.cm.Apply(clip=test_clip, model=model, method="lut", lut_size=17), 0), read_clip(out, 0))
d65 = frame_max_diff(read_clip(core.cm.Apply(clip=test_clip, model=model, method="lut", lut_size=65), 0), read_clip(out, 0))
print(f"       max error by size: 17^3={d17*100:.2f}%  33^3={d*100:.2f}%  65^3={d65*100:.2f}%")
check("a finer LUT is not worse", d65 <= d * 1.05 + 1e-6, f"65^3 {d65*100:.2f}% vs 33^3 {d*100:.2f}%")

print("== 7. Subsampled YUV ==")
# The transform needs co-sited full-resolution planes, so 4:2:0/4:2:2 is upsampled, transformed and
# downsampled back. The contract is that the output format is unchanged, and that an identity model
# is a true no-op: nearest-downsample on bilinearly-upsampled chroma recovers the original samples
# exactly, because interpolation at a grid knot returns the knot.
rgb_ref = clip_from_arrays([make_scene(700 + i) for i in range(4)], vs.RGB48)


def to_yuv(fmt, rng_):
    return core.resize.Bicubic(
        rgb_ref, format=fmt, matrix_in=0, matrix=1, range_in=1, range=rng_
    )


for fmt, label in ((vs.YUV420P8, "420p8"), (vs.YUV420P10, "420p10"),
                   (vs.YUV422P10, "422p10"), (vs.YUV444P10, "444p10")):
    for rng_, rlabel in ((0, "limited"), (1, "full")):
        src = to_yuv(fmt, rng_)
        # The plugin never reads `_Matrix` from props: only the wrapper does. Calling the plugin
        # directly means passing it explicitly.
        m = core.cm.BuildModel(test=[src], ref=[src], directions=32, matrix=1, range=rng_)
        o = core.cm.Apply(clip=src, model=m, matrix=1, range=rng_)
        check(f"{label}/{rlabel}: format preserved", o.format.id == src.format.id,
              f"{o.format.name} vs {src.format.name}")
        check(f"{label}/{rlabel}: dimensions and length preserved",
              (o.width, o.height, o.num_frames) == (src.width, src.height, src.num_frames))
        worst = 0
        for p in range(src.format.num_planes):
            worst = max(worst, float(np.abs(read_plane(src, 0, p) - read_plane(o, 0, p)).max()))
        peak = (1 << src.format.bits_per_sample) - 1
        # Exact, not "close": anything else means the resampling round trip or the range maths is off.
        check(f"{label}/{rlabel}: identity model is an exact no-op", worst == 0,
              f"max diff {worst * peak:.0f}/{peak}")

y420 = to_yuv(vs.YUV420P10, 0)
model420 = core.cm.BuildModel(test=[y420], ref=[y420], directions=8, matrix=1, range=0)
box = core.cm.Apply(clip=y420, model=model420, chroma="box", matrix=1, range=0)
check("chroma=box accepted and preserves format", box.format.id == y420.format.id)
expect_error("bad chroma mode rejected",
             lambda: core.cm.Apply(clip=y420, model=model420, chroma="bicubic", matrix=1, range=0),
             "chroma")

print("== 8. Plugin metadata ==")
# The plugin entry point is hand-written so that the version VapourSynth reports is derived from the
# crate version. The `export_vapoursynth_plugin!` macro hard-codes `1`, which made `core.plugins()`
# report 0.1 for every release.
meta = next(p for p in core.plugins() if p.namespace == "cm")
check("namespace", meta.namespace == "cm", meta.namespace)
check("identifier", meta.identifier == "dev.miralia.colormatch", meta.identifier)
actual = (meta.version.major, meta.version.minor)
# Compare against the wrapper's version, which is generated from the Cargo manifest.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "src"))
import vs_colormatch  # noqa: E402

want = tuple(int(x) for x in vs_colormatch.__version__.split(".")[:2])
check(f"plugin version matches the manifest ({want[0]}.{want[1]})", actual == want, str(actual))
check("all four filters registered",
      all(hasattr(core.cm, f) for f in ("BuildModel", "Apply", "Devices", "ExportLUT")))

print("== 9. Serialized model statistics ==")
# The plugin only *writes* the cache; the cache-hit policy belongs to the wrapper. What the plugin
# must guarantee is that a model reloaded from a file reports the same provenance as the build that
# wrote it, which is why the statistics live in the .cm header.
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.cm")
    built = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=12, cache=path)
    bf = built.get_frame(0)
    check("cache file written", os.path.exists(path))
    hit = core.cm.BuildModel(model_file=path)
    hf = hit.get_frame(0)
    check("reloaded model has the same byte length",
          hf.props["_CM_ModelBytes"] == bf.props["_CM_ModelBytes"],
          f"{bf.props['_CM_ModelBytes']} vs {hf.props['_CM_ModelBytes']}")
    # `_CM_Stats` is one flat f64 array: samples, frames, w1_before, w1_after, max_gain,
    # build_ms, directions.
    names = ("samples", "frames", "w1_before", "w1_after", "max_gain", "build_ms", "directions")
    bstats = list(bf.props["_CM_Stats"])
    hstats = list(hf.props["_CM_Stats"])
    check("stats array has the documented length", len(bstats) == len(names), str(len(bstats)))
    for i, name in enumerate(names):
        # Exact: the header stores the same f32 values the build reports, so a cache hit must be
        # indistinguishable from the build that wrote it.
        check(f"cache hit preserves {name}", abs(hstats[i] - bstats[i]) < 1e-9,
              f"{bstats[i]} vs {hstats[i]}")
    # The build actually produced something; otherwise the comparison above is vacuous.
    check("the built model reports a real sample count", bstats[0] > 0, str(bstats[0]))
    check("the built model reports a real direction count", bstats[6] == 12, str(bstats[6]))

print("== 10. DrDre compatibility builder ==")
# `compat="drdre"` reproduces the original's structure so the two builders can be compared. It has to
# build a usable model: monotone tables, a real reduction, and an apply that moves the clip toward the
# reference like any other model.
drdre = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=16, compat="drdre")
df = drdre.get_frame(0)
check("drdre compat builds", df.props["_CM_ModelBytes"] > 0, str(df.props["_CM_ModelBytes"]))
dstats = list(df.props["_CM_Stats"])
check("drdre model reduces the distance", dstats[3] < dstats[2] * 0.5,
      f"{dstats[2]:.5f} -> {dstats[3]:.5f}")
# The gain is sample-size dependent for this builder -- its percentile knots come from the data, so a
# small cloud gives a steeper fit. Reported rather than asserted against an arbitrary bound; what
# matters is that it is finite, positive and not absurd.
check("drdre model has a finite positive gain", 1.0 <= dstats[4] < 1e4, f"{dstats[4]:.2f}")
vstats = list(core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=16).get_frame(0).props["_CM_Stats"])
print(f"       gain: drdre {dstats[4]:.2f} vs v2 {vstats[4]:.2f} | "
      f"w1_after: drdre {dstats[3]:.5f} vs v2 {vstats[3]:.5f}")

# Tables must be monotone. The original has to repair this with a deletion pass; ours has to end up
# monotone either way, and a non-monotone table is a visible artefact rather than a subtle one.
import struct as _struct  # noqa: E402

raw = bytes(df.props["_CM_Model"])
nodes = _struct.unpack_from("<I", raw, 8)[0]
n_slices = _struct.unpack_from("<I", raw, 12)[0]
header = 64
stride = 9 * 4 + 3 * (2 * 4 + nodes * 4)
non_monotone = 0
for k in range(n_slices):
    base = header + k * stride + 9 * 4
    for axis in range(3):
        a = base + axis * (8 + nodes * 4)
        lo, scale = _struct.unpack_from("<ff", raw, a)
        tab = _struct.unpack_from(f"<{nodes}f", raw, a + 8)
        for i in range(1, nodes):
            if tab[i] < tab[i - 1] - 1e-4:
                non_monotone += 1
check("drdre tables are monotone", non_monotone == 0, f"{non_monotone} decreasing step(s)")

dout = core.cm.Apply(clip=test_clip, model=drdre)
d = frame_max_diff(read_clip(dout, 0), read_clip(ref_clip, 0))
g = frame_max_diff(read_clip(test_clip, 0), read_clip(ref_clip, 0))
print(f"       drdre apply: test->ref {g * 65535:.0f} LSB becomes {d * 65535:.0f} LSB")
check("drdre apply moves the clip toward the reference", d < g * 0.6, f"{g * 65535:.0f} -> {d * 65535:.0f}")

# And it must be a different model from the default, or the comparison is meaningless.
default = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=16)
check("drdre and v2 produce different models",
      bytes(default.get_frame(0).props["_CM_Model"]) != raw)

print("== 11. GPU backend ==")
# The GPU path has to agree with the CPU one. A backend that reports itself as having run while
# producing different pixels is worse than having no GPU backend, so this compares them on real
# frames across every supported layout rather than trusting the shader transliteration.
gpu_flag = core.cm.Devices().get_frame(0).props["_CM_GpuApply"]
if not gpu_flag:
    print("       no usable GPU adapter here; skipping the comparison")
else:
    cpu_model = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=16)
    for label, src, mtx, rng_ in (
        ("RGB48", test_clip, 0, 1),
        ("YUV444P10", core.resize.Bicubic(test_clip, format=vs.YUV444P10, matrix_in=0, matrix=1,
                                          range_in=1, range=0), 1, 0),
        ("YUV420P10", core.resize.Bicubic(test_clip, format=vs.YUV420P10, matrix_in=0, matrix=1,
                                          range_in=1, range=0), 1, 0),
    ):
        kw = {} if mtx == 0 else {"matrix": mtx, "range": rng_}
        model = cpu_model if mtx == 0 else core.cm.BuildModel(test=[src], ref=[src], directions=16, **kw)
        cpu = core.cm.Apply(clip=src, model=model, backend="simd", **kw)
        gpu = core.cm.Apply(clip=src, model=model, backend="gpu", **kw)
        check(f"{label}: gpu backend reported",
              gpu.get_frame(0).props["_CM_Backend"].decode().startswith("gpu:"),
              gpu.get_frame(0).props["_CM_Backend"].decode())
        check(f"{label}: cpu backend reported as simd",
              cpu.get_frame(0).props["_CM_Backend"].decode() == "simd",
              cpu.get_frame(0).props["_CM_Backend"].decode())
        worst = 0
        for p in range(src.format.num_planes):
            worst = max(worst, float(np.abs(read_plane(cpu, 0, p) - read_plane(gpu, 0, p)).max()))
        peak = (1 << src.format.bits_per_sample) - 1
        # 2 LSB, not 0: the two paths differ by f32 reassociation and FMA contraction, and the final
        # conversion rounds to the nearest integer, so a sub-LSB difference in f32 can land on either
        # side of a rounding boundary. That is a rounding difference, not a disagreement about the
        # transform -- a genuinely wrong transform would be off by orders of magnitude more.
        check(f"{label}: gpu agrees with cpu within 2 LSB", worst * peak <= 2.0,
              f"max diff {worst * peak:.2f}/{peak}")

    # The identity property must hold on the GPU too, or the backend is not interchangeable.
    ident = core.cm.BuildModel(test=[test_clip], ref=[test_clip], directions=16)
    gi = core.cm.Apply(clip=test_clip, model=ident, backend="gpu")
    worst = max(float(np.abs(read_plane(test_clip, 0, p) - read_plane(gi, 0, p)).max())
                for p in range(3))
    check("GPU: an identity model is an exact no-op", worst == 0, f"{worst * 65535:.0f}/65535")

print("== 12. SIMD backend ==")
# The wide path must be bit-identical to the scalar one, not merely close. It exists to be faster,
# and any difference at all would mean a user's grade depends on which CPU backend `auto` happened
# to pick. That is a stronger requirement than the GPU's, and it is met because each lane evaluates
# its own table lookup with the same scalar code and the rotations keep the scalar association.
for label, src, mtx, rng_ in (
    ("RGB48", test_clip, 0, 1),
    # No matrix arguments on the RGB resize: passing a YUV matrix for an RGB destination is an error,
    # and the plugin needs none for RGB input.
    ("RGB24", core.resize.Bicubic(test_clip, format=vs.RGB24, range_in=1, range=1), 0, 1),
    ("YUV444P10", core.resize.Bicubic(test_clip, format=vs.YUV444P10, matrix_in=0, matrix=1,
                                      range_in=1, range=0), 1, 0),
    ("YUV420P10", core.resize.Bicubic(test_clip, format=vs.YUV420P10, matrix_in=0, matrix=1,
                                      range_in=1, range=0), 1, 0),
    ("YUV420P8", core.resize.Bicubic(test_clip, format=vs.YUV420P8, matrix_in=0, matrix=1,
                                     range_in=1, range=0), 1, 0),
    ("YUV444P16", core.resize.Bicubic(test_clip, format=vs.YUV444P16, matrix_in=0, matrix=1,
                                      range_in=1, range=1), 1, 1),
):
    kw = {} if mtx == 0 else {"matrix": mtx, "range": rng_}
    model = (core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=32) if mtx == 0
             else core.cm.BuildModel(test=[src], ref=[src], directions=32, **kw))
    cpu = core.cm.Apply(clip=src, model=model, backend="cpu", **kw)
    simd = core.cm.Apply(clip=src, model=model, backend="simd", **kw)
    check(f"{label}: simd backend reported",
          simd.get_frame(0).props["_CM_Backend"].decode() == "simd",
          simd.get_frame(0).props["_CM_Backend"].decode())
    check(f"{label}: cpu backend reported as scalar",
          cpu.get_frame(0).props["_CM_Backend"].decode() == "scalar",
          cpu.get_frame(0).props["_CM_Backend"].decode())
    worst = 0.0
    for p in range(src.format.num_planes):
        worst = max(worst, float(np.abs(read_plane(cpu, 0, p) - read_plane(simd, 0, p)).max()))
    peak = (1 << src.format.bits_per_sample) - 1
    check(f"{label}: simd is bit-identical to cpu", worst == 0.0,
          f"max diff {worst * peak:.2f}/{peak}")

# Every gamut and luma-lock combination, and a length that is not a multiple of the vector width, so
# the scalar tail is covered in the integration path too and not only in the unit tests.
for gamut in ("none", "clip", "soft"):
    for lock in (0.0, 1.0):
        gm = core.cm.BuildModel(test=[test_clip], ref=[ref_clip], directions=8)
        a = core.cm.Apply(clip=test_clip, model=gm, backend="cpu", gamut=gamut, luma_lock=lock)
        b = core.cm.Apply(clip=test_clip, model=gm, backend="simd", gamut=gamut, luma_lock=lock)
        d = max(float(np.abs(read_plane(a, 0, p) - read_plane(b, 0, p)).max()) for p in range(3))
        check(f"gamut={gamut} luma_lock={lock}: simd matches cpu exactly", d == 0.0, f"{d * 65535:.0f}")

# Non-multiple-of-8 widths exercise the tail on both paths.
for w in (7, 9, 63, 65):
    odd = core.std.CropAbs(test_clip, width=w, height=16)
    m = core.cm.BuildModel(test=[odd], ref=[odd], directions=8)
    a = core.cm.Apply(clip=odd, model=m, backend="cpu")
    b = core.cm.Apply(clip=odd, model=m, backend="simd")
    d = max(float(np.abs(read_plane(a, 0, p) - read_plane(b, 0, p)).max()) for p in range(3))
    check(f"width {w} (scalar tail): simd matches cpu exactly", d == 0.0, f"{d * 65535:.0f}")

# And the identity property, which is the test that has found four real defects here.
ident = core.cm.BuildModel(test=[test_clip], ref=[test_clip], directions=16)
si = core.cm.Apply(clip=test_clip, model=ident, backend="simd")
worst = max(float(np.abs(read_plane(test_clip, 0, p) - read_plane(si, 0, p)).max())
            for p in range(3))
check("SIMD: an identity model is an exact no-op", worst == 0, f"{worst * 65535:.0f}/65535")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("all integration checks passed")
