"""Edge cases across the wrapper's surface.

Everything here is a boundary the plugin enforces or a shape that has caught a real bug. The two
groups are kept distinct on purpose: `ok` is for calls that must succeed, `rejects` for calls that
must fail *with a message naming the offending argument*. A rejection with an unhelpful message is a
failure, because the caller has to be able to tell which argument to change.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import vs_colormatch as vscm


def _core():
    return pytest.importorskip("vapoursynth").core


def _clip(w=64, h=48, fmt_id=None, n=4, seed=0):
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    fmt_id = vs.RGB48 if fmt_id is None else fmt_id
    blank = core.std.BlankClip(width=w, height=h, format=fmt_id, length=n)

    def fill(n, f):  # VapourSynth passes the frame index as the keyword `n`
        f = f.copy()
        rng = np.random.default_rng(seed + n)
        for p in range(f.format.num_planes):
            arr = np.asarray(f[p])
            if f.format.sample_type == vs.FLOAT:
                arr[:] = rng.random(arr.shape).astype(np.float32)
            else:
                peak = (1 << f.format.bits_per_sample) - 1
                arr[:] = (rng.random(arr.shape) * peak).astype(arr.dtype)
        return f

    return core.std.ModifyFrame(blank, blank, fill)


def rejects(call, *needles):
    """Asserts the call raises, and that the message mentions each needle."""
    with pytest.raises(Exception) as exc:  # noqa: PT011 - the plugin raises several types
        call()
    text = str(exc.value)
    for needle in needles:
        assert needle in text, f"message does not mention {needle!r}: {text[:200]}"


# --- shapes ------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
@pytest.mark.parametrize(
    "w,h",
    [(1, 1), (1, 64), (64, 1), (3, 3), (7, 7), (8, 8), (9, 9), (1920, 1), (1, 1080), (1921, 1080)],
)
def test_extreme_geometry_round_trips(w, h):
    """Degenerate and non-multiple-of-eight widths.

    The odd widths matter more than they look: they are the only thing that exercises the scalar tail
    of the SIMD loop, and 1921 is prime so nothing divides it.
    """
    vscm.load_plugin()
    src = _clip(w, h)
    model = vscm.build(test=[src], ref=[src], directions=8, samples=2048)
    assert vscm.apply(src, model).num_frames == 4


# --- numeric bounds ----------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_direction_and_bin_bounds():
    vscm.load_plugin()
    src = _clip()
    assert vscm.build(test=[src], ref=[src], directions=1).stats.directions == 1
    assert vscm.build(test=[src], ref=[src], directions=256).stats.directions == 256
    rejects(lambda: vscm.build(test=[src], ref=[src], directions=0), "directions")
    rejects(lambda: vscm.build(test=[src], ref=[src], directions=257), "directions")
    # `bins` is capped by the model format's fixed 257-node table, so 16 and 256 are legal and the
    # rest are rejected rather than silently truncated.
    for bins in (16, 256):
        assert vscm.build(test=[src], ref=[src], bins=bins).stats.directions == 32
    rejects(lambda: vscm.build(test=[src], ref=[src], bins=15), "bins")
    rejects(lambda: vscm.build(test=[src], ref=[src], bins=257), "bins")


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_sample_budget_bounds():
    vscm.load_plugin()
    src = _clip()
    assert vscm.build(test=[src], ref=[src], samples=1024).stats.directions == 32
    rejects(lambda: vscm.build(test=[src], ref=[src], samples=1023), "samples")
    rejects(lambda: vscm.build(test=[src], ref=[src], samples=0), "samples")


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_weights_are_validated():
    vscm.load_plugin()
    a, b = _clip(seed=1), _clip(seed=2)
    for w in ([1, 1], [3, 1], [0.5, 0.5], [1e-3, 1.0]):
        assert vscm.build(test=[a], ref=[b], weights=w).stats.directions == 32
    rejects(lambda: vscm.build(test=[a], ref=[b], weights=[-1, 1]), "weights")
    rejects(lambda: vscm.build(test=[a], ref=[b], weights=[float("nan"), 1]), "weights")
    rejects(lambda: vscm.build(test=[a], ref=[b], weights=[1]), "weights")


# --- one frame ---------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_a_single_frame_clip_works_everywhere():
    """A one-frame clip is a scene detector's degenerate input, not a normal one.

    `vstools.sc_detect` shifts the clip by a frame, which fails with a bare "Trim: invalid last
    frame" from inside the dependency, so this path returns the whole clip as one scene instead.
    """
    vscm.load_plugin()
    one = _clip(n=1)
    assert vscm.build(test=[one], ref=[one], directions=8).stats.frames == 1
    assert vscm.apply(one, vscm.build(test=[one], ref=[one], directions=8)).num_frames == 1
    assert vscm.detect_scenes(one) == [(0, 1)]
    assert vscm.match_groups(one, one).num_frames == 1


# --- plane selection ---------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_partial_plane_selection():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    vscm.load_plugin()
    yuv = core.resize.Bicubic(
        _clip(), format=vs.YUV420P10, matrix_in=0, matrix=1, range_in=1, range=0
    )
    model = vscm.build(test=[yuv], ref=[yuv], directions=8)
    for planes in ([0], [1], [2], [0, 1], [1, 2], [0, 1, 2]):
        assert vscm.apply(yuv, model, planes=planes).num_frames == 4
    rejects(lambda: vscm.apply(yuv, model, planes=[3]), "planes")
    rejects(lambda: vscm.apply(yuv, model, planes=[-1]), "planes")


# --- scene ranges ------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_scene_ranges_are_validated_and_gaps_pass_through():
    vscm.load_plugin()
    src = _clip(n=10)
    assert vscm.match_groups(src, src, ranges=[(0, 5), (5, 10)]).num_frames == 10
    # A gap is allowed: the uncovered frames pass through unchanged rather than being given a
    # neighbouring model.
    assert vscm.match_groups(src, src, ranges=[(0, 3), (7, 10)]).num_frames == 10
    rejects(lambda: vscm.match_groups(src, src, ranges=[(0, 6), (4, 10)]), "ranges")
    rejects(lambda: vscm.match_groups(src, src, ranges=[(0, 99)]), "ranges")
    rejects(lambda: vscm.match_groups(src, src, ranges=[(5, 2)]), "ranges")
    rejects(lambda: vscm.check_ranges([], 10), "ranges")


# --- model files -------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_model_file_round_trip_and_rejections():
    vscm.load_plugin()
    src = _clip()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.cm")
        model = vscm.build(test=[src], ref=_clip(seed=3), directions=8, cache=path)
        assert vscm.load(path).stats.directions == 8
        assert vscm.apply(src, path).num_frames == 4

        for fmt, size in (("cube", 33), ("clut", 17), ("cm", 17)):
            out = os.path.join(td, f"l.{fmt}")
            assert os.path.getsize(model.export_lut(out, size=size, format=fmt)) > 1000

        rejects(lambda: model.export_lut(os.path.join(td, "x.cube"), size=32), "lut_size")
        rejects(
            lambda: model.export_lut(os.path.join(td, "x.3dl"), size=17, format="3dl"), "format"
        )

        # A malformed file must say what is wrong with it, not merely fail to load.
        empty = Path(td) / "empty.cm"
        empty.write_bytes(b"")
        rejects(lambda: vscm.load(empty), "empty.cm")
        junk = Path(td) / "junk.cm"
        junk.write_bytes(b"\x00" * 200)
        rejects(lambda: vscm.load(junk), "junk.cm")
        truncated = Path(td) / "trunc.cm"
        truncated.write_bytes(Path(path).read_bytes()[:50])
        rejects(lambda: vscm.load(truncated), "trunc.cm")
        rejects(lambda: vscm.load(Path(td) / "missing.cm"), "missing.cm")
        rejects(lambda: vscm.load(td), "no model file")


# --- non-finite input --------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_non_finite_input_never_becomes_non_finite_output():
    """A NaN in a float clip must not spread across the frame.

    Non-finite samples are mapped to zero before evaluation rather than allowed to propagate, so a
    single bad pixel stays a single bad pixel instead of poisoning the output.
    """
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    vscm.load_plugin()
    blank = core.std.BlankClip(width=32, height=32, format=vs.RGBS, length=2)

    def fill(n, f):
        f = f.copy()
        np.asarray(f[0])[:] = np.nan
        np.asarray(f[1])[:] = np.inf
        np.asarray(f[2])[:] = -np.inf
        return f

    poisoned = core.std.ModifyFrame(blank, blank, fill)
    model = vscm.build(test=[_clip(fmt_id=vs.RGBS)], ref=_clip(fmt_id=vs.RGBS, seed=4), directions=8)
    out = np.asarray(vscm.apply(poisoned, model).get_frame(0)[0])
    assert np.isfinite(out).all()


# --- backend names -----------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_backend_names_the_plugin_accepts_are_the_ones_the_wrapper_accepts():
    vscm.load_plugin()
    src = _clip()
    for name in ("auto", "cpu", "scalar", "simd", "gpu", "wgpu", "lut", "LUT", " gpu "):
        out = vscm.match(src, src, directions=8, backend=name)
        assert out.get_frame(0).props["_CM_Backend"].decode()
    rejects(lambda: vscm.match(src, src, backend="nope"), "backend")
