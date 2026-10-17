"""Wrapper tests. These do not need the compiled plugin except where marked."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import vs_colormatch as vscm


def test_version_is_semver():
    parts = vscm.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cache_dir_follows_the_vsjet_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("VSCOLORMATCH_CACHE_DIR", raising=False)
    monkeypatch.delenv("VSCOLORMATCH_GLOBAL", raising=False)
    local = vscm.get_local_cache()
    assert local.name == "vs_colormatch"
    assert local.parent.name == ".vsjet"

    explicit = vscm.get_cache_dir(tmp_path)
    assert explicit == Path(tmp_path).resolve()

    monkeypatch.setenv("VSCOLORMATCH_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert vscm.get_cache_dir() == (tmp_path / "elsewhere").resolve()


def test_backend_names_and_kwargs():
    assert vscm.Backend.CPU().name == "cpu"
    assert vscm.Backend.SIMD().name == "simd"
    assert vscm.Backend.GPU(1).device_id == 1
    assert vscm.Backend.LUT(65).to_kwargs() == {
        "backend": "lut",
        "device_id": 0,
        "method": "lut",
        "lut_size": 65,
    }
    assert vscm.Backend.CPU().to_kwargs()["method"] == "native"


def test_backend_accepts_the_plugin_s_aliases_and_normalises_case():
    """The wrapper and the plugin must agree on which names work.

    The plugin accepted `scalar` and `wgpu` while the wrapper rejected them, so a name that worked
    when passed to `cm.Apply` directly failed through `Backend(name=...)`.
    """
    assert vscm.Backend(name="scalar").name == "cpu"
    assert vscm.Backend(name="wgpu").name == "gpu"
    assert vscm.Backend(name="  GPU  ").name == "gpu"
    assert vscm.Backend(name="LUT").name == "lut"


def test_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="auto/cpu/simd/gpu/lut"):
        vscm.Backend(name="tensorrt")
    with pytest.raises(ValueError, match="device_id"):
        vscm.Backend.GPU(-1)


def test_backend_validates_the_lut_size_eagerly():
    """`Backend.LUT(32)` looked valid and then failed at filter-creation time."""
    assert vscm.Backend.LUT(17).lut_size == 17
    assert vscm.Backend.LUT(65).lut_size == 65
    with pytest.raises(ValueError, match="lut_size"):
        vscm.Backend.LUT(32)


def test_apply_rejects_a_model_that_is_neither_a_model_nor_a_path():
    """An int used to become the string "42" and be reported as an unreadable file."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.RGB24, length=2)
    for bad in (42, None, 3.5, object()):
        with pytest.raises(TypeError, match="must be a Model or a path"):
            vscm.apply(clip, bad)


def test_config_key_is_stable_and_sensitive():
    from vs_colormatch.settings import config_key

    a = config_key([32, 256, "fibonacci", 1000])
    b = config_key([32, 256, "fibonacci", 1000])
    c = config_key([32, 256, "fibonacci", 1001])
    assert a == b
    assert a != c
    assert len(a) == 16


def test_algorithm_tag_tracks_the_algorithm_not_the_format():
    """The cache key needs an algorithm tag, not the file-format version.

    A `.cm` written by an older build is still perfectly parseable — it just describes a different
    transform for the same arguments. 0.2.0 changed the gamut default, the grid extrapolation, the
    CDF inversion and added smoothing, so the tag has to move with the algorithm.

    The tag is deliberately *not* pinned to the package version: a patch release, or one that adds an
    execution path without changing the fit, produces the same models from the same arguments and must
    not invalidate every cache entry. 0.3.0 added the GPU backend and left the tag alone for that
    reason.
    """
    from vs_colormatch import settings
    from vs_colormatch.cmfile import FORMAT_VERSION

    assert settings.ALGORITHM_TAG
    assert settings.ALGORITHM_TAG != str(FORMAT_VERSION)

    # The tag names a version, and it must never be ahead of the code that ships it: a tag from the
    # future would mean every model was built by a different algorithm than the cache assumes.
    tagged = tuple(int(x) for x in settings.ALGORITHM_TAG.rsplit("-", 1)[-1].split("."))
    current = tuple(int(x) for x in vscm.__version__.split("."))
    assert tagged <= current, f"tag {tagged} is ahead of the package version {current}"


def test_trim_and_crop_validation():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=64, height=32, format=vs.RGB24, length=20)

    assert vscm.trim(clip, 5, 10).num_frames == 5
    assert vscm.trim(clip, 5).num_frames == 15
    assert vscm.trim(clip, 0, 20).num_frames == 20
    with pytest.raises(ValueError, match="greater than start"):
        vscm.trim(clip, 10, 10)
    with pytest.raises(ValueError, match="negative"):
        vscm.trim(clip, -1, 5)
    with pytest.raises(ValueError, match="past the end"):
        vscm.trim(clip, 100, 200)
    # An out-of-bounds end is an error, not a clamp: `check_ranges` and `crop` are both strict, and
    # silently clamping here meant the same mistake failed two different ways.
    with pytest.raises(ValueError, match="past the end"):
        vscm.trim(clip, 0, 99)

    assert vscm.crop(clip, x=(8, 40)).width == 32
    assert vscm.crop(clip, x=(8, 40)).height == 32
    assert vscm.crop(clip, y=(4, 20)).height == 16
    assert vscm.crop(clip).width == 64
    with pytest.raises(ValueError, match="not inside"):
        vscm.crop(clip, x=(0, 999))


def test_normalize_rejects_variable_format_and_non_clips():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.RGB24, length=2)

    with pytest.raises(TypeError, match="must be a vapoursynth clip"):
        vscm.normalize_inputs(["not a clip"])
    with pytest.raises(ValueError, match="at least one clip"):
        vscm.normalize_inputs([])
    normalized, changes = vscm.normalize_inputs([clip])
    assert normalized[0].matrix == 0
    assert normalized[0].range == 1
    assert changes == []


def test_normalize_leaves_subsampled_yuv_alone_by_default():
    """The plugin resamples chroma internally, so pre-converting is wasted work and changes the clip."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.YUV420P10, length=2)
    normalized, changes = vscm.normalize_inputs([clip], matrix=1, strict=True)
    assert normalized[0].clip.format.subsampling_w == 1
    assert normalized[0].clip.format.subsampling_h == 1
    assert changes == [], "nothing should have been changed"


def test_normalize_can_convert_subsampled_yuv_when_asked():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.YUV420P10, length=2)
    normalized, changes = vscm.normalize_inputs([clip], matrix=1, convert_subsampled=True)
    assert normalized[0].clip.format.subsampling_w == 0
    assert normalized[0].clip.format.subsampling_h == 0
    assert len(changes) == 1
    assert "4:4:4" in changes[0].what


def test_resolve_range_follows_the_vapoursynth_default():
    """A missing `_Range` is a documented default, not an ambiguity."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    yuv = core.std.BlankClip(width=16, height=16, format=vs.YUV444P10, length=2)
    rgb = core.std.BlankClip(width=16, height=16, format=vs.RGB24, length=2)
    normalized, _ = vscm.normalize_inputs([yuv], matrix=1, strict=True)
    assert normalized[0].range == 0, "YUV defaults to limited range"
    normalized, _ = vscm.normalize_inputs([rgb], strict=True)
    assert normalized[0].range == 1, "RGB defaults to full range"


def test_normalize_requires_a_matrix_for_yuv_under_strict():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.YUV444P10, length=2)
    with pytest.raises(ValueError, match="matrix"):
        vscm.normalize_inputs([clip], strict=True)
    normalized, _ = vscm.normalize_inputs([clip], matrix=1)
    assert normalized[0].matrix == 1


def test_normalize_rejects_unsupported_matrix():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.RGB24, length=2)
    with pytest.raises(ValueError, match="not supported"):
        vscm.normalize_inputs([clip], matrix=10)


def test_summarize_is_readable():
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    clip = core.std.BlankClip(width=16, height=16, format=vs.RGB24, length=2)
    normalized, changes = vscm.normalize_inputs([clip])
    text = vscm.summarize(changes, normalized)
    assert "1 input clip" in text
    assert "changed: nothing" in text


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_cache_key_covers_every_model_affecting_argument():
    """A cache hit must not hand back a model built under different settings."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    vscm.load_plugin()
    clip = core.std.BlankClip(width=32, height=32, format=vs.RGB48, length=4)

    with tempfile.TemporaryDirectory() as td:

        def key(**kwargs) -> str:
            return vscm.build(test=[clip], ref=[clip], cache=True, cache_dir=td, **kwargs).path.name

        base = key(directions=8)
        assert key(directions=8) == base, "identical arguments must reuse the model"
        for label, kwargs in (
            ("directions", {"directions": 16}),
            ("bins", {"bins": 64}),
            ("samples", {"samples": 4096}),
            ("seed", {"seed": 99}),
            ("weights", {"weights": [2, 1]}),
            ("cache_tag", {"cache_tag": "source-a"}),
        ):
            assert key(**kwargs) != base, f"{label} must change the key"


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_match_forwards_arguments_instead_of_dropping_them():
    """`match` used to accept `**kwargs` and pass none of them on."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    vscm.load_plugin()
    clip = core.std.BlankClip(width=64, height=64, format=vs.RGB48, length=4)

    # A build argument must reach `build`.
    out = vscm.match(clip, clip, directions=4, bins=64)
    assert out.num_frames == clip.num_frames

    # An apply argument must reach `apply`, and only through apply_kwargs.
    out = vscm.match(clip, clip, directions=4, apply_kwargs={"method": "lut", "lut_size": 17})
    assert out.get_frame(0).props["_CM_Method"].decode() == "lut:17"

    # A typo must be rejected rather than silently ignored.
    with pytest.raises(TypeError):
        vscm.match(clip, clip, nonsense=1)
    with pytest.raises(TypeError):
        vscm.match(clip, clip, apply_kwargs={"nonsense": 1})


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_plugin_backed_roundtrip():
    """A real build/apply round trip. Requires the compiled plugin."""
    vs = pytest.importorskip("vapoursynth")
    core = vs.core
    vscm.load_plugin()

    h, w, n = 32, 48, 6
    rng = np.random.default_rng(3)
    frames = [np.clip(0.3 + 0.28 * rng.standard_normal((h, w, 3)), 0, 1) for _ in range(n)]
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGB48, length=n)

    def _set(n, f):  # VapourSynth passes the frame index as the keyword `n`
        f = f.copy()
        for p in range(3):
            np.asarray(f[p])[:] = (frames[n][:, :, p] * 65535).astype(np.uint16)
        return f

    clip = core.std.ModifyFrame(blank, blank, _set)
    model = vscm.build(test=[clip], ref=[clip], directions=8)
    out = vscm.apply(clip, model)
    assert out.format.id == clip.format.id
    assert (out.width, out.height, out.num_frames) == (clip.width, clip.height, clip.num_frames)
    # Any of the known backends is fine here; what matters is that the label is one of them rather
    # than an invented name. The integration suite checks that the label matches the path that ran.
    label = out.get_frame(0).props["_CM_Backend"].decode()
    assert label.split(":")[0] in ("scalar", "simd", "gpu", "lut"), label

    # And requesting one explicitly must report that one.
    forced = vscm.apply(clip, model, backend="simd").get_frame(0).props["_CM_Backend"].decode()
    assert forced == "simd", forced


# --- mixed input families ---------------------------------------------------


def _core():
    """The VapourSynth core, skipping the test rather than failing collection when it is absent."""
    return pytest.importorskip("vapoursynth").core


def _rgb48(arrays):
    core, vs_mod = _core(), pytest.importorskip("vapoursynth")
    h, w, _ = arrays[0].shape
    blank = core.std.BlankClip(width=w, height=h, format=vs_mod.RGB48, length=len(arrays))

    def fill(n, f):  # VapourSynth passes the frame index as the keyword `n`
        f = f.copy()
        for p in range(3):
            np.asarray(f[p])[:] = (arrays[n][:, :, p] * 65535).astype(np.uint16)
        return f

    return core.std.ModifyFrame(blank, blank, fill)


@pytest.fixture(scope="module")
def rgb_pair():
    vs_mod = pytest.importorskip("vapoursynth")
    rng = np.random.default_rng(9)
    h, w, count = 32, 48, 4
    base = [np.clip(0.3 + 0.25 * rng.standard_normal((h, w, 3)), 0, 1) for _ in range(count)]
    look = np.array([[0.86, 0.10, 0.04], [0.06, 0.90, 0.04], [0.02, 0.12, 0.86]])
    graded = [np.clip(a.reshape(-1, 3) @ look.T, 0, 1).reshape(a.shape) for a in base]
    del vs_mod
    return _rgb48(base), _rgb48(graded)


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_each_input_is_converted_with_its_own_matrix(rgb_pair):
    """Every supported matrix must round-trip exactly on an identity model.

    The plugin used to take a single `matrix` for the whole call and the wrapper only passed one when
    all inputs agreed, so a YUV test clip against an RGB reference silently failed to resolve a
    matrix at all. It now takes one per input.
    """
    vscm.load_plugin()
    core, vs_mod = _core(), pytest.importorskip("vapoursynth")
    source, _ = rgb_pair
    for mtx in (1, 6, 9):
        yuv = core.resize.Bicubic(
            source, format=vs_mod.YUV444P10, matrix_in=0, matrix=mtx, range_in=1, range=0
        )
        model = vscm.build(test=[yuv], ref=[yuv], directions=16)
        out = vscm.apply(yuv, model)
        for p in range(3):
            a = np.asarray(yuv.get_frame(0)[p]).astype(int)
            b = np.asarray(out.get_frame(0)[p]).astype(int)
            assert np.abs(a - b).max() == 0, f"matrix {mtx} plane {p}: not an exact no-op"


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_mixed_families_match_the_single_family_result(rgb_pair):
    """A YUV test clip against an RGB reference must give the same model as RGB against RGB.

    Both describe the same look in the same working space, so any difference means a matrix or range
    was applied to the wrong side.
    """
    vscm.load_plugin()
    core, vs_mod = _core(), pytest.importorskip("vapoursynth")
    source, reference = rgb_pair
    yuv = core.resize.Bicubic(
        source, format=vs_mod.YUV444P10, matrix_in=0, matrix=9, range_in=1, range=0
    )
    mixed = vscm.build(test=[yuv], ref=[reference], directions=32)
    pure = vscm.build(test=[source], ref=[reference], directions=32)

    # Not identical: the YUV clip is a 10-bit quantisation of the RGB source, so its distribution
    # differs slightly. Measured at 1.002x, so 2% is a tight bound that still tolerates the
    # quantisation.
    assert mixed.stats.w1_after == pytest.approx(pure.stats.w1_after, rel=0.02), (
        f"mixed {mixed.stats.w1_after} vs pure {pure.stats.w1_after}"
    )

    # The real failure mode, and the one that matters: if the YUV side is converted with the wrong
    # matrix the model is visibly worse. Forcing `matrix=1` on a bt2020nc clip measures 1.19x the
    # residual, so the comparison below is what the fix is actually protecting.
    wrong = vscm.build(test=[yuv], ref=[reference], directions=32, matrix=1)
    assert wrong.stats.w1_after > mixed.stats.w1_after * 1.10, (
        f"a wrong matrix should be clearly worse: wrong {wrong.stats.w1_after} vs "
        f"mixed {mixed.stats.w1_after}"
    )

    # And the model built from mixed input is usable on either side.
    assert vscm.apply(source, mixed).format.id == source.format.id
    assert vscm.apply(yuv, mixed).format.id == yuv.format.id


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_inputs_may_differ_in_range_as_well_as_matrix(rgb_pair):
    """Two YUV clips of different ranges must each be interpreted with their own range."""
    vscm.load_plugin()
    core, vs_mod = _core(), pytest.importorskip("vapoursynth")
    source, _ = rgb_pair
    limited = core.resize.Bicubic(
        source, format=vs_mod.YUV444P10, matrix_in=0, matrix=1, range_in=1, range=0
    )
    full = core.resize.Bicubic(
        source, format=vs_mod.YUV444P10, matrix_in=0, matrix=1, range_in=1, range=1
    )
    model = vscm.build(test=[limited], ref=[full], directions=8)
    assert model.stats.directions == 8
    # The two differ only in range, so the model is not the identity but should be mild.
    assert model.stats.max_gain < 5.0, model.stats.max_gain


@pytest.mark.skipif(
    not os.environ.get("VSCOLORMATCH_PLUGIN"),
    reason="set VSCOLORMATCH_PLUGIN to run the plugin-backed tests",
)
def test_compat_selects_the_builder_and_is_part_of_the_cache_key(rgb_pair):
    """`compat` changes the fit, so it has to reach the plugin and the cache key."""
    vscm.load_plugin()
    source, reference = rgb_pair
    v2 = vscm.build(test=[source], ref=[reference], directions=8)
    drdre = vscm.build(test=[source], ref=[reference], directions=8, compat="drdre")
    assert v2.stats.w1_after != pytest.approx(drdre.stats.w1_after, rel=1e-9), (
        "the two builders produced the same model"
    )

    with tempfile.TemporaryDirectory() as td:

        def key(**kwargs):
            return vscm.build(
                test=[source], ref=[reference], cache=True, cache_dir=td, **kwargs
            ).path.name

        assert key(directions=8) != key(directions=8, compat="drdre")

    with pytest.raises(ValueError, match="v2.*drdre"):
        vscm.build(test=[source], ref=[reference], compat="v13")
