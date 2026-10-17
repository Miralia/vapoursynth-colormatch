"""Per-scene and per-range models.

These are mostly about the edges: a feature that splices clips back together has to preserve the
length exactly, and the failure modes that matter are off-by-one and silently-wrong-model, not
crashes.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import vs_colormatch as vscm

vs = pytest.importorskip("vapoursynth")
core = vs.core


@pytest.fixture(scope="module")
def two_scene_clip():
    """A 16-frame clip that is one scene for 8 frames and another afterwards."""
    h, w, count = 32, 48, 16
    rng = np.random.default_rng(3)
    base_a = np.clip(0.20 + 0.15 * rng.standard_normal((h, w, 3)), 0, 1)
    base_b = np.clip(0.70 + 0.15 * rng.standard_normal((h, w, 3)), 0, 1)
    frames = [
        np.clip((base_a if i < 8 else base_b) + 0.01 * rng.standard_normal((h, w, 3)), 0, 1)
        for i in range(count)
    ]
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGB48, length=count)

    def make(n, f):  # VapourSynth passes the frame index as the keyword `n`
        f = f.copy()
        for p in range(3):
            np.asarray(f[p])[:] = (frames[n][:, :, p] * 65535).astype(np.uint16)
        return f

    return core.std.ModifyFrame(blank, blank, make)


# --- range validation -------------------------------------------------------


def test_check_ranges_accepts_sorted_disjoint_ranges():
    assert vscm.check_ranges([(0, 5), (5, 10)], 10) == [(0, 5), (5, 10)]
    # A gap is allowed; it means "leave these frames alone".
    assert vscm.check_ranges([(0, 3), (7, 10)], 10) == [(0, 3), (7, 10)]


def test_check_ranges_drops_empty_ranges_but_keeps_the_rest():
    """A scene detector can legitimately emit an empty range; it should not be fatal."""
    assert vscm.check_ranges([(0, 0), (0, 5), (5, 5)], 5) == [(0, 5)]


def test_check_ranges_rejects_everything_malformed():
    with pytest.raises(ValueError, match="at least one range"):
        vscm.check_ranges([], 10)
    with pytest.raises(ValueError, match=r"must be a \(start, end\) pair"):
        vscm.check_ranges([(0, 1, 2)], 10)
    with pytest.raises(ValueError, match="must be a"):
        vscm.check_ranges([5], 10)
    with pytest.raises(ValueError, match="negative"):
        vscm.check_ranges([(-1, 5)], 10)
    with pytest.raises(ValueError, match="before its start"):
        vscm.check_ranges([(5, 3)], 10)
    with pytest.raises(ValueError, match="past the end"):
        vscm.check_ranges([(0, 11)], 10)
    with pytest.raises(ValueError, match="overlaps or precedes"):
        vscm.check_ranges([(0, 6), (5, 10)], 10)
    with pytest.raises(ValueError, match="overlaps or precedes"):
        vscm.check_ranges([(5, 10), (0, 6)], 10)
    with pytest.raises(ValueError, match="every range is empty"):
        vscm.check_ranges([(0, 0)], 10)


def test_check_ranges_handles_a_zero_length_clip():
    with pytest.raises(ValueError, match="every range is empty|past the end"):
        vscm.check_ranges([(0, 1)], 0)


# --- build_groups -----------------------------------------------------------


def test_build_groups_validates_its_arguments(two_scene_clip):
    with pytest.raises(ValueError, match="one entry per group"):
        vscm.build_groups(test=[two_scene_clip], ref=[])
    with pytest.raises(ValueError, match="at least one group"):
        vscm.build_groups(test=[], ref=[])
    with pytest.raises(ValueError, match="group_kwargs"):
        vscm.build_groups(
            test=[two_scene_clip], ref=[two_scene_clip], group_kwargs=[{}, {}]
        )


def test_build_groups_gives_each_group_its_own_cache_entry(two_scene_clip):
    """Two groups with identical geometry and settings must not share a model.

    The cache key covers the arguments and the clip geometry but not *which* clips, so without a
    per-group suffix the second group silently receives the first group's model. That is a
    wrong-result bug, not a slow-path bug.
    """
    with tempfile.TemporaryDirectory() as td:
        models = vscm.build_groups(
            test=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
            ref=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
            directions=8,
            cache=True,
            cache_dir=td,
        )
        paths = {m.path.name for m in models}
        assert len(paths) == 2, paths

        # A caller-supplied tag must not defeat the separation either.
        tagged = vscm.build_groups(
            test=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
            ref=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
            directions=8,
            cache=True,
            cache_dir=td,
            cache_tag="mine",
        )
        assert len({m.path.name for m in tagged}) == 2


def test_build_groups_accepts_per_group_overrides(two_scene_clip):
    models = vscm.build_groups(
        test=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
        ref=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
        directions=8,
        group_kwargs=[{}, {"directions": 4}],
    )
    assert len(models) == 2
    assert models[0].stats.directions == 8
    assert models[1].stats.directions == 4


# --- apply_groups -----------------------------------------------------------


def test_apply_groups_preserves_length_and_geometry(two_scene_clip):
    models = vscm.build_groups(
        test=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
        ref=[vscm.trim(two_scene_clip, 0, 8), vscm.trim(two_scene_clip, 8, 16)],
        directions=8,
    )
    out = vscm.apply_groups(two_scene_clip, models, [(0, 8), (8, 16)])
    assert (out.width, out.height, out.num_frames) == (48, 32, 16)
    assert out.format.id == two_scene_clip.format.id


def test_apply_groups_leaves_uncovered_frames_untouched(two_scene_clip):
    """A gap means "not graded", not "graded with the nearest model"."""
    models = vscm.build_groups(
        test=[vscm.trim(two_scene_clip, 0, 4)],
        ref=[vscm.trim(two_scene_clip, 0, 4)],
        directions=8,
    )
    out = vscm.apply_groups(two_scene_clip, models, [(0, 4)])
    assert out.num_frames == two_scene_clip.num_frames
    for n in (4, 7, 15):
        a = np.asarray(two_scene_clip.get_frame(n)[0])
        b = np.asarray(out.get_frame(n)[0])
        assert np.array_equal(a, b), f"frame {n} should have been left alone"


def test_apply_groups_handles_a_single_range_covering_everything(two_scene_clip):
    models = vscm.build_groups(
        test=[two_scene_clip], ref=[two_scene_clip], directions=8
    )
    out = vscm.apply_groups(two_scene_clip, models, [(0, 16)])
    assert out.num_frames == 16


def test_apply_groups_rejects_a_model_count_mismatch(two_scene_clip):
    models = vscm.build_groups(test=[two_scene_clip], ref=[two_scene_clip], directions=8)
    with pytest.raises(ValueError, match="must correspond one to one"):
        vscm.apply_groups(two_scene_clip, models, [(0, 8), (8, 16)])
    with pytest.raises(ValueError, match="must correspond one to one"):
        vscm.apply_groups(two_scene_clip, models * 2, [(0, 16)])


def test_apply_groups_rejects_malformed_ranges(two_scene_clip):
    models = vscm.build_groups(test=[two_scene_clip], ref=[two_scene_clip], directions=8)
    with pytest.raises(ValueError, match="past the end"):
        vscm.apply_groups(two_scene_clip, models, [(0, 999)])


def test_apply_groups_output_is_contiguous_even_with_many_gaps(two_scene_clip):
    """Gaps and ranges interleave; the spliced result must still be the right length."""
    models = vscm.build_groups(
        test=[vscm.trim(two_scene_clip, 2, 4), vscm.trim(two_scene_clip, 9, 12)],
        ref=[vscm.trim(two_scene_clip, 2, 4), vscm.trim(two_scene_clip, 9, 12)],
        directions=8,
    )
    out = vscm.apply_groups(two_scene_clip, models, [(2, 4), (9, 12)])
    assert out.num_frames == 16
    # Frames outside both ranges are byte-identical.
    for n in (0, 1, 4, 5, 8, 12, 15):
        assert np.array_equal(
            np.asarray(two_scene_clip.get_frame(n)[0]), np.asarray(out.get_frame(n)[0])
        ), n


# --- detect_scenes ----------------------------------------------------------


def test_detect_scenes_finds_the_two_scenes(two_scene_clip):
    ranges = vscm.detect_scenes(two_scene_clip, threshold=0.1)
    assert ranges == [(0, 8), (8, 16)], ranges


def test_detect_scenes_ranges_always_cover_the_clip(two_scene_clip):
    ranges = vscm.detect_scenes(two_scene_clip)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == two_scene_clip.num_frames
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert end == start, "detected ranges must be contiguous"
    assert sum(e - s for s, e in ranges) == two_scene_clip.num_frames


def test_detect_scenes_rejects_a_non_positive_threshold():
    with pytest.raises(ValueError, match="must be positive"):
        vscm.detect_scenes(core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=4), 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        vscm.detect_scenes(core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=4), -1.0)


def test_detect_scenes_rejects_an_empty_clip():
    """VapourSynth cannot build a zero-frame clip, so this guard is reached with a stub."""

    class Empty:
        num_frames = 0

    with pytest.raises(ValueError, match="no frames"):
        vscm.detect_scenes(Empty())


def test_detect_scenes_on_a_single_frame_clip():
    """One frame is one scene, and the detector cannot run without a neighbour to compare against."""
    one = core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=1)
    assert vscm.detect_scenes(one) == [(0, 1)]


def test_detect_scenes_on_a_two_frame_clip():
    two = core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=2)
    ranges = vscm.detect_scenes(two)
    assert ranges[0][0] == 0 and ranges[-1][1] == 2
    assert sum(e - s for s, e in ranges) == 2


# --- match_groups -----------------------------------------------------------


def test_match_groups_detects_and_applies(two_scene_clip):
    out = vscm.match_groups(two_scene_clip, two_scene_clip, directions=8)
    assert out.num_frames == two_scene_clip.num_frames
    assert out.format.id == two_scene_clip.format.id


def test_match_groups_rejects_a_short_frame_aligned_reference(two_scene_clip):
    short = core.std.BlankClip(width=48, height=32, format=vs.RGB48, length=4)
    with pytest.raises(ValueError, match="at least as long as"):
        vscm.match_groups(two_scene_clip, short, ranges=[(0, 8), (8, 16)], directions=8)


def test_match_groups_rejects_a_reference_count_mismatch(two_scene_clip):
    with pytest.raises(ValueError, match="correspond one to one"):
        vscm.match_groups(
            two_scene_clip,
            [two_scene_clip],
            ranges=[(0, 8), (8, 16)],
            directions=8,
        )


def test_match_groups_accepts_an_independent_reference_per_range(two_scene_clip):
    """A list of references keeps the two sides independent, unlike a frame-aligned clip."""
    refs = [
        core.std.BlankClip(width=48, height=32, format=vs.RGB48, length=5),
        core.std.BlankClip(width=48, height=32, format=vs.RGB48, length=3),
    ]
    out = vscm.match_groups(two_scene_clip, refs, ranges=[(0, 8), (8, 16)], directions=8)
    assert out.num_frames == 16


def test_match_groups_validates_explicit_ranges(two_scene_clip):
    with pytest.raises(ValueError, match="past the end"):
        vscm.match_groups(two_scene_clip, two_scene_clip, ranges=[(0, 999)], directions=8)


# --- does per-scene grading actually help? ----------------------------------


def _two_look_clip(seed: int = 11):
    """A clip whose two halves were graded differently, with a matching reference.

    This is the case per-scene models exist for: one global model cannot be right for both halves, so
    the error on each half is the thing to measure.
    """
    h, w, per = 32, 48, 12
    rng = np.random.default_rng(seed)
    look_a = np.array([[0.88, 0.08, 0.04], [0.05, 0.91, 0.04], [0.02, 0.10, 0.88]])
    look_b = np.array([[0.70, 0.20, 0.10], [0.10, 0.80, 0.10], [0.05, 0.25, 0.70]])
    bases, graded = [], []
    for i in range(per * 2):
        base = np.clip(0.25 + 0.18 * rng.standard_normal((h, w, 3)), 0, 1)
        look = look_a if i < per else look_b
        g = np.clip(base.reshape(-1, 3) @ look.T, 0, 1).reshape(base.shape)
        bases.append(base)
        graded.append(g)

    def make(arrs):
        blank = core.std.BlankClip(width=w, height=h, format=vs.RGB48, length=len(arrs))

        def fill(n, f):
            f = f.copy()
            for p in range(3):
                np.asarray(f[p])[:] = (arrs[n][:, :, p] * 65535).astype(np.uint16)
            return f

        return core.std.ModifyFrame(blank, blank, fill)

    return make(bases), make(graded), per


def _half_errors(clip, ref, out, per):
    """Mean absolute error per half, as a fraction of full scale."""
    errors = []
    for start, end in ((0, per), (per, 2 * per)):
        total, count = 0.0, 0
        for n in range(start, end):
            want = np.asarray(ref.get_frame(n)[0]).astype(np.float64) / 65535.0
            got = np.asarray(out.get_frame(n)[0]).astype(np.float64) / 65535.0
            total += float(np.abs(want - got).mean())
            count += 1
        errors.append(total / count)
    return errors


def test_per_scene_models_beat_one_global_model_when_the_grading_differs():
    """The point of the feature, measured rather than asserted.

    Two halves graded with different matrices. A single model has to compromise and is wrong on both;
    per-scene models are near-exact on each.
    """
    clip, ref, per = _two_look_clip()

    global_out = vscm.match(clip, ref, directions=16)
    global_errors = _half_errors(clip, ref, global_out, per)

    scene_out = vscm.match_groups(clip, ref, ranges=[(0, per), (per, 2 * per)], directions=16)
    scene_errors = _half_errors(clip, ref, scene_out, per)

    for i, (g, s) in enumerate(zip(global_errors, scene_errors)):
        assert s < g, f"half {i}: per-scene {s:.5f} should beat global {g:.5f}"
    # A single model built from both looks is a compromise, so it should be clearly worse on at least
    # one half. If this stops holding, the test above is passing for the wrong reason.
    assert max(global_errors) > 2.0 * max(scene_errors), (global_errors, scene_errors)


def test_a_one_frame_scene_is_handled():
    """A scene of one frame is legitimate (a flash, a cut on the last frame) and must not break."""
    clip, ref, per = _two_look_clip(seed=5)
    ranges = [(0, 1), (1, per), (per, 2 * per)]
    out = vscm.match_groups(clip, ref, ranges=ranges, directions=4)
    assert out.num_frames == clip.num_frames
    assert out.format.id == clip.format.id


def test_scenes_shorter_than_the_frame_budget_are_handled():
    """`frames="even:64"` on a 3-frame scene must sample 3 frames, not fail."""
    clip, ref, per = _two_look_clip(seed=6)
    out = vscm.match_groups(clip, ref, ranges=[(0, 3), (3, 2 * per)], directions=4, frames="even:64")
    assert out.num_frames == clip.num_frames
