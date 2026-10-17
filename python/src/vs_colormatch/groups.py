"""Per-scene and per-range models.

The plugin deliberately knows nothing about segments: it builds one model from a set of clips and
applies one model to a clip. Everything here is graph assembly on top of that, which is why adding
per-scene grading did not touch the plugin, the model format or the shader.

The motivating case is a reel whose grading is not uniform — a different reference for a flashback, a
day-for-night sequence, or a reel assembled from several sources. One global model is the right answer
for a uniformly graded reel and the wrong answer for one that is not, and the difference is visible as
a systematic error on the segments that disagree.

Two shapes are supported:

* :func:`build_groups` / :func:`apply_groups` — explicit control. You say which clips make each model
  and which frame range each model covers.
* :func:`match_groups` — the common case. Give a clip, a reference and a list of ranges (or let
  :func:`detect_scenes` find them) and it builds one model per range and splices the result back.

Both keep the output identical to the input in width, height, format and length, and both report what
they did: which range got which model, and which frames were left alone.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import vapoursynth as vs

from . import api
from .model import Model

__all__ = [
    "SceneGroup",
    "build_groups",
    "apply_groups",
    "match_groups",
    "detect_scenes",
    "check_ranges",
]

core = vs.core

#: A half-open frame range, ``[start, end)``.
FrameRange = tuple[int, int]


@dataclass(frozen=True)
class SceneGroup:
    """One range and the model that covers it."""

    start: int
    end: int
    model: Model

    @property
    def length(self) -> int:
        """Number of frames in the range."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}) {self.model}"


def check_ranges(ranges: Sequence[FrameRange], num_frames: int) -> list[FrameRange]:
    """Validates and normalises a list of half-open frame ranges.

    Args:
        ranges: The ranges to check.
        num_frames: Length of the clip they apply to.

    Returns:
        The same ranges, verified sorted, non-overlapping, in range and non-empty. Empty ranges are
        dropped rather than rejected, because a scene detector can legitimately emit one.

    Raises:
        ValueError: If a range is malformed, out of bounds, out of order or overlapping.
    """
    if not ranges:
        raise ValueError("vs_colormatch: `ranges` must contain at least one range")

    cleaned: list[FrameRange] = []
    for i, rng in enumerate(ranges):
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            raise ValueError(
                f"vs_colormatch: `ranges[{i}]` must be a (start, end) pair, got {rng!r}"
            )
        start, end = int(rng[0]), int(rng[1])
        if start < 0:
            raise ValueError(f"vs_colormatch: `ranges[{i}]` start ({start}) is negative")
        if end < start:
            raise ValueError(
                f"vs_colormatch: `ranges[{i}]` end ({end}) is before its start ({start})"
            )
        if end > num_frames:
            raise ValueError(
                f"vs_colormatch: `ranges[{i}]` end ({end}) is past the end of the clip "
                f"({num_frames} frames)"
            )
        if end == start:
            continue
        if cleaned and start < cleaned[-1][1]:
            raise ValueError(
                f"vs_colormatch: `ranges[{i}]` ({start}, {end}) overlaps or precedes "
                f"({cleaned[-1][0]}, {cleaned[-1][1]}); ranges must be sorted and disjoint"
            )
        cleaned.append((start, end))

    if not cleaned:
        raise ValueError(
            "vs_colormatch: every range is empty; there is nothing to apply a model to"
        )
    return cleaned


def build_groups(
    *,
    test: Sequence[Any],
    ref: Sequence[Any],
    group_kwargs: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[Model]:
    """Builds one model per group.

    Args:
        test: One entry per group. Each entry is a clip or a list of clips to sample from.
        ref: One entry per group, in the same shape as ``test``.
        group_kwargs: Optional per-group overrides, merged over ``kwargs``. Use this when a segment
            needs different settings — more directions for a difficult scene, say.
        **kwargs: Shared arguments for every :func:`~vs_colormatch.build` call.

    Returns:
        One :class:`~vs_colormatch.model.Model` per group, in order.

    Raises:
        ValueError: If the two sides have different lengths, or a group is empty.
    """
    if len(test) != len(ref):
        raise ValueError(
            f"vs_colormatch.build_groups: `test` and `ref` must have one entry per group, got "
            f"{len(test)} and {len(ref)}"
        )
    if not test:
        raise ValueError("vs_colormatch.build_groups: at least one group is required")
    if group_kwargs is not None and len(group_kwargs) != len(test):
        raise ValueError(
            f"vs_colormatch.build_groups: `group_kwargs` must have one entry per group "
            f"({len(test)}), got {len(group_kwargs)}"
        )

    models: list[Model] = []
    for i in range(len(test)):
        merged = dict(kwargs)
        if group_kwargs is not None:
            merged.update(group_kwargs[i])
        # Each group needs its own cache entry. The configuration key covers the arguments and the
        # clip geometry but not *which* clips, so two groups with the same shapes and settings would
        # otherwise share one — and the second group would silently receive the first group's model.
        # Suffixing the tag always, including when the caller supplied one, is what prevents that.
        if merged.get("cache"):
            user_tag = merged.get("cache_tag")
            merged["cache_tag"] = f"{user_tag}|group{i}" if user_tag else f"group{i}"
        models.append(api.build(test=test[i], ref=ref[i], **merged))
    return models


def apply_groups(
    clip: vs.VideoNode,
    models: Sequence[Model],
    ranges: Sequence[FrameRange],
    *,
    backend: str | Any | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """Applies one model per frame range and splices the result back together.

    Frames not covered by any range pass through **unchanged** rather than being given a nearby
    model. That is the conservative choice: a wrong model is a visible artefact, while an ungraded
    frame is merely ungraded, and the caller is told which ranges were left alone.

    Args:
        clip: The clip to transform.
        models: One model per range, in the same order.
        ranges: Half-open ``(start, end)`` ranges, sorted and disjoint. They need not cover the clip.
        backend: Backend name or :class:`~vs_colormatch.backend.Backend`.
        **kwargs: Arguments for :func:`~vs_colormatch.apply`.

    Returns:
        A clip with the same width, height, format, framerate and length as ``clip``.

    Raises:
        ValueError: If the number of models does not match the number of ranges, or the ranges are
            malformed.
    """
    ranges = check_ranges(ranges, clip.num_frames)
    if len(models) != len(ranges):
        raise ValueError(
            f"vs_colormatch.apply_groups: {len(models)} model(s) for {len(ranges)} range(s); they "
            f"must correspond one to one"
        )

    parts: list[vs.VideoNode] = []
    cursor = 0
    for (start, end), model in zip(ranges, models):
        if start > cursor:
            # A gap: keep the source frames as they are.
            parts.append(api.trim(clip, cursor, start))
        parts.append(api.apply(api.trim(clip, start, end), model, backend=backend, **kwargs))
        cursor = end
    if cursor < clip.num_frames:
        parts.append(api.trim(clip, cursor, clip.num_frames))

    if not parts:
        raise ValueError("vs_colormatch.apply_groups: nothing to output")
    if len(parts) == 1:
        return parts[0]
    # Every part has the same format and geometry: `apply` preserves both, and `trim` cannot change
    # them, so `Splice` is valid and the length adds up to the original.
    return core.std.Splice(parts)


def detect_scenes(clip: vs.VideoNode, threshold: float = 0.1) -> list[FrameRange]:
    """Finds scene boundaries and returns them as half-open ranges covering the whole clip.

    Requires ``vsjetpack``, whose ``sc_detect`` is a tuned detector; writing another one here would
    mean shipping a worse version of a solved problem. If it is not installed, pass ``ranges``
    explicitly instead.

    Args:
        clip: The clip to analyse.
        threshold: Detector sensitivity. Higher is less sensitive; 0.1 is vsjetpack's default.

    Returns:
        Contiguous ranges covering every frame, e.g. ``[(0, 48), (48, 121), ...]``.

    Raises:
        RuntimeError: If ``vsjetpack`` is not installed.
        ValueError: If the clip has no frames or the threshold is not positive.
    """
    # VapourSynth cannot represent a zero-frame clip (`std.Trim` refuses), so this guard is
    # defensive rather than reachable. It costs nothing and makes the failure mode explicit.
    if clip.num_frames == 0:
        raise ValueError("vs_colormatch.detect_scenes: the clip has no frames")
    if not (threshold > 0.0):
        raise ValueError(
            f"vs_colormatch.detect_scenes: `threshold` must be positive, got {threshold}"
        )
    # A single frame is one scene, and there is nothing to compare it against: `vstools.sc_detect`
    # shifts the clip by one frame, which it cannot do with no neighbour, and fails with a bare
    # "Trim: invalid last frame" from inside the dependency.
    if clip.num_frames == 1:
        return [(0, 1)]
    try:
        import vstools
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "vs_colormatch.detect_scenes: scene detection needs `vsjetpack` for its `sc_detect`. "
            "Install it, or pass `ranges=` explicitly."
        ) from exc

    try:
        marked = vstools.sc_detect(clip, threshold)
    except Exception as exc:  # noqa: BLE001 - attribute the dependency's failure
        raise RuntimeError(
            f"vs_colormatch.detect_scenes: `vstools.sc_detect` failed on a {clip.num_frames}-frame "
            f"clip ({clip.width}x{clip.height}, {clip.format.name}): {exc}. Pass `ranges=` explicitly "
            f"if the detector cannot handle this source."
        ) from exc

    cuts = [0]
    for n in range(clip.num_frames):
        # `_SceneChangeNext` marks the last frame of a scene, so the next scene starts at n + 1.
        if marked.get_frame(n).props.get("_SceneChangeNext", 0):
            cuts.append(n + 1)
    cuts.append(clip.num_frames)

    # The detector can mark the same boundary twice at a hard cut; dedupe rather than emit an empty
    # range, which `check_ranges` would then drop anyway.
    unique = sorted({c for c in cuts if 0 <= c <= clip.num_frames})
    return [(unique[i], unique[i + 1]) for i in range(len(unique) - 1) if unique[i] < unique[i + 1]]


def match_groups(
    clip: vs.VideoNode,
    ref: vs.VideoNode | Sequence[vs.VideoNode],
    ranges: Sequence[FrameRange] | None = None,
    *,
    threshold: float = 0.1,
    backend: str | Any | None = None,
    compat: str = "v2",
    verbose: bool = False,
    **kwargs: Any,
) -> vs.VideoNode:
    """Builds one model per range and applies it, in one call.

    Args:
        clip: The clip to transform.
        ref: The reference. A single clip is sliced per range, which means it must be at least as
            long as ``clip`` and frame-aligned with it — that is a departure from the single-model
            path, where the reference is independent, and it is unavoidable when the reference is
            being cut into the same segments. A sequence is used as one reference per range instead,
            which keeps the two sides independent.
        ranges: Half-open ``(start, end)`` ranges. Detected with :func:`detect_scenes` when omitted.
        threshold: Scene-detection sensitivity, used only when ``ranges`` is omitted.
        backend: Backend name or :class:`~vs_colormatch.backend.Backend`.
        compat: Which builder fits the transfer tables; see :func:`~vs_colormatch.build`.
        verbose: Print the range and model summary.
        **kwargs: Arguments for :func:`~vs_colormatch.build`.

    Returns:
        A clip with the same width, height, format, framerate and length as ``clip``.
    """
    if ranges is None:
        ranges = detect_scenes(clip, threshold)
        if verbose:
            print(f"vs_colormatch: detected {len(ranges)} scene(s)")
    ranges = check_ranges(ranges, clip.num_frames)

    if isinstance(ref, vs.VideoNode):
        if ref.num_frames < clip.num_frames:
            raise ValueError(
                f"vs_colormatch.match_groups: a frame-aligned reference must be at least as long as "
                f"the clip ({clip.num_frames} frames), got {ref.num_frames}. Pass a list of clips to "
                f"use an independent reference per range instead."
            )
        refs = [api.trim(ref, start, end) for start, end in ranges]
    else:
        refs = list(ref)
        if len(refs) != len(ranges):
            raise ValueError(
                f"vs_colormatch.match_groups: {len(refs)} reference clip(s) for {len(ranges)} "
                f"range(s); they must correspond one to one"
            )

    tests = [api.trim(clip, start, end) for start, end in ranges]
    models = build_groups(test=tests, ref=refs, compat=compat, verbose=verbose, **kwargs)

    if verbose:
        covered = sum(end - start for start, end in ranges)
        print(f"vs_colormatch: {len(ranges)} range(s) covering {covered}/{clip.num_frames} frames")
        for (start, end), model in zip(ranges, models):
            print(f"  [{start:>6}, {end:>6})  {model}")
        if covered < clip.num_frames:
            print(f"vs_colormatch: {clip.num_frames - covered} frame(s) left ungraded")

    return apply_groups(clip, models, ranges, backend=backend)
