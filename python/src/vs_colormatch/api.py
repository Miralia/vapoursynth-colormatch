"""The public API: ``build``, ``apply``, ``match``, plus the two segment helpers.

The design rule is that the wrapper adds *convenience*, never a new concept. Inputs are ordinary
:class:`vapoursynth.VideoNode` objects; a segment is just a clip you already sliced with
``std.Trim`` / ``std.Crop``. The plugin takes arrays of clips, so there is no pairing to express:
every ``test`` clip contributes to the source distribution and every ``ref`` clip to the target one,
and the model is global unless you build several.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

import vapoursynth as vs

from . import settings
from .backend import Backend, list_adapters
from .model import Model, ModelStats
from .validate import normalize_inputs, summarize

__all__ = [
    "apply",
    "build",
    "crop",
    "load",
    "load_plugin",
    "match",
    "trim",
]

core = vs.core

_PLUGIN_ENV = "VSCOLORMATCH_PLUGIN"
_plugin_loaded = False


_LIB_NAMES = ("libcolormatch.dylib", "libcolormatch.so", "colormatch.dll")


def _default_plugin_path() -> Path | None:
    """Finds the plugin for a wheel install or an in-tree build.

    Two locations are searched, in order:

    1. ``<site-packages>/vapoursynth/plugins/colormatch/`` — where the wheel puts it. VapourSynth
       auto-loads this, but only when ``vs-colormatch`` is installed into the *same* environment as
       ``vapoursynth`` itself: ``vapoursynth.get_plugin_dir()`` is resolved relative to the
       ``vapoursynth`` module's own directory, so a plugin in a different site-packages (a venv
       created with ``--system-site-packages``, for instance) is not found automatically. Loading it
       explicitly is what makes that case work.
    2. ``<repo>/target/release/`` — an in-tree ``cargo build``.
    """
    here = Path(__file__).resolve()

    # The wheel layout: sibling of this package, under the `vapoursynth` namespace directory.
    for parent in here.parents:
        bundled = parent / "vapoursynth" / "plugins" / "colormatch"
        if bundled.is_dir():
            for name in _LIB_NAMES:
                if (bundled / name).exists():
                    return bundled / name

    # The in-tree layout.
    for parent in here.parents:
        release = parent / "target" / "release"
        if release.is_dir():
            for name in _LIB_NAMES:
                if (release / name).exists():
                    return release / name
    return None


def load_plugin(path: str | os.PathLike[str] | None = None) -> None:
    """Ensures the ``cm`` namespace is available.

    VapourSynth auto-loads plugins under ``vapoursynth/plugins/``, but only when this package and
    ``vapoursynth`` share a site-packages. When they do not, this finds the copy the wheel shipped
    next to this package and loads it explicitly. It is a no-op if ``core.cm`` already exists.
    """
    global _plugin_loaded
    if hasattr(core, "cm"):
        _plugin_loaded = True
        return
    if _plugin_loaded:
        return

    candidate = path or os.getenv(_PLUGIN_ENV) or _default_plugin_path()
    if candidate is None:
        raise RuntimeError(
            "vs_colormatch: the `cm` plugin is not loaded and no build was found. Install the wheel, "
            f"or set ${_PLUGIN_ENV} to the plugin path, or pass `plugin=` to load_plugin()."
        )
    core.std.LoadPlugin(path=str(candidate))
    _plugin_loaded = True


def trim(clip: vs.VideoNode, start: int, end: int | None = None) -> vs.VideoNode:
    """``std.Trim`` with a half-open ``[start, end)`` range.

    Thin on purpose: ``trim(clip, 100, 200)`` reads better than ``clip.std.Trim(100, 199)`` and
    removes the inclusive-end off-by-one that this kind of range work invites.
    """
    if start < 0:
        raise ValueError(f"vs_colormatch.trim: start must not be negative, got {start}")
    if end is None:
        return clip.std.Trim(first=start)
    if end <= start:
        raise ValueError(f"vs_colormatch.trim: end ({end}) must be greater than start ({start})")
    if start >= clip.num_frames:
        raise ValueError(
            f"vs_colormatch.trim: start ({start}) is past the end of a {clip.num_frames}-frame clip"
        )
    if end > clip.num_frames:
        # An error rather than a clamp, because the rest of this package is strict about ranges:
        # `check_ranges` rejects an out-of-bounds end and so does `crop`. Silently clamping here meant
        # `trim(clip, 0, 99)` on a 4-frame clip quietly returned 4 frames while `trim(clip, 99, 100)`
        # raised, which is the same mistake failing two different ways.
        raise ValueError(
            f"vs_colormatch.trim: end ({end}) is past the end of a {clip.num_frames}-frame clip"
        )
    return clip.std.Trim(first=start, last=end - 1)


def crop(
    clip: vs.VideoNode,
    *,
    x: tuple[int, int] | None = None,
    y: tuple[int, int] | None = None,
) -> vs.VideoNode:
    """``std.Crop`` with half-open ``(start, end)`` pixel ranges.

    Both ranges are optional and independent, so ``crop(clip, x=(300, 400))`` keeps full height.
    """
    left = top = 0
    right = clip.width
    bottom = clip.height
    if x is not None:
        left, right = x
    if y is not None:
        top, bottom = y
    if not (0 <= left < right <= clip.width):
        raise ValueError(f"vs_colormatch.crop: x range ({left}, {right}) is not inside 0..{clip.width}")
    if not (0 <= top < bottom <= clip.height):
        raise ValueError(f"vs_colormatch.crop: y range ({top}, {bottom}) is not inside 0..{clip.height}")
    if (left, top, right, bottom) == (0, 0, clip.width, clip.height):
        return clip
    return clip.std.Crop(left=left, right=clip.width - right, top=top, bottom=clip.height - bottom)


def _as_list(clips: vs.VideoNode | Sequence[vs.VideoNode] | None, name: str) -> list[vs.VideoNode]:
    if clips is None:
        raise ValueError(f"vs_colormatch: `{name}` is required")
    if isinstance(clips, vs.VideoNode):
        return [clips]
    items = list(clips)
    if not items:
        raise ValueError(f"vs_colormatch: `{name}` must contain at least one clip")
    return items


def build(
    *,
    test: vs.VideoNode | Sequence[vs.VideoNode],
    ref: vs.VideoNode | Sequence[vs.VideoNode],
    weights: Sequence[float] | None = None,
    frames: str = "even:64",
    samples: int = 262144,
    directions: int = 32,
    bins: int = 256,
    axes: str = "fibonacci",
    passes: int = 1,
    step: float = 1.0,
    seed: int = 0x5EED,
    compat: str = "v2",
    matrix: int | None = None,
    range: int | None = None,
    strict: bool = True,
    convert_subsampled: bool = False,
    cache: bool | str | os.PathLike[str] = False,
    cache_dir: str | os.PathLike[str] | None = None,
    cache_tag: str | None = None,
    plugin: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> Model:
    """Pools samples from several clips into one global model.

    Args:
        test: The material to be corrected. One clip or a list; segments are just trimmed clips.
        ref: The target look. Independent of ``test``: different length, resolution and format are
            all fine, and the two sides are pooled all-to-all rather than paired.
        weights: Optional per-clip weights, ordered ``test...`` then ``ref...``.
        frames: Frame selection: ``"even:N"``, ``"every:N"``, ``"all"`` or ``"0,5,9"``.
        samples: Total sample pixels per side. ~20k already saturates; 262144 is comfortable.
        directions: Number of projection directions (``K``). 32 is the measured sweet spot.
        bins: Histogram bins per axis (``B``). 256 is ample.
        axes: ``"fibonacci"`` (deterministic, default) or ``"drdre"``.
        passes: Extra sweeps over the direction list. 1 reproduces the original's behaviour.
        step: Gradient-flow step size. The original uses a full step.
        seed: Seed for the axes scheme.
        compat: Which builder fits the transfer tables. ``"v2"`` (default) is the one this project
            is built around. ``"drdre"`` reproduces the original's structure — percentile-paired
            knots, monotonicity by deletion, `±√3` sentinels, a monotone cubic through the knots — so
            the two can be compared. It is not bit-identical to the MATLAB original; see the
            ``Builder::DrDre`` documentation for the two approximated details.
        matrix: Force a matrix coefficient. Resolved from ``_Matrix`` when omitted.
        range: Force a range (0 limited, 1 full). Resolved from ``_Range`` when omitted.
        strict: Raise rather than guess when the colour space is ambiguous.
        convert_subsampled: Upsample 4:2:0/4:2:2 to 4:4:4 before matching. Off by default; the
            plugin resamples chroma internally and keeps the clip's format.
        cache: ``True`` to cache under a configuration key, or an explicit path to write to.
        cache_dir: Override the cache directory.
        cache_tag: Extra text mixed into the configuration key. **The key does not look at pixel
            content** — see the note on caching below — so pass something content-identifying here
            (a source filename, a checksum you already have) when the same script runs over
            different footage of the same geometry.
        plugin: Plugin path, for in-tree use.
        verbose: Print the normalisation report and the build statistics.

    Returns:
        A :class:`~vs_colormatch.model.Model`.

    Note:
        ``cache=True`` keys on the *configuration*: the build arguments, the clip geometries, and
        :func:`settings.ALGORITHM_TAG`. It does **not** look at pixel content, and that is a
        deliberate trade rather than an oversight. Any content-sensitive key would have to evaluate
        the sampled frames, which is most of the cost of the build itself — so it could not save
        time. What the cache does give you is stability: re-running the same script reuses the same
        model instead of refitting it.

        The hazard is real, so it is worth stating plainly: two different clips of the same
        resolution, format and length produce the same key. Pass ``cache_tag=`` with something
        content-identifying when that can happen.
    """
    load_plugin(plugin)

    test_clips = _as_list(test, "test")
    ref_clips = _as_list(ref, "ref")
    all_clips = test_clips + ref_clips

    compat_key = compat.strip().lower()
    if compat_key not in ("v2", "drdre"):
        raise ValueError(
            f"vs_colormatch.build: `compat` must be \"v2\" or \"drdre\", got {compat!r}"
        )

    if weights is not None and len(weights) != len(all_clips):
        raise ValueError(
            f"vs_colormatch.build: `weights` must have one entry per clip "
            f"({len(all_clips)}), got {len(weights)}"
        )

    normalized, changes = normalize_inputs(
        all_clips,
        matrix=matrix,
        range=range,
        strict=strict,
        convert_subsampled=convert_subsampled,
    )

    if verbose:
        print(summarize(changes, normalized))

    params = dict(
        directions=directions,
        bins=bins,
        axes=axes,
        passes=passes,
        step=step,
        samples=samples,
        frames=frames,
        compat=compat_key,
        matrix=matrix,
        range=range,
    )

    # Everything that can change the resulting model has to be in the key, or a cache hit hands back
    # a model built under different settings.
    #
    # `ALGORITHM_TAG` is the one that is easy to forget: the model format's own version covers the
    # byte layout, not the *meaning* of the bytes. 0.2.0 changed the gamut default, the grid
    # extrapolation, the CDF inversion and added smoothing, so a `.cm` written by 0.1.0 describes a
    # different transform for identical arguments. Without the tag, upgrading the plugin would
    # silently keep serving the old model.
    key = settings.config_key(
        [
            settings.ALGORITHM_TAG,
            directions,
            bins,
            axes,
            passes,
            step,
            seed,
            # `compat` changes the fit, so it has to be in the key or the two builders would share
            # cache entries and whichever ran first would win.
            compat_key,
            samples,
            frames,
            matrix or -1,
            range if range is not None else -1,
            len(test_clips),
            len(ref_clips),
            # Per-clip weights change the pooled distributions, so they change the model.
            *(weights if weights is not None else []),
            cache_tag or "",
            *[f"{c.width}x{c.height}x{c.format.id}x{c.num_frames}" for c in all_clips],
        ]
    )

    kwargs: dict = dict(
        test=[item.clip for item in normalized[: len(test_clips)]],
        ref=[item.clip for item in normalized[len(test_clips) :]],
        samples=samples,
        directions=directions,
        bins=bins,
        axes=axes,
        passes=passes,
        step=step,
        seed=seed,
        frames=frames,
        compat=compat_key,
    )
    # One colour space per input clip.
    #
    # The plugin used to take a single `matrix`/`range` for the whole call, and the wrapper only
    # passed them when every input agreed. That silently broke two ordinary things: a YUV test clip
    # against an RGB reference (no agreement, so no matrix was passed and the plugin refused to guess
    # one for the YUV side), and two YUV clips of different ranges (no agreement, so both were
    # treated as limited). Both now work, because the plugin takes per-input arrays.
    #
    # `-1` means "unspecified", which the plugin resolves to the family default. An RGB or grayscale
    # input has no matrix of its own, which is why `matrix` may be `None` there.
    kwargs["matrices"] = [-1 if item.matrix is None else int(item.matrix) for item in normalized]
    kwargs["ranges"] = [int(item.range) for item in normalized]

    if weights is not None:
        kwargs["weights"] = list(weights)

    target: Path | None = None
    if cache is True:
        target = settings.get_cache_dir(cache_dir) / f"{key}.cm"
    elif cache:
        candidate = Path(str(cache))
        target = candidate if candidate.is_absolute() else settings.get_cache_dir(cache_dir) / candidate

    if target is not None and target.exists():
        if verbose:
            print(f"vs_colormatch: reusing cached model {target}")
        kwargs.pop("cache", None)
        return _model_from_file(target, params, verbose)

    if target is not None:
        # The plugin writes the file itself; make sure the directory exists first so that a missing
        # cache directory surfaces as a normal error here rather than a failed write inside the
        # filter.
        target.parent.mkdir(parents=True, exist_ok=True)
        kwargs["cache"] = str(target)

    model_clip = core.cm.BuildModel(**kwargs)
    frame = model_clip.get_frame(0)
    stats = ModelStats.from_props(frame.props["_CM_Stats"])

    if verbose:
        print(f"vs_colormatch: {stats}")

    return Model(clip=model_clip, stats=stats, params=params, path=target)


def _model_from_file(path: Path, params: dict, verbose: bool) -> Model:
    """Reconstructs a :class:`Model` from a cached ``.cm`` file without rebuilding.

    The build statistics travel in the ``.cm`` header, so a cache hit reports the same provenance as
    the original build rather than fabricating zeros.
    """
    from .cmfile import read_header

    header = read_header(path)
    stats = ModelStats(
        samples=header.samples,
        frames=header.frames,
        w1_before=header.w1_before,
        w1_after=header.w1_after,
        max_gain=header.max_gain,
        build_ms=header.build_ms,
        directions=header.directions,
    )
    # `cm.BuildModel(model_file=...)` loads a model and emits a model clip, so reloading and building
    # produce the same kind of object. This used to run `cm.Apply` on a throwaway 1x1 clip, which
    # conflated loading a model with applying one.
    loaded = core.cm.BuildModel(model_file=str(path))
    if verbose:
        print(f"vs_colormatch: loaded {path}")
        print(f"vs_colormatch: {stats}")
    return Model(clip=loaded, stats=stats, params=params, path=path)


def load(
    path: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
    plugin: str | os.PathLike[str] | None = None,
) -> Model:
    """Loads a serialized ``.cm`` model.

    The build statistics travel in the file's header, so the returned model reports the same
    provenance as the build that wrote it. Relative paths resolve against the cache directory, the
    same way :meth:`Model.save` writes them.

    Args:
        path: The ``.cm`` file, absolute or relative to the cache directory.
        cache_dir: Cache directory for a relative ``path``.
        verbose: Print the loaded statistics.
        plugin: Plugin path, for in-tree use.

    Returns:
        The loaded model.
    """
    load_plugin(plugin)
    target = Path(path)
    if not target.is_absolute():
        target = settings.get_cache_dir(cache_dir) / target
    if not target.is_file():
        raise FileNotFoundError(f"vs_colormatch.load: no model file at {target}")
    return _model_from_file(target, {}, verbose)


def apply(
    clip: vs.VideoNode,
    model: Model | str | os.PathLike[str],
    *,
    backend: str | Backend | None = None,
    device_id: int | None = None,
    method: str | None = None,
    lut_size: int | None = None,
    planes: list[int] | None = None,
    gamut: str | None = None,
    luma_lock: float | None = None,
    chroma: str | None = None,
    matrix: int | None = None,
    range: int | None = None,
    strict: bool = True,
    plugin: str | os.PathLike[str] | None = None,
) -> vs.VideoNode:
    """Applies a model to a clip.

    The output is identical to the input in width, height, format, framerate and length; the filter
    copies the input's video info and never resizes. Subsampled chroma is upsampled, transformed and
    downsampled inside the filter, so a 4:2:0 clip stays 4:2:0. The backend and method that actually
    ran are written to ``_CM_Backend`` and ``_CM_Method`` on every output frame.

    Args:
        clip: The clip to transform.
        model: A :class:`~vs_colormatch.model.Model`, a :class:`~pathlib.Path`, or a string path to
            a ``.cm`` file.
        backend: Backend name or :class:`~vs_colormatch.backend.Backend`.
        device_id: GPU index.
        method: ``"native"`` (exact, cost scales with the slice count) or ``"lut"``.
        lut_size: LUT edge length for the ``"lut"`` method: 17, 33 or 65.
        planes: Which output planes to transform. Unlisted planes keep their original data.
        gamut: ``"clip"`` or ``"soft"``.
        luma_lock: Restore the source luma by this fraction, in ``[0, 1]``.
        chroma: How full-resolution chroma is reduced back: ``"nearest"`` (default; makes an
            identity model an exact no-op) or ``"box"`` (better reconstruction, not exact).
        matrix: Matrix coefficient, resolved from ``_Matrix`` when omitted.
        range: Colour range, resolved from ``_Range`` when omitted.
        strict: Raise rather than guess when the colour space is ambiguous.
        plugin: Plugin path, for in-tree use.
    """
    load_plugin(plugin)

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f"vs_colormatch.apply: clip must be a vapoursynth clip, got {type(clip)!r}")

    if not isinstance(model, (Model, str, os.PathLike)):
        # Without this, an int or None became the *string* "42"/"None" and was reported as an
        # unreadable model file, which points at the filesystem instead of at the caller.
        raise TypeError(
            f"vs_colormatch.apply: `model` must be a Model or a path to a .cm file, got "
            f"{type(model).__name__}"
        )

    normalized, _ = normalize_inputs([clip], matrix=matrix, range=range, strict=strict)
    item = normalized[0]

    kwargs: dict = {"clip": item.clip}
    if item.matrix is not None:
        kwargs["matrix"] = item.matrix
    kwargs["range"] = item.range

    if isinstance(model, Model):
        kwargs["model"] = model.clip
    else:
        kwargs["model_file"] = str(model)

    if isinstance(backend, Backend):
        kwargs.update(backend.to_kwargs())
    elif backend is not None:
        kwargs["backend"] = backend
    if device_id is not None:
        kwargs["device_id"] = device_id
    if method is not None:
        kwargs["method"] = method
    if lut_size is not None:
        kwargs["lut_size"] = lut_size
    if planes is not None:
        kwargs["planes"] = list(planes)
    if gamut is not None:
        kwargs["gamut"] = gamut
    if luma_lock is not None:
        kwargs["luma_lock"] = luma_lock
    if chroma is not None:
        kwargs["chroma"] = chroma

    return core.cm.Apply(**kwargs)


def match(
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    *,
    backend: str | Backend | None = None,
    cache: bool | str | os.PathLike[str] = True,
    cache_dir: str | os.PathLike[str] | None = None,
    directions: int = 32,
    frames: str = "even:64",
    samples: int = 262144,
    cache_tag: str | None = None,
    compat: str = "v2",
    verbose: bool = False,
    apply_kwargs: Mapping[str, Any] | None = None,
    **build_kwargs: Any,
) -> vs.VideoNode:
    """Builds a model from ``clip`` and ``ref`` and applies it to ``clip``.

    The one-liner for the common case. ``clip`` and ``ref`` need not be the same length, resolution
    or format — unlike frame-aligned colour-transfer tools, the reference is only read for its
    distribution.

    Args:
        clip: The clip to transform.
        ref: The reference clip, read only for its colour distribution.
        backend: Backend name or :class:`~vs_colormatch.backend.Backend`.
        cache: ``True`` to reuse a model keyed on the configuration, or an explicit path.
        cache_dir: Override the cache directory.
        directions: Projection directions (``K``).
        frames: Frame selection for sampling.
        samples: Sample pixels per side.
        cache_tag: Extra text mixed into the cache key; see :func:`build`.
        compat: Which builder fits the transfer tables; see :func:`build`.
        verbose: Print the normalisation report and the build statistics.
        apply_kwargs: Keyword arguments for :func:`apply` — ``method``, ``lut_size``, ``chroma``,
            ``gamut``, ``luma_lock``, ``planes``. Kept separate because they are a different stage's
            arguments, and mixing them into one ``**kwargs`` would make a typo silently land in the
            wrong call.
        **build_kwargs: Any remaining :func:`build` argument (``bins``, ``axes``, ``passes``, ``step``,
            ``seed``, ``matrix``, ``range``, ``strict``, ``convert_subsampled``, ``weights``,
            ``plugin``).

    Returns:
        The transformed clip.
    """
    model = build(
        test=clip,
        ref=ref,
        directions=directions,
        frames=frames,
        samples=samples,
        cache=cache,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        compat=compat,
        verbose=verbose,
        **build_kwargs,
    )
    return apply(clip, model, backend=backend, **(apply_kwargs or {}))
