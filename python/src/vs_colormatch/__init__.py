"""``vs_colormatch`` — colour matching for VapourSynth.

Rotated-marginal sliced-optimal-transport colour matching, ported from Menno Dreischor's ColorMatch
and reworked for video: one model built from many frames, applied to a whole reel, with no bleed and
no per-frame flicker.

Quick start::

    import vapoursynth as vs
    import vs_colormatch as vscm

    # One line: make `clip` look like `ref`. They need not be the same length or resolution.
    out = vscm.match(clip, ref)

    # Or build once and reuse, pooling several segments from each side:
    model = vscm.build(
        test=[clip.std.Trim(100, 199), clip.std.Trim(300, 399)],
        ref=[ref.std.Trim(1100, 1199), ref.std.Trim(1300, 1399)],
        cache=True,
    )
    out = vscm.apply(clip, model)

Everything the plugin does is per-pixel and global, so the output is temporally stable by
construction and the transform can be exported as a LUT::

    model.export_lut("look.cube", size=33)

The plugin itself exposes only four filters; the graph-level work — slicing, normalisation, caching,
backend choice, reporting — lives here.
"""

from __future__ import annotations

from ._version import __version__, version_tuple__
from .api import apply, build, crop, load, load_plugin, match, trim
from .groups import (
    SceneGroup,
    apply_groups,
    build_groups,
    check_ranges,
    detect_scenes,
    match_groups,
)
from .backend import Adapter, Backend, list_adapters
from .model import Model, ModelStats
from .settings import (
    clear_cache_dir,
    get_cache_dir,
    get_global_cache,
    get_local_cache,
)
from .validate import normalize_inputs, summarize

__all__ = [
    "Adapter",
    "Backend",
    "Model",
    "ModelStats",
    "__version__",
    "version_tuple__",
    "apply",
    "build",
    "build_groups",
    "clear_cache_dir",
    "crop",
    "check_ranges",
    "detect_scenes",
    "apply_groups",
    "match_groups",
    "SceneGroup",
    "get_cache_dir",
    "get_global_cache",
    "get_local_cache",
    "list_adapters",
    "load",
    "load_plugin",
    "match",
    "normalize_inputs",
    "summarize",
    "trim",
]
