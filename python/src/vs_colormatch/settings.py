"""Cache and storage paths, aligned with the vs-jetpack `.vsjet` convention.

The layout mirrors ``vsscale/mlrt/settings.py`` so that a project already using vs-jetpack finds our
artifacts in the place it expects::

    <script dir>/.vsjet/vs_colormatch/          # local, the default
    <platformdirs user cache>/vs_colormatch/    # global

Precedence, highest first:

1. an explicit ``cache_dir=`` argument
2. ``$VSCOLORMATCH_CACHE_DIR``
3. ``[tool.vs_colormatch] cache_dir = ...`` in ``pyproject.toml`` / ``vsjet.toml``
4. the local ``.vsjet`` package storage
5. the global cache, when the local one has no matching file and fallback is enabled

Global selection follows vs-jetpack: ``$VSCOLORMATCH_GLOBAL`` or ``[tool.vs_colormatch] global``.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Sequence
from contextlib import suppress
from functools import cache
from pathlib import Path

__all__ = [
    "APP_AUTHOR",
    "APP_NAME",
    "clear_cache_dir",
    "get_cache_dir",
    "get_global_cache",
    "get_local_cache",
    "get_toml_config",
    "is_fallback_enabled",
    "model_path",
]

APP_AUTHOR = "vsjet"
APP_NAME = "vs_colormatch"

TRUTHY = frozenset({"1", "true", "yes", "on"})
FALSY = frozenset({"0", "false", "no", "off"})

ENV_CACHE_DIR = "VSCOLORMATCH_CACHE_DIR"
ENV_GLOBAL = ("VSCOLORMATCH_GLOBAL",)

TOML_FILES = ("vsjet.toml", "pyproject.toml")
TOML_KEYS: tuple[tuple[str, ...], ...] = (("vs_colormatch",), ("tool", "vs_colormatch"))


@cache
def get_toml_config() -> dict:
    """Reads ``[vs_colormatch]`` from ``vsjet.toml`` or ``[tool.vs_colormatch]`` from ``pyproject.toml``.

    The search starts in the current directory, matching vs-jetpack's behaviour.
    """
    for name, keys in zip(TOML_FILES, TOML_KEYS):
        path = Path(name).expanduser().resolve()
        if not path.exists():
            continue
        with suppress(OSError, tomllib.TOMLDecodeError), path.open("rb") as handle:
            config = tomllib.load(handle)
        for key in keys:
            if not isinstance(config, dict) or key not in config:
                config = {}
                break
            config = config[key]
        if config:
            return config
    return {}


@cache
def get_global_cache() -> Path:
    """The per-user cache directory, as ``platformdirs`` resolves it for this platform."""
    import platformdirs

    return Path(platformdirs.user_cache_path(APP_NAME, APP_AUTHOR))


@cache
def get_local_cache() -> Path:
    """``<script dir>/.vsjet/vs_colormatch``, using ``vstools`` when vs-jetpack is installed."""
    try:
        from vstools import PackageStorage

        return Path(PackageStorage(package_name=APP_NAME).folder)
    except Exception:
        # vs-jetpack is optional. Reimplement the three lines it takes so that the on-disk layout is
        # byte-for-byte the same either way.
        try:
            from jetpytools import get_script_path

            base = Path(get_script_path()).parent
        except Exception:
            base = Path.cwd()
        return base / ".vsjet" / APP_NAME


def _env_truthy(names: Sequence[str]) -> bool:
    return any(os.getenv(name, "").lower() in TRUTHY for name in names)


def _env_falsy(names: Sequence[str]) -> bool:
    return any(os.getenv(name, "").lower() in FALSY for name in names)


def is_global_enabled() -> bool:
    """Whether the global cache should be used instead of the local one."""
    if _env_truthy(ENV_GLOBAL):
        return True
    if _env_falsy(ENV_GLOBAL):
        return False
    return bool(get_toml_config().get("global", False))


def is_fallback_enabled() -> bool:
    """Whether a miss in the local cache should fall back to the global cache."""
    for name in ("VSCOLORMATCH_FALLBACK",):
        if os.getenv(name):
            return os.getenv(name, "").lower() not in FALSY
    config = get_toml_config()
    if "fallback" in config:
        return bool(config["fallback"])
    return True


def get_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolves the cache directory.

    Args:
        cache_dir: Explicit override. Wins over everything else.

    Returns:
        The directory to read and write model files in. It is not created here.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    if env := os.getenv(ENV_CACHE_DIR):
        return Path(env).expanduser().resolve()
    if configured := get_toml_config().get("cache_dir"):
        return Path(configured).expanduser().resolve()
    if is_global_enabled():
        return get_global_cache()
    return get_local_cache()


#: Identifies the *algorithm*, not the file format.
#:
#: Bump this whenever a change alters the model a given set of arguments produces — a different fit,
#: a different default, a different resampling. The `.cm` format version covers the byte layout and is
#: not a substitute: a stale cache entry is still perfectly parseable, just wrong.
#:
#: History: `0.2.0` changed the gamut default, the grid extrapolation, the CDF inversion, added CDF
#: smoothing, and moved chroma resampling into the plugin. `0.3.0` added the GPU backend but produces
#: the same model from the same arguments, so the tag does not move.
ALGORITHM_TAG = "v2-0.2.0"


def config_key(parts: Sequence[str | bytes | int | float]) -> str:
    """Builds a stable 16-hex-character cache key from arbitrary provenance parts.

    Deterministic so that re-running the same build reuses the same file, and specific enough that a
    changed model parameter or a changed source segment produces a different one.
    """
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        if isinstance(part, bytes):
            digest.update(part)
        else:
            digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def model_path(
    key: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    fallback: bool | None = None,
) -> Path:
    """Returns the path for a cached model, preferring an existing local file.

    Args:
        key: A key from :func:`config_key`.
        cache_dir: Explicit cache directory override.
        fallback: Override the global-fallback policy.

    Returns:
        A path that may or may not exist; callers decide whether to read or write.
    """
    directory = get_cache_dir(cache_dir)
    local = directory / f"{key}.cm"
    if local.exists():
        return local

    use_fallback = is_fallback_enabled() if fallback is None else fallback
    if use_fallback:
        global_dir = get_global_cache() if cache_dir is None else get_cache_dir(None)
        if global_dir != directory:
            candidate = global_dir / f"{key}.cm"
            if candidate.exists():
                return candidate
    return local


def clear_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> int:
    """Deletes every ``.cm`` file in the cache directory. Returns how many were removed."""
    directory = get_cache_dir(cache_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.cm"):
        path.unlink()
        removed += 1
    return removed
