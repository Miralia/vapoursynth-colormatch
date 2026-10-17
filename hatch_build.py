"""Hatchling build hook that compiles the Rust plugin and stages it into the wheel.

The goal is that ``pip install vs-colormatch`` is the *only* step: no separate plugin download, no
manual copy into a plugin directory. VapourSynth scans ``vapoursynth/plugins/`` inside site-packages
and auto-loads anything it finds there (this is how the ``vapoursynth-*`` wheels work), so the wheel
places the shared library at::

    vapoursynth/plugins/colormatch/libcolormatch.dylib   # .so on Linux, colormatch.dll on Windows
    vapoursynth/plugins/colormatch/manifest.vs

Cross-compilation is the caller's business: set ``CARGO_BUILD_TARGET`` and this hook passes it
through, which is how the per-platform wheels are produced.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

REPO_ROOT = Path(__file__).resolve().parent
STAGING = REPO_ROOT / "python" / "build" / "wheel" / "vapoursynth" / "plugins" / "colormatch"

LIB_NAMES = {
    "darwin": "libcolormatch.dylib",
    "linux": "libcolormatch.so",
    "win32": "colormatch.dll",
}

# Rust target triple -> wheel platform tag. The tag has to follow the *target*, not the host, because
# the per-platform wheels are produced by cross-compiling with CARGO_BUILD_TARGET.
TARGET_TAGS = {
    "aarch64-apple-darwin": "macosx_11_0_arm64",
    "x86_64-apple-darwin": "macosx_10_12_x86_64",
    "x86_64-unknown-linux-gnu": "manylinux_2_28_x86_64",
    "aarch64-unknown-linux-gnu": "manylinux_2_28_aarch64",
    "x86_64-pc-windows-msvc": "win_amd64",
    "aarch64-pc-windows-msvc": "win_arm64",
}


def _platform_tag() -> str:
    """The wheel platform tag for the target being built.

    This must not be `any`. The wheel ships a compiled shared library, so an `any` tag would let pip
    install a macOS dylib on Linux, where it cannot load -- and would do so silently, because the
    Python half of the package imports fine either way. Deriving the tag from the Rust target keeps it
    honest under cross-compilation, which is how these wheels are actually produced.
    """
    if target := os.environ.get("CARGO_BUILD_TARGET"):
        try:
            return TARGET_TAGS[target]
        except KeyError:
            raise RuntimeError(
                f"vs_colormatch: no wheel platform tag known for target {target!r}; add it to "
                f"TARGET_TAGS so the wheel is not mislabelled"
            ) from None
    # No cross-compilation: the host is the target.
    platform = sys.platform
    if platform == "darwin":
        import platform as _p

        arch = "arm64" if _p.machine() == "arm64" else "x86_64"
        return TARGET_TAGS[f"{'aarch64' if arch == 'arm64' else 'x86_64'}-apple-darwin"]
    if platform.startswith("linux"):
        import platform as _p

        machine = _p.machine()
        triple = "x86_64" if machine in ("x86_64", "AMD64") else "aarch64"
        return TARGET_TAGS[f"{triple}-unknown-linux-gnu"]
    if platform == "win32":
        import platform as _p

        return TARGET_TAGS["x86_64-pc-windows-msvc" if _p.machine() == "AMD64" else "aarch64-pc-windows-msvc"]
    raise RuntimeError(f"vs_colormatch: unsupported platform {platform!r}")


def _library_name() -> str:
    try:
        return LIB_NAMES[sys.platform]
    except KeyError:
        raise RuntimeError(f"vs_colormatch: unsupported platform {sys.platform!r}") from None


def _cargo_target_dir() -> Path:
    if env := os.environ.get("CARGO_TARGET_DIR"):
        return Path(env)
    return REPO_ROOT / "target"


class CustomBuildHook(BuildHookInterface):
    """Builds ``colormatch-vs`` and copies the artifact into the wheel tree."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return
        if os.environ.get("VSCOLORMATCH_SKIP_CARGO"):
            self.app.display_warning("VSCOLORMATCH_SKIP_CARGO set; shipping a wheel without the plugin")
            return

        self._sync_version()
        self._build_rust()
        self._stage(build_data)

        # `py3-none-any` is the default and is wrong here: the wheel carries a native library.
        tag = f"py3-none-{_platform_tag()}"
        build_data["tag"] = tag
        # A wheel containing a shared library is not pure Python, and `Root-Is-Purelib` has to agree
        # with the tag or installers can place it in the wrong scheme.
        build_data["pure_python"] = False
        self.app.display_info(f"vs_colormatch: wheel tag {tag}")

    def _sync_version(self) -> None:
        """Regenerates `_version.py` from the Cargo workspace version.

        The Rust manifest is the single source of truth; regenerating here means a wheel can never
        carry a Python version that disagrees with the plugin it ships.
        """
        script = REPO_ROOT / "python" / "gen_version.py"
        subprocess.run([sys.executable, str(script)], check=True)  # noqa: S603

    def _build_rust(self) -> None:
        target = os.environ.get("CARGO_BUILD_TARGET")
        cmd = ["cargo", "build", "--release", "-p", "colormatch-vs"]
        if target:
            cmd += ["--target", target]
        self.app.display_info(f"vs_colormatch: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)  # noqa: S603

    def _stage(self, build_data: dict) -> None:
        target = os.environ.get("CARGO_BUILD_TARGET")
        base = _cargo_target_dir() / target / "release" if target else _cargo_target_dir() / "release"
        source = base / _library_name()
        if not source.exists():
            raise FileNotFoundError(
                f"vs_colormatch: expected the built plugin at {source}. "
                f"Check that `cargo build --release -p colormatch-vs` succeeds."
            )

        STAGING.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, STAGING / source.name)
        # `manifest.vs` is the VapourSynth manifest that points at the library in the same folder.
        (STAGING / "manifest.vs").write_text(
            f"[VapourSynth Manifest V1]\n{source.stem}\n", encoding="utf-8"
        )
        # The staging tree is placed by the `force-include` entry in pyproject.toml. Do not add a
        # second mapping here: an earlier version did, with one too many `parent` calls, which put a
        # duplicate copy of the plugin under `vapoursynth/wheel/...` in every wheel.
        self.app.display_info(f"vs_colormatch: staged {source.name} into the wheel")
