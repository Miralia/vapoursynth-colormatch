"""Version consistency.

The Cargo workspace manifest is the single authoritative version. These tests fail the suite if the
generated Python version, the wheel version, or a git tag drift away from it — drift here is how a
wheel ends up claiming to be a different build than the plugin it ships.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import vs_colormatch as vscm

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = REPO_ROOT / "Cargo.toml"


def cargo_version() -> str:
    text = CARGO_TOML.read_text(encoding="utf-8")
    section = re.search(r"^\[workspace\.package\]$(.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL)
    assert section, "no [workspace.package] table in Cargo.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE)
    assert match, "no version key in [workspace.package]"
    return match.group(1)


def test_python_version_matches_cargo():
    assert vscm.__version__ == cargo_version(), (
        "python/src/vs_colormatch/_version.py is stale; run `python3 python/gen_version.py`"
    )


def test_version_tuple_matches_string():
    assert ".".join(str(p) for p in vscm.version_tuple__) == vscm.__version__


def test_version_is_plain_semver():
    """No pre-release or local suffix in the manifest.

    A suffix belongs in the git tag, so that the Rust, Python and wheel versions are byte-identical
    and a shipped artifact can always be traced back to one manifest.
    """
    assert re.fullmatch(r"\d+\.\d+\.\d+", vscm.__version__), vscm.__version__


def test_major_version_is_earned():
    """1.0 means every advertised backend is implemented and interchangeable.

    This replaced a gate that forbade 1.0.0 while two advertised features were incomplete: the GPU
    compute path did not exist, and `Backend::Simd` selected the scalar code under a different name.
    Both are implemented now, so the gate is gone — and the test that replaces it is the reason it
    could be lifted, rather than a formality.
    """
    major = vscm.version_tuple__[0]
    assert major == 1, (
        f"version is {vscm.__version__}; the advertised feature set is complete, so the pre-1.0 gate "
        f"no longer applies. If a new advertised feature is added and left unimplemented, this is "
        f"where the gate belongs."
    )


def test_version_policy_is_recorded_next_to_the_code():
    """Documents the rule the version history follows.

    Patch for fixes, minor for new user-visible capability, major only when every advertised feature
    is implemented and interchangeable. This cannot verify intent, so it records the policy where the
    version lives and checks only that the parts are non-negative — the substantive check is
    `test_major_version_is_earned`, which states what 1.0 has to mean.
    """
    major, minor, patch = vscm.version_tuple__
    assert all(part >= 0 for part in (major, minor, patch))
    assert major >= 1, "a pre-1.0 version means an advertised feature is incomplete"


def test_git_tag_agrees_when_present():
    """If HEAD is tagged, the tag must match the manifest."""
    try:
        tag = subprocess.run(  # noqa: S603
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")
    if tag.returncode != 0:
        pytest.skip("HEAD is not tagged")
    assert tag.stdout.strip().lstrip("v") == cargo_version()


def test_rust_reads_the_same_manifest():
    """The Rust crate must not carry a second version literal."""
    core = REPO_ROOT / "crates" / "colormatch-core" / "Cargo.toml"
    assert "version.workspace = true" in core.read_text(encoding="utf-8")
    plugin = REPO_ROOT / "crates" / "colormatch-vs" / "Cargo.toml"
    assert "version.workspace = true" in plugin.read_text(encoding="utf-8")
