"""Test-suite configuration.

Two jobs, both about not leaving things behind.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _keep_the_cache_out_of_the_source_tree(tmp_path_factory):
    """Points the model cache at a temporary directory for the whole session.

    The local cache is resolved relative to the *script* directory, which under pytest is the test
    module's directory — so `vscm.match(...)`, whose `cache` defaults to `True`, wrote `.cm` files
    into `tests/python/.vsjet/`. Nothing broke, because `*.cm` is git-ignored, but a test run should
    not modify the source tree, and a contributor running the suite would find unexplained files
    appear.

    Tests that assert on the resolution order set and unset the variable themselves; those that read
    `get_local_cache()` are unaffected because that is the raw location, not the effective one.
    """
    cache = tmp_path_factory.mktemp("vs_colormatch-cache")
    previous = os.environ.get("VSCOLORMATCH_CACHE_DIR")
    os.environ["VSCOLORMATCH_CACHE_DIR"] = str(cache)
    yield
    if previous is None:
        os.environ.pop("VSCOLORMATCH_CACHE_DIR", None)
    else:
        os.environ["VSCOLORMATCH_CACHE_DIR"] = previous
