"""Reader for the fixed header of a ``.cm`` model file.

The header is a documented, stable layout (see ``crates/colormatch-core/src/model.rs``), and it
carries the build statistics. Reading them back here is what lets a cache hit report the *real*
provenance of a model instead of fabricating zeros.

Parsing is strict and mirrored by a test that cross-checks these values against the ones the plugin
reports when it builds the same model, so the two readers cannot drift apart silently.
"""

from __future__ import annotations

import struct
from pathlib import Path

__all__ = ["HEADER_LEN", "MAGIC", "FORMAT_VERSION", "CmHeader", "read_header"]

MAGIC = b"CMAT"
FORMAT_VERSION = 1
HEADER_LEN = 64

#: ``<4sHHII4s4sIIQIf f f fI`` — little-endian, no padding.
_HEADER = struct.Struct("<4sHHII4s4sIIQIf f f fI".replace(" ", ""))

_NAMES = ("magic", "format_version", "flags", "nodes", "n_slices", "test_space", "ref_space",
          "transfer_hint", "directions", "samples", "frames", "w1_before", "w1_after",
          "max_gain", "build_ms", "reserved")


class CmHeader:
    """The parsed fixed header of a ``.cm`` file."""

    __slots__ = _NAMES

    def __init__(self, **kwargs) -> None:
        for name in _NAMES:
            setattr(self, name, kwargs[name])

    @property
    def serialized_len(self) -> int:
        """Total size of the file this header belongs to, in bytes."""
        return HEADER_LEN + self.n_slices * (9 * 4 + 3 * (2 * 4 + self.nodes * 4))

    def __repr__(self) -> str:
        return (
            f"<CmHeader v{self.format_version} K={self.n_slices} nodes={self.nodes} "
            f"w1 {self.w1_before:.6f}->{self.w1_after:.6f}>"
        )


def read_header(path: str | Path) -> CmHeader:
    """Reads and validates the header of a ``.cm`` file.

    Args:
        path: The model file.

    Returns:
        The parsed header.

    Raises:
        ValueError: If the magic, format version, flags or reserved fields are wrong, or if the file
            is shorter than the header claims.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER_LEN:
        raise ValueError(
            f"vs_colormatch: {path} is {len(raw)} bytes, shorter than the {HEADER_LEN}-byte header"
        )

    values = _HEADER.unpack_from(raw, 0)
    header = CmHeader(**dict(zip(_NAMES, values)))

    if header.magic != MAGIC:
        raise ValueError(f"vs_colormatch: {path} is not a .cm model (magic {header.magic!r})")
    if header.format_version != FORMAT_VERSION:
        raise ValueError(
            f"vs_colormatch: {path} has format version {header.format_version}, "
            f"this build supports {FORMAT_VERSION}"
        )
    if header.flags != 0 or header.reserved != 0 or header.transfer_hint != 0:
        raise ValueError(f"vs_colormatch: {path} has non-zero reserved fields")
    if header.serialized_len != len(raw):
        raise ValueError(
            f"vs_colormatch: {path} is {len(raw)} bytes but the header describes "
            f"{header.serialized_len} bytes for K={header.n_slices}, nodes={header.nodes}"
        )
    return header
