"""Strict input validation and normalisation.

The plugin refuses to guess: it will not invent a matrix for YUV, will not silently resize, and will
not accept subsampled planes. Those refusals are correct but unhelpful on their own, so this module
does the normalisation the caller almost always wants, *and says what it did*.

Two rules:

1. Anything that changes the pixels is reported, never silent.
2. Anything that would change the pixels in a way we cannot justify is an error, not a warning.

Validation style follows the ecosystem convention (`vs_colorfix`): plain ``TypeError`` /
``ValueError`` with the function name prefixed, so failures are actionable from the message alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import vapoursynth as vs

__all__ = ["Normalization", "NormalizedInput", "normalize_inputs", "resolve_matrix", "resolve_range"]

core = vs.core

#: VapourSynth matrix coefficient codes we are willing to interpret.
SUPPORTED_MATRIX = {0, 1, 4, 5, 6, 7, 8, 9}

MATRIX_NAMES = {
    0: "rgb",
    1: "bt709",
    2: "unspec",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
}


@dataclass
class Normalization:
    """One change the wrapper made to an input clip, for reporting."""

    clip: int
    what: str
    detail: str

    def __str__(self) -> str:
        return f"clip {self.clip}: {self.what} ({self.detail})"


@dataclass
class NormalizedInput:
    """A clip ready to hand to the plugin, plus how it got there."""

    clip: vs.VideoNode
    #: Resolved matrix coefficient, or ``None`` for RGB/GRAY.
    matrix: int | None
    #: Resolved colour range: ``0`` limited, ``1`` full.
    range: int
    notes: list[str] = field(default_factory=list)


def _frame_props(clip: vs.VideoNode, index: int):
    """Fetches frame 0's properties, reporting an evaluation failure as itself.

    Swallowing this used to turn "the clip cannot be evaluated" into "the clip has no `_Matrix`",
    which sent the caller looking in entirely the wrong place. A frame that will not evaluate is a
    hard error and says so.
    """
    try:
        return clip.get_frame(0).props
    except Exception as exc:  # noqa: BLE001 - the point is to re-raise with context
        raise ValueError(
            f"vs_colormatch: clip {index} could not be evaluated to read its frame properties: {exc}"
        ) from exc


def resolve_matrix(clip: vs.VideoNode, matrix: int | None, index: int, strict: bool) -> int | None:
    """Resolves the matrix coefficient for a clip.

    Order: explicit argument, then the ``_Matrix`` frame property, then RGB/GRAY inference, then an
    error under ``strict``.
    """
    if matrix is not None:
        if matrix not in SUPPORTED_MATRIX:
            raise ValueError(
                f"vs_colormatch: matrix {matrix} ({MATRIX_NAMES.get(matrix, 'unknown')}) is not "
                f"supported; use one of {sorted(SUPPORTED_MATRIX)}"
            )
        return matrix

    family = clip.format.color_family
    if family in (vs.RGB, vs.GRAY):
        return 0

    tagged = int(_frame_props(clip, index).get("_Matrix", -1))

    if tagged in SUPPORTED_MATRIX:
        return tagged

    if strict:
        raise ValueError(
            f"vs_colormatch: clip {index} is YUV but has no usable `_Matrix` property "
            f"(got {tagged}). Pass `matrix=` explicitly, or tag the clip with "
            f"`core.std.SetFrameProps(clip, _Matrix=1)`. The values are 1=bt709, 5=bt470bg, "
            f"6=smpte170m, 9=bt2020nc."
        )
    return None


def resolve_range(clip: vs.VideoNode, rng: int | None, index: int, strict: bool) -> int:
    """Resolves the colour range: explicit argument, then ``_Range``, then the family default.

    Unlike the matrix, a missing ``_Range`` is *not* an error. VapourSynth documents the default
    (full range for RGB/GRAY, limited for YUV), so falling back to it is following a convention
    rather than guessing. A wrong matrix silently degrades the match; a wrong range is a 7% level
    error that shows up immediately in the output.
    """
    del strict  # kept in the signature so both resolvers read the same way
    if rng is not None:
        if rng not in (0, 1):
            raise ValueError(f"vs_colormatch: range must be 0 (limited) or 1 (full), got {rng}")
        return rng
    tagged = int(_frame_props(clip, index).get("_Range", -1))
    if tagged in (0, 1):
        return tagged
    return 1 if clip.format.color_family in (vs.RGB, vs.GRAY) else 0


def normalize_inputs(
    clips: list[vs.VideoNode],
    *,
    matrix: int | None = None,
    range: int | None = None,
    strict: bool = True,
    convert_subsampled: bool = False,
    resizer: str = "Bicubic",
) -> tuple[list[NormalizedInput], list[Normalization]]:
    """Validates a list of input clips.

    Args:
        clips: The clips to prepare.
        matrix: Matrix coefficient to force on every clip.
        range: Colour range to force on every clip.
        strict: Raise instead of guessing when the colour space is ambiguous.
        convert_subsampled: Upsample 4:2:0/4:2:2 to 4:4:4. Off by default: the plugin resamples
            chroma internally so that the output keeps the input's format, and pre-converting would
            change the clip and cost an extra pass for no benefit. Turn it on only when handing the
            clip to something that needs full-resolution planes.
        resizer: Resizer used for the optional chroma conversion.

    Returns:
        ``(normalized, changes)`` — the prepared clips and a list of what was changed.

    Raises:
        TypeError: If an entry is not a video node.
        ValueError: If a clip has a variable format or resolution, or if ``strict`` and the colour
            space cannot be resolved.
    """
    if not clips:
        raise ValueError("vs_colormatch: at least one clip is required")

    changes: list[Normalization] = []
    out: list[NormalizedInput] = []

    for i, clip in enumerate(clips):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f"vs_colormatch: clip {i} must be a vapoursynth clip, got {type(clip)!r}")
        if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
            raise ValueError(
                f"vs_colormatch: clip {i} must have a constant format and dimensions; "
                f"a variable-resolution or variable-format clip cannot be matched"
            )
        if clip.num_frames == 0:
            raise ValueError(f"vs_colormatch: clip {i} has no frames")

        notes: list[str] = []
        working = clip

        fmt = working.format
        if fmt.color_family == vs.YUV and (fmt.subsampling_w or fmt.subsampling_h):
            sub = f"{fmt.subsampling_w}x{fmt.subsampling_h}"
            if convert_subsampled:
                resizer_fn = getattr(core.resize, resizer, None) or core.resize.Bicubic
                working = resizer_fn(
                    working,
                    format=working.format.replace(
                        subsampling_w=0, subsampling_h=0, color_family=vs.YUV
                    ),
                )
                changes.append(
                    Normalization(i, "converted YUV to 4:4:4", f"{sub} -> 0x0 via resize.{resizer}")
                )
            else:
                notes.append(
                    f"YUV {sub}: chroma is resampled inside the plugin so the output keeps this format"
                )

        m = resolve_matrix(working, matrix, i, strict)
        r = resolve_range(working, range, i, strict)

        if working.format.bits_per_sample <= 8:
            notes.append(
                f"{working.format.bits_per_sample}-bit input; 10-bit or higher avoids banding"
            )

        out.append(NormalizedInput(clip=working, matrix=m, range=r, notes=notes))

    return out, changes


def summarize(changes: list[Normalization], inputs: list[NormalizedInput]) -> str:
    """Renders a human-readable report of what normalisation did."""
    lines = [f"vs_colormatch: {len(inputs)} input clip(s)"]
    for i, item in enumerate(inputs):
        fmt = item.clip.format
        space = (
            f"{fmt.name} matrix={MATRIX_NAMES.get(item.matrix or 0, 'rgb')} range={item.range}"
        )
        lines.append(f"  clip {i}: {item.clip.width}x{item.clip.height} {fmt.num_planes}p {space}")
        for note in item.notes:
            lines.append(f"    note: {note}")
    for change in changes:
        lines.append(f"  changed: {change}")
    if not changes:
        lines.append("  changed: nothing")
    return "\n".join(lines)
