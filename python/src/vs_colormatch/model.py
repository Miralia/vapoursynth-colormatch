"""The model object: a built colour transform plus its provenance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import vapoursynth as vs

from . import settings

__all__ = ["Model", "ModelStats"]

core = vs.core


@dataclass(frozen=True)
class ModelStats:
    """Build statistics, mirrored from the ``_CM_Stats`` frame property."""

    samples: int
    frames: int
    w1_before: float
    w1_after: float
    max_gain: float
    build_ms: float
    directions: int

    @classmethod
    def from_props(cls, values) -> ModelStats:
        v = list(values)
        return cls(
            samples=int(v[0]),
            frames=int(v[1]),
            w1_before=float(v[2]),
            w1_after=float(v[3]),
            max_gain=float(v[4]),
            build_ms=float(v[5]),
            directions=int(v[6]),
        )

    @property
    def reduction(self) -> float:
        """Fraction of the sliced-Wasserstein distance removed, in ``[0, 1]``."""
        if self.w1_before <= 0:
            return 0.0
        return 1.0 - self.w1_after / self.w1_before

    def __str__(self) -> str:
        return (
            f"{self.directions} directions, {self.samples} samples from {self.frames} frames; "
            f"sliced-W1 {self.w1_before:.5f} -> {self.w1_after:.5f} "
            f"({self.reduction * 100:.1f}% removed); max gain {self.max_gain:.2f}; "
            f"built in {self.build_ms:.0f} ms"
        )


@dataclass
class Model:
    """A colour transform that can be applied to any clip.

    The heavy lifting lives in the plugin; this object owns the one-frame model clip that carries the
    serialized model, plus the parameters it was built with so that caching and re-export are
    reproducible.
    """

    clip: vs.VideoNode
    """The 1x1 GRAY8 single-frame clip holding the serialized model."""

    stats: ModelStats
    """Build statistics."""

    params: dict = field(default_factory=dict)
    """The arguments the model was built with, used to derive its cache key."""

    path: Path | None = None
    """Where the model was read from or written to, if anywhere."""

    def __repr__(self) -> str:
        return f"<Model {self.stats.directions} dirs, w1_after={self.stats.w1_after:.5f}>"

    def apply(
        self,
        clip: vs.VideoNode,
        *,
        backend: str | None = None,
        device_id: int | None = None,
        method: str | None = None,
        lut_size: int | None = None,
        planes: list[int] | None = None,
        gamut: str | None = None,
        luma_lock: float | None = None,
        matrix: int | None = None,
        range: int | None = None,
    ) -> vs.VideoNode:
        """Applies the model. The result always matches the input in size, format and length."""
        from .api import apply

        return apply(
            clip,
            self,
            backend=backend,
            device_id=device_id,
            method=method,
            lut_size=lut_size,
            planes=planes,
            gamut=gamut,
            luma_lock=luma_lock,
            matrix=matrix,
            range=range,
        )

    def save(self, path: str | os.PathLike[str], *, cache_dir: str | os.PathLike[str] | None = None) -> Path:
        """Writes the model as a ``.cm`` file.

        Relative paths resolve against the cache directory, so ``save("look.cm")`` lands next to the
        other cached models rather than in the current working directory.
        """
        target = Path(path)
        if not target.is_absolute():
            target = settings.get_cache_dir(cache_dir) / target
        target.parent.mkdir(parents=True, exist_ok=True)
        data = bytes(self.clip.get_frame(0).props["_CM_Model"])
        target.write_bytes(data)
        self.path = target
        return target

    def export_lut(
        self,
        path: str | os.PathLike[str],
        *,
        size: int = 33,
        format: str = "cube",
    ) -> Path:
        """Bakes the model into a 3D LUT and writes it.

        Note that a baked LUT is an approximation: measured on a cross-channel look, a 33^3
        tetrahedral bake reproduces the exact transform to within ~3 LSB at 16-bit. Prefer the native
        method for delivery and export a LUT for interchange.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        blank = core.std.BlankClip(width=1, height=1, format=vs.GRAY8, length=1)
        core.cm.ExportLUT(clip=blank, model=self.clip, path=str(target), size=size, format=format).get_frame(0)
        return target
