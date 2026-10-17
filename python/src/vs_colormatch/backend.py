"""Backend selection.

The user-facing shape follows ``vs-jetpack``'s ``vsscale.mlrt.Backend``: a frozen dataclass with an
``autoselect`` classmethod and named constructors, so existing muscle memory transfers. The
*mechanism* differs, and deliberately so:

* vs-jetpack dispatches to several third-party plugins and probes them with ``hasattr(core, ...)``,
  learning the GPU vendor from OpenCL via ``pyopencl``.
* Our backends are compiled into this plugin, so the adapter list comes from ``cm.Devices`` — the
  same code that would run the compute — and the wrapper needs no GPU dependency of its own.

What both share is the important property: the backend that *actually ran* is reported back on the
output frames as ``_CM_Backend``, so a fallback is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import vapoursynth as vs

__all__ = ["Adapter", "Backend", "list_adapters"]

core = vs.core


@dataclass(frozen=True, kw_only=True)
class Adapter:
    """A GPU adapter reported by the plugin."""

    index: int
    name: str
    api: str
    device_type: str
    vendor: int
    device: int

    def __str__(self) -> str:
        return f"{self.name} ({self.api}, {self.device_type})"


@dataclass(frozen=True, kw_only=True)
class Backend:
    """Which implementation should run the transform.

    Args:
        name: ``auto``, ``cpu``, ``simd``, ``gpu`` or ``lut``.
        device_id: GPU index, ignored by the CPU backends.
        lut_size: LUT edge length, only meaningful for ``lut``.
    """

    name: str = "auto"
    device_id: int = 0
    lut_size: int = 33

    #: Names accepted by the plugin, in preference order for ``auto``.
    ORDER: ClassVar[tuple[str, ...]] = ("gpu", "simd", "cpu")

    #: Aliases, matching `BackendPreference::parse` on the plugin side.
    #:
    #: The two layers have to agree on this. The plugin accepted `scalar` and `wgpu` while the
    #: wrapper rejected them, so a name that worked when passed to `cm.Apply` directly failed through
    #: `Backend(name=...)` — a confusing inconsistency for no reason.
    ALIASES: ClassVar[dict[str, str]] = {
        "auto": "auto",
        "cpu": "cpu",
        "scalar": "cpu",
        "simd": "simd",
        "gpu": "gpu",
        "wgpu": "gpu",
        "lut": "lut",
    }

    #: LUT edge lengths the plugin supports.
    LUT_SIZES: ClassVar[tuple[int, ...]] = (17, 33, 65)

    def __post_init__(self) -> None:
        # Normalised the same way the plugin normalises it, so `" GPU "` and `"gpu"` are one thing.
        resolved = self.ALIASES.get(str(self.name).strip().lower())
        if resolved is None:
            raise ValueError(
                f"vs_colormatch: backend name must be one of auto/cpu/simd/gpu/lut "
                f"(aliases: scalar = cpu, wgpu = gpu), got {self.name!r}"
            )
        object.__setattr__(self, "name", resolved)
        if self.device_id < 0:
            raise ValueError(f"vs_colormatch: device_id must not be negative, got {self.device_id}")
        if resolved == "lut" and self.lut_size not in self.LUT_SIZES:
            # Checked here rather than left to the plugin: `Backend.LUT(32)` looked valid and then
            # failed at filter-creation time, which is a worse place to find out.
            raise ValueError(
                f"vs_colormatch: lut_size must be one of {list(self.LUT_SIZES)}, got {self.lut_size}"
            )

    def to_kwargs(self) -> dict[str, Any]:
        """The keyword arguments to pass to ``cm.Apply``."""
        return {
            "backend": "lut" if self.name == "lut" else self.name,
            "device_id": self.device_id,
            "method": "lut" if self.name == "lut" else "native",
            "lut_size": self.lut_size,
        }

    def __str__(self) -> str:
        return self.name if self.name != "gpu" else f"gpu:{self.device_id}"

    # Named constructors, matching the vs-jetpack ergonomics.
    @classmethod
    def CPU(cls) -> Backend:
        """Portable scalar code. Always available."""
        return cls(name="cpu")

    @classmethod
    def SIMD(cls) -> Backend:
        """CPU SIMD. The best CPU option and the default when no GPU backend is built."""
        return cls(name="simd")

    @classmethod
    def GPU(cls, device_id: int = 0) -> Backend:
        """GPU compute. Falls back to SIMD when unavailable, and says so via ``_CM_Backend``."""
        return cls(name="gpu", device_id=device_id)

    @classmethod
    def LUT(cls, lut_size: int = 33) -> Backend:
        """Bakes the model to a 3D LUT. Cost is independent of the slice count."""
        return cls(name="lut", lut_size=lut_size)

    @classmethod
    def autoselect(cls, device_id: int = 0, **kwargs: Any) -> Backend:
        """Picks the fastest backend that is actually available.

        Args:
            device_id: Which GPU to prefer, when there is a choice.
            **kwargs: Forwarded to the constructor.

        Returns:
            A backend. GPU is only chosen when the plugin reports that its GPU *apply* path exists —
            adapter enumeration landing before the compute pipelines must not make ``auto`` promise
            something it cannot deliver.
        """
        devices = _devices()
        if devices.get("gpu_apply") and devices.get("adapters"):
            return cls(name="gpu", device_id=device_id, **kwargs)
        return cls(name="simd", device_id=device_id, **kwargs)


def _devices() -> dict[str, Any]:
    """Reads the plugin's adapter report, tolerating a missing ``cm.Devices``."""
    if not hasattr(core, "cm"):
        # `autoselect` is often the first call in a script, before anything touched `core.cm`.
        try:
            from .api import load_plugin

            load_plugin()
        except Exception:
            return {"adapters": [], "gpu_enabled": False, "gpu_apply": False, "backends": []}
    if not hasattr(core, "cm"):
        return {"adapters": [], "gpu_enabled": False, "gpu_apply": False, "backends": []}
    frame = core.cm.Devices().get_frame(0)
    props = frame.props
    count = int(props.get("_CM_AdapterCount", 0))
    adapters = [
        Adapter(
            index=i,
            name=props[f"_CM_Adapter{i}_Name"].decode(),
            api=props[f"_CM_Adapter{i}_Api"].decode(),
            device_type=props[f"_CM_Adapter{i}_Type"].decode(),
            vendor=int(props[f"_CM_Adapter{i}_Vendor"]),
            device=int(props[f"_CM_Adapter{i}_Device"]),
        )
        for i in range(count)
    ]
    return {
        "adapters": adapters,
        "gpu_enabled": bool(props.get("_CM_GpuEnabled", 0)),
        "gpu_apply": bool(props.get("_CM_GpuApply", 0)),
        "backends": props.get("_CM_Backends", b"").decode().split(","),
        "version": props.get("_CM_Version", b"").decode(),
    }


def list_adapters() -> list[Adapter]:
    """Returns the GPU adapters the plugin can see, best first."""
    return list(_devices()["adapters"])
