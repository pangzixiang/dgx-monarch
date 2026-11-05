"""Pure validation for per-render model-sampling overrides."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypedDict


class ModelSamplingWire(TypedDict):
    kind: Literal["sd3"]
    shift: float


class MinimaxH3AVWire(TypedDict):
    kind: Literal["minimax_h3_av"]
    shift_video: float
    shift_audio: float


@dataclass(frozen=True)
class ModelSamplingSpec:
    """Canonical immutable graph-side representation of an sd3 sampling patch."""

    kind: Literal["sd3"]
    shift: float

    def to_wire(self) -> ModelSamplingWire:
        return {"kind": self.kind, "shift": self.shift}


@dataclass(frozen=True)
class MinimaxH3AVSamplingSpec:
    """Canonical immutable graph-side representation of a MiniMax H3 AV patch.

    ``shift_video`` drives the sampler's video sigma schedule
    (``ModelSamplingAV``); both values are handed to the DiT, which inverts the
    video schedule to the shared base grid and derives the audio schedule from
    it — the exact semantics of the stock ``MiniMaxH3SigmaShift`` node.
    """

    kind: Literal["minimax_h3_av"]
    shift_video: float
    shift_audio: float

    def to_wire(self) -> MinimaxH3AVWire:
        return {
            "kind": self.kind,
            "shift_video": self.shift_video,
            "shift_audio": self.shift_audio,
        }


def _validate_shift(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"model_sampling {name} must be a finite number in (0,100]"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized <= 100.0:
        raise ValueError(
            f"model_sampling {name} must be a finite number in (0,100]"
        )
    return normalized


def normalize_model_sampling(
    value: object | None,
) -> ModelSamplingWire | MinimaxH3AVWire | None:
    """Validate and copy a model-sampling wire value.

    The exact-key check keeps the protocol fail-closed as new sampling variants
    arrive. JSON integers become floats; booleans are refused although ``bool``
    subclasses ``int``. Both AV shifts must be strictly positive: the DiT
    inverts the video schedule and derives the audio schedule from it, so zero
    would divide by zero inside the model.
    """
    if value is None:
        return None
    if isinstance(value, (ModelSamplingSpec, MinimaxH3AVSamplingSpec)):
        value = value.to_wire()
    if not isinstance(value, Mapping):
        raise ValueError("model_sampling must be a mapping or null")
    kind = value.get("kind")
    if kind == "sd3":
        if set(value) != {"kind", "shift"}:
            raise ValueError("model_sampling must contain exactly 'kind' and 'shift'")
        shift = value["shift"]
        if isinstance(shift, bool) or not isinstance(shift, Real):
            raise ValueError("model_sampling shift must be a finite number in 0..100")
        normalized_shift = float(shift)
        if not math.isfinite(normalized_shift) or not 0.0 <= normalized_shift <= 100.0:
            raise ValueError("model_sampling shift must be a finite number in 0..100")
        return {"kind": "sd3", "shift": normalized_shift}
    if kind == "minimax_h3_av":
        if set(value) != {"kind", "shift_video", "shift_audio"}:
            raise ValueError(
                "minimax_h3_av must contain exactly 'kind', 'shift_video' and 'shift_audio'"
            )
        return {
            "kind": "minimax_h3_av",
            "shift_video": _validate_shift(value["shift_video"], "shift_video"),
            "shift_audio": _validate_shift(value["shift_audio"], "shift_audio"),
        }
    raise ValueError("model_sampling kind must be 'sd3' or 'minimax_h3_av'")


def freeze_model_sampling(
    value: object | None,
) -> ModelSamplingSpec | MinimaxH3AVSamplingSpec | None:
    """Return the immutable graph-side form of a validated wire value."""
    normalized = normalize_model_sampling(value)
    if normalized is None:
        return None
    if normalized["kind"] == "sd3":
        return ModelSamplingSpec(kind="sd3", shift=normalized["shift"])
    return MinimaxH3AVSamplingSpec(
        kind="minimax_h3_av",
        shift_video=normalized["shift_video"],
        shift_audio=normalized["shift_audio"],
    )
