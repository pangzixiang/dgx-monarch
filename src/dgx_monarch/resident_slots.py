"""Named model slots; the refiner is independent of dual-model CFG."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

ATTRIBUTES = {"cond": "current", "uncond": "uncond", "refiner": "refiner"}


def attribute(slot: str) -> str:
    """Resolve a checked slot without silently aliasing unknown names to cond."""
    if slot not in ATTRIBUTES:
        raise ValueError(f"unknown resident model slot {slot!r}")
    return ATTRIBUTES[slot]


def held(store: Any, slot: str) -> Any:
    return getattr(store, attribute(slot), None)


def items(store: Any) -> Iterator[tuple[str, Any]]:
    """Keep the legacy empty snapshot shape until a refiner is held."""
    for slot in ATTRIBUTES:
        value = held(store, slot)
        if slot != "refiner" or value is not None:
            yield slot, value


def peers(store: Any, slot: str) -> tuple[Any, ...]:
    """Return the live residents a single-slot operation leaves in place."""
    return tuple(value for name, value in items(store) if name != slot and value is not None)


def primary(spec: Mapping[str, Any]) -> str:
    slot = spec.get("slot", "cond")
    if slot not in ("cond", "refiner"):
        raise ValueError(f"invalid primary model slot {slot!r}")
    return str(slot)


def validate_refiner_topology(topology: Mapping[str, Any]) -> None:
    """The first refiner contract is serial pure Ulysses, checked before loading."""
    if (any(int(topology.get(key, 1)) != 1 for key in ("cfg", "ring", "dp"))
            or topology.get("fsdp", False)):
        raise ValueError("refiner slot requires serial pure Ulysses, CFG degree one and no FSDP")
