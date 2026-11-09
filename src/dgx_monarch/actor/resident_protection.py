"""Keep peer slots resident through ComfyUI's public memory-manager calls.

Worker GPU endpoints execute serially. A context-local keep list is active only
while loading or sampling one slot, so explicit store-wide unload still works.
No weights, capacity prices, or ComfyUI implementations are replaced here.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from .. import resident_slots

# GPU-call lifetime; reset on every exit, including cancellation and failure.
_PROTECTED: ContextVar[frozenset[int]] = ContextVar("dgxm_resident_peers", default=frozenset())


def is_protected(stored: Any) -> bool:
    return any(id(getattr(p, "model", None)) in _PROTECTED.get()
               for p in (getattr(stored, "base_patcher", None),
                         getattr(stored, "active_patcher", None)))


def _install(mm: Any) -> None:
    original = mm.free_memory
    if getattr(original, "_dgxm_resident_slots", False):
        return

    @wraps(original)
    def free_memory(memory_required: Any, device: Any, *args: Any, **kwargs: Any) -> Any:
        identities = _PROTECTED.get()
        if identities:
            keep = list(args[0] if args else kwargs.get("keep_loaded", ()))
            for entry in list(mm.current_loaded_models):
                patcher = getattr(entry, "model", None)
                if id(getattr(patcher, "model", None)) in identities and entry not in keep:
                    keep.append(entry)
            if args:
                args = (keep, *args[1:])
            else:
                kwargs["keep_loaded"] = keep
        return original(memory_required, device, *args, **kwargs)

    free_memory._dgxm_resident_slots = True  # type: ignore[attr-defined]
    mm.free_memory = free_memory


@contextmanager
def peers(store: Any, slot: str) -> Iterator[None]:
    resident_slots.attribute(slot)
    if slot != "refiner" and resident_slots.held(store, "refiner") is None:
        yield
        return
    import comfy.model_management as mm

    _install(mm)
    identities = frozenset(
        id(patcher.model)
        for name, stored in resident_slots.items(store) if name != slot and stored is not None
        for patcher in (stored.base_patcher, stored.active_patcher)
        if patcher is not None and getattr(patcher, "model", None) is not None)
    token = _PROTECTED.set(_PROTECTED.get() | identities)
    try:
        yield
    finally:
        _PROTECTED.reset(token)


def bind_attention_for_sample(worker: Any, resident: Any) -> None:
    """A reused primary must not inherit the last loaded refiner's family binding."""
    if resident_slots.held(worker.store, "refiner") is None:
        return
    from ..adapters import adapter_for, attention_capability, sol_attention
    from ..family_select import override_for_worker

    model = resident.base_patcher.model
    adapter = adapter_for(model, override_for_worker(worker))
    family = getattr(adapter, "family", "unknown")
    head_dim = getattr(adapter, "attention_head_dim", None)
    worker._attn.bind_capability(family, head_dim)
    attention = getattr(adapter, "attention_dispatch", lambda value: value)(worker._attn)
    sol_attention.bind_sol_scope(attention, family)
    attention_capability.assert_kernel_capability(worker._attn.effective_kernel, family, head_dim, model)
