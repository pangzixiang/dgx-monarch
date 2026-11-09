"""A finishing model must survive primary loads, swaps and sampler pressure."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import resident_slots
from dgx_monarch.actor import resident_protection
from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.nodes.common import ModelSpec


def resident() -> SimpleNamespace:
    patcher = SimpleNamespace(model=object())
    return SimpleNamespace(base_patcher=patcher, active_patcher=patcher)


@pytest.fixture
def memory_manager(monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []
    mm.calls = []

    def free_memory(required, device, keep_loaded=(), **kwargs):
        mm.calls.append((required, device, list(keep_loaded), kwargs))
        removed = [entry for entry in mm.current_loaded_models if entry not in keep_loaded]
        mm.current_loaded_models[:] = [entry for entry in mm.current_loaded_models if entry in keep_loaded]
        return removed

    mm.free_memory = free_memory
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return mm


def test_unknown_slot_never_aliases_the_primary():
    store = ModelStore()
    with pytest.raises(ValueError, match="unknown resident"):
        store._stored("refienr")
    assert store.current is store.refiner is None


def test_primary_wire_stays_compatible_and_refiner_is_explicit():
    mesh = SimpleNamespace()
    base = ModelSpec(mesh=mesh, unet_name="draft.safetensors")
    refiner = ModelSpec(mesh=mesh, unet_name="finish.safetensors", slot="refiner")
    assert "slot" not in base.request_dict()
    assert refiner.request_dict()["slot"] == "refiner"
    assert resident_slots.primary(refiner.request_dict()) == "refiner"
    with pytest.raises(ValueError, match="invalid primary"):
        resident_slots.primary({"slot": "uncond"})


@pytest.mark.parametrize("slot", ["cond", "refiner"])
def test_global_memory_free_preserves_only_the_other_slot(memory_manager, slot):
    store = ModelStore()
    store.current, store.refiner = resident(), resident()
    entries = {name: SimpleNamespace(model=value.active_patcher)
               for name, value in resident_slots.items(store) if value is not None}
    memory_manager.current_loaded_models[:] = entries.values()
    peer = "refiner" if slot == "cond" else "cond"
    with resident_protection.peers(store, slot):
        memory_manager.free_memory(10**30, "cuda")
        assert memory_manager.current_loaded_models == [entries[peer]]
        store._mark_other_slot_evicted(slot)
        assert peer not in store._gpu_evicted
    memory_manager.free_memory(10**30, "cuda")
    assert memory_manager.current_loaded_models == []


def test_refiner_first_load_protects_an_existing_primary(memory_manager):
    store = ModelStore()
    store.current = resident()
    entry = SimpleNamespace(model=store.current.active_patcher)
    memory_manager.current_loaded_models[:] = [entry]
    with resident_protection.peers(store, "refiner"):
        memory_manager.free_memory(10**30, "cuda", [], for_dynamic=False)
    assert memory_manager.current_loaded_models == [entry]
    assert memory_manager.calls[-1][3] == {"for_dynamic": False}


def test_protection_is_reset_after_interruption(memory_manager):
    store = ModelStore()
    store.current = resident()
    memory_manager.current_loaded_models[:] = [SimpleNamespace(model=store.current.active_patcher)]
    with pytest.raises(KeyboardInterrupt), resident_protection.peers(store, "refiner"):
        raise KeyboardInterrupt
    memory_manager.free_memory(10**30, "cuda")
    assert memory_manager.current_loaded_models == []


def test_legacy_store_does_not_change_the_memory_manager(memory_manager):
    original = memory_manager.free_memory
    with resident_protection.peers(ModelStore(), "cond"):
        assert memory_manager.free_memory is original


def test_snapshot_and_full_unload_include_refiner(monkeypatch):
    store = ModelStore()
    assert "refiner" not in store.snapshot()
    store.refiner = resident()
    dropped = []
    monkeypatch.setattr("dgx_monarch.actor.resident_ledger.pin_store_for_unload", lambda *items: None)
    monkeypatch.setattr(store, "_drop", dropped.append)
    monkeypatch.setattr(store, "release_retained_cleanup", lambda: None)
    store.unload_all()
    assert dropped == ["refiner", "uncond", "cond"]


@pytest.mark.parametrize("topology", [{"cfg": 2}, {"ring": 2}, {"dp": 2}, {"fsdp": True}])
def test_unsupported_topology_refuses_before_model_load(monkeypatch, topology):
    from dgx_monarch.actor import store_fsdp

    touched = []
    monkeypatch.setattr(store_fsdp, "_assert_artifact_runnable", lambda *args: touched.append(args))
    worker = SimpleNamespace(topology=topology)
    with pytest.raises(ValueError, match="pure Ulysses"):
        store_fsdp.ensure(worker, "finish.safetensors", slot="refiner")
    assert not touched


def test_ulysses_topology_is_supported():
    resident_slots.validate_refiner_topology({"ulysses": 2, "cfg": 1, "ring": 1, "dp": 1})
