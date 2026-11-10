"""A readiness barrier must not strip the same resident LoRAs every queue."""
from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import capacity_quote
from dgx_monarch.nodes import graph_live, loader_lora, loaders
from dgx_monarch.nodes.common import MeshSpec

NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
STACK = (("pika.safetensors", 1.0), ("camera.safetensors", 0.8))


def graph(loader_class="DGXMonarchUNETLoader") -> dict:
    return {
        "1": {"class_type": "DGXMonarchInit", "inputs": {}},
        "29": {"class_type": "H3LTXRefinementReady", "inputs": {"mesh": ["1", 0]}},
        "2": {"class_type": loader_class, "inputs": {"unet_name": NAME, "mesh": ["29", 2]}},
        "6": {"class_type": "DGXMonarchLoraLoader", "inputs": {
            "model": ["2", 0], "lora_name": STACK[0][0], "strength_model": STACK[0][1]}},
        "7": {"class_type": "DGXMonarchLoraLoader", "inputs": {
            "model": ["6", 0], "lora_name": STACK[1][0], "strength_model": STACK[1][1]}},
        "out": {"class_type": "PreviewAny", "inputs": {"source": ["7", 0]}},
    }


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    # The same output flags Comfy's real registry supplies, without importing it.
    monkeypatch.setattr(graph_live, "output_node", lambda kind: kind == "PreviewAny")


@pytest.mark.parametrize("kind", ["DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader",
                                  "DGXMonarchRefinerUNETLoader"])
def test_each_native_loader_can_defer_to_its_actual_lora_request(kind):
    assert loader_lora.has_live_lora_successor(graph(kind), "2", NAME)


@pytest.mark.parametrize("case", ["headless", "no-id", "wrong-id", "other-file", "unknown-loader",
                                  "unknown-transform", "other-parent", "other-output", "unused"])
def test_only_a_live_native_transform_of_this_loader_defers(case):
    prompt, node_id, name = graph(), "2", NAME
    if case == "headless":
        prompt = None
    if case == "no-id":
        node_id = None
    if case == "wrong-id":
        node_id = "42"
    if case == "other-file":
        name = "different.safetensors"
    if case == "unknown-loader":
        prompt["2"]["class_type"] = "ForeignLoader"
    if case == "unknown-transform":
        prompt["6"]["class_type"] = "ForeignLoRA"
    if case == "other-parent":
        prompt["6"]["inputs"]["model"] = ["42", 0]
    if case == "other-output":
        prompt["6"]["inputs"]["model"] = ["2", 1]
    if case == "unused":
        prompt["out"]["inputs"]["source"] = ["2", 0]
    assert not loader_lora.has_live_lora_successor(prompt, node_id, name)


@pytest.fixture
def rig(monkeypatch):
    from dgx_monarch import capacity_agreement, first_render, mesh_setup
    from dgx_monarch.nodes import (
        abandoned_heal,
        consent_projection,
        loader_preflight,
        render_preflight,
        render_session,
        render_validation,
    )

    events, loads = [], []
    held_key = repr((NAME, (), STACK, "int8"))
    resident = {"cond": {"request_key": held_key, "residency": "slab",
                         "residency_rung": capacity_quote.RUNG_EXPLICIT}, "uncond": None}

    def call_all(endpoint, name, options, loras, **kwargs):
        # This is the observed V1 failure: an empty eager load removes the blend.
        loads.append((endpoint, tuple(loras)))
        resident["cond"]["request_key"] = repr((name, (), tuple(loras), "int8"))
        return [{"host": "rank", "transition": "hot-swap"}]

    handle = SimpleNamespace(world=2, call_all=call_all)
    mesh = MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True)
    for name in ("preflight_upstream_gated_artifact", "preflight_comfy_managed_topology"):
        monkeypatch.setattr(loader_preflight, name, lambda *a, **k: None)
    monkeypatch.setattr(render_validation, "preflight_graph_batch_divides_dp", lambda *a: None)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *a, **k: events.append("footprint") or None)
    monkeypatch.setattr(consent_projection, "project_for_loader", lambda *a, **k: None)
    monkeypatch.setattr(loaders, "ensure_live", lambda *a, **k: events.append("live") or handle)
    monkeypatch.setattr(abandoned_heal, "heal_before_eager_setup", lambda h, **k: h)
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel", lambda *a: None)
    monkeypatch.setattr(render_session, "mutation_render_session", lambda h: contextlib.nullcontext())
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *a: events.append("setup"))
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda token: {})
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *a: events.append("eager-admission"))
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *a, **k: events.append("eager-certificate"))
    monkeypatch.setattr(first_render, "load_deferred", lambda *a, **k: events.append("deferred"))
    return mesh, events, loads, resident, held_key


def test_two_invalidated_queues_preserve_the_blend_and_native_reuse_quote(rig):
    mesh, events, loads, resident, held_key = rig
    for _ in range(2):
        spec, = loaders.DGXMonarchUNETLoader().load(mesh, NAME, unique_id="2", prompt=copy.deepcopy(graph()))
        for name, strength in STACK:
            spec, = loaders.DGXMonarchLoraLoader().load_lora(spec, name, strength)
        key = repr((spec.unet_name, (), tuple((p["name"], p["strength"]) for p in spec.loras), "int8"))
        assert key == held_key == resident["cond"]["request_key"]
        row = capacity_quote.price(path="/missing/test.safetensors", unet_name=NAME, model_options={},
            slab_weights=True, slab_capable_path=True, lora_low_rss=True, fsdp_launch=False,
            blocked_reason="", authoritative_slab_retry=False, memoized_family=lambda p: None,
            vouched_families=frozenset(), rescue_consent=None, store_snapshot=resident, request_key=key,
            fit_probe=lambda *a: pytest.fail("a true reuse must not price a fresh load"))
        assert row.would_load == "reuse" and row.verdict == "fits" and not row.priced
    assert not loads
    assert events == ["footprint", "live", "setup", "deferred"] * 2


def test_without_a_lora_successor_the_normal_eager_admission_and_load_stay(rig):
    mesh, events, loads, _resident, _key = rig
    prompt = graph()
    prompt["out"]["inputs"]["source"] = ["2", 0]
    loaders.DGXMonarchUNETLoader().load(mesh, NAME, unique_id="2", prompt=prompt)
    assert loads == [("load_model", ())]
    assert events == ["footprint", "live", "setup", "eager-admission", "eager-certificate"]


def test_a_changed_lora_stack_still_reads_as_a_native_swap(rig):
    _mesh, _events, _loads, resident, held_key = rig
    changed = repr((NAME, (), ((STACK[0][0], 0.5), STACK[1]), "int8"))
    assert changed != held_key
    assert capacity_quote._slot_context(resident, "cond", changed)["would_load"] == "swap"
