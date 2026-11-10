"""A completed real reference encode is not a still-pending 40 GiB workspace."""
from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest

from dgx_monarch import driver_footprint
from dgx_monarch.nodes import graph_load_price, loader_graph, loader_preflight
from dgx_monarch.nodes import h3_reference_completion as completion

GIB = 2**30
REAL_CONTEXT = completion._context


def graph() -> dict:
    return {
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "h3-nvfp4.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "h3-vae.safetensors"}},
        "20": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": "A test", "length": 124,
            "ref_videos.ref_video_0": ["104", 0]}},
        "29": {"class_type": "H3LTXRefinementReady", "inputs": {
            "conditioning": ["20", 0], "av_latent": ["20", 1]}},
        "104": {"class_type": "GetVideoComponents", "inputs": {"video": ["103", 0]}},
        "103": {"class_type": "LoadVideo", "inputs": {"file": "source-tail56.mp4"}},
    }


def encoded() -> list:
    return [[object(), {"minimax_refs": [{"kind": "video", "latent_t": 17,
        "latent_h": 48, "latent_w": 84, "latent": SimpleNamespace(shape=(1, 24, 17, 48, 84))}]}]]


def terms() -> loader_graph.DriverStackTerms:
    return loader_graph.DriverStackTerms(te_bytes=33*GIB, vae_bytes=5*GIB,
        ref_bytes=40*GIB, video_refs=1, resolved=True, activation_bytes=14*GIB)


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    completion.clear()
    state = SimpleNamespace(prompt_id="request-a", node_id="29", list_index=0)
    monkeypatch.setattr(completion, "_context", lambda: (state.prompt_id, state.node_id, state.list_index))
    monkeypatch.setattr(completion, "_source", lambda: "a"*64)
    monkeypatch.setattr(loader_graph, "_comfy_has_loaded_models", lambda: False)
    loader_preflight.reset_memos()
    yield state
    completion.clear()


def record(prompt, conditioning=None, boundary="29"):
    return completion.record_completed_reference_encoding(prompt=prompt, boundary_node_id=boundary,
        conditioning=encoded() if conditioning is None else conditioning, av_latent={"samples": object()})


def test_cold_graph_and_a_completion_boolean_keep_the_entire_price():
    prompt = graph()
    prompt["29"]["inputs"]["encoded"] = True
    assert completion.credit(terms(), prompt) == terms()
    assert graph_load_price.site_terms(terms(), "loader", prompt=prompt).ref_bytes == 40*GIB


def test_a_cold_price_never_needs_runtime_or_source_probes(monkeypatch):
    def unexpected():
        pytest.fail("cold graph probed a runtime without any completion proof")
    monkeypatch.setattr(completion, "_context", unexpected)
    monkeypatch.setattr(completion, "_source", unexpected)
    assert completion.credit(terms(), graph()).ref_bytes == 40*GIB


def test_only_this_completed_transient_is_credited_and_every_other_term_survives(isolation):
    prompt, before = graph(), terms()
    assert record(prompt)
    isolation.node_id = "2"
    after = graph_load_price.site_terms(before, "loader", prompt=prompt)
    assert after.ref_bytes == 0
    assert after.total == before.total-40*GIB
    expected = asdict(before)
    expected.update(ref_bytes=0, ref_encode_note=after.ref_encode_note)
    assert asdict(after) == expected
    assert "completed" in after.ref_encode_note


@pytest.mark.parametrize("change", ["prompt-id", "replayed-id", "reference", "encoder", "source", "list", "reset"])
def test_stale_or_reused_evidence_never_crosses_an_execution(change, isolation, monkeypatch):
    prompt = graph()
    assert record(prompt)
    if change == "prompt-id":
        isolation.prompt_id = "request-b"
    if change == "replayed-id":
        prompt = copy.deepcopy(prompt)
    if change == "reference":
        prompt["103"]["inputs"]["file"] = "different.mp4"
    if change == "encoder":
        prompt["3"]["inputs"]["clip_name"] = "changed.safetensors"
    if change == "source":
        monkeypatch.setattr(completion, "_source", lambda: "b"*64)
    if change == "list":
        isolation.list_index = 1
    if change == "reset":
        loader_preflight.reset_memos()
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB


@pytest.mark.parametrize("payload", [[], [[None, {}]], [[None, {"minimax_refs": []}]],
    [[None, {"minimax_refs": [{"kind": "video", "latent_t": 17, "latent_h": 48, "latent_w": 84}]}]]])
def test_missing_encoded_payload_cannot_claim_completion(payload):
    prompt = graph()
    assert not record(prompt, payload)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB


def test_wrong_boundary_context_or_unfinished_latent_cannot_record(isolation):
    prompt = graph()
    isolation.node_id = "2"
    assert not record(prompt)
    isolation.node_id = "29"
    assert not completion.record_completed_reference_encoding(prompt=prompt, boundary_node_id="29",
        conditioning=encoded(), av_latent={})
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB


def test_unknown_video_nodes_and_a_second_unfinished_reference_keep_the_reservation():
    prompt = graph()
    assert record(prompt)
    # Recording another graph first is needed because adding nodes invalidates
    # the graph digest too; after this record the extra reference is unproved.
    prompt["21"] = copy.deepcopy(prompt["20"])
    assert record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB
    prompt["21"]["class_type"] = "UnknownReferenceEncoder"
    assert record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB


def test_two_completed_reference_nodes_can_share_one_completed_transient(isolation):
    prompt = graph()
    prompt["21"] = copy.deepcopy(prompt["20"])
    prompt["28"] = {"class_type": "H3LTXRefinementReady", "inputs": {
        "conditioning": ["21", 0], "av_latent": ["21", 1]}}
    assert record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB
    isolation.node_id = "28"
    assert record(prompt, boundary="28")
    assert completion.credit(terms(), prompt).ref_bytes == 0


def test_eviction_only_loses_a_credit(isolation):
    first = graph()
    assert record(first)
    for i in range(completion._LIMIT):
        isolation.prompt_id = f"later-{i}"
        assert record(graph())
    assert len(completion._COMPLETED) == completion._LIMIT
    isolation.prompt_id = "request-a"
    assert completion.credit(terms(), first).ref_bytes == 40*GIB


def test_no_comfy_context_or_broken_provenance_keeps_cold_price(monkeypatch):
    prompt = graph()
    monkeypatch.setattr(completion, "_context", lambda: None)
    assert not record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB
    monkeypatch.setattr(completion, "_context", lambda: ("p", "29", 0))
    monkeypatch.setattr(completion, "_source", lambda: (_ for _ in ()).throw(OSError("unreadable")))
    assert not record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB


def test_recorded_1080p_budget_refuses_cold_and_fits_only_the_completed_window():
    prompt = graph()
    original = loader_graph.DriverStackTerms(te_bytes=35_719_128_713, vae_bytes=5_253_409_878,
        ref_bytes=40*GIB, video_refs=1, resolved=True, activation_bytes=14_476_980_224)
    def windows(stack):
        shape = loader_preflight._Shape(profile=driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"],
            mem_available=int(110.3*GIB), weight_bytes=20_970_379_616,
            weights_resident=False, co_resident=True, floor_reserve=5*GIB, stack_bytes=stack.total)
        return loader_preflight._windows(shape, slab=True)
    cold = windows(original)
    assert not all(x.fits for x in cold)
    assert cold[1].projected+cold[1].reserve == 124_738_280_511
    assert record(prompt)
    credited = graph_load_price.site_terms(original, "loader", prompt=prompt)
    completed = windows(credited)
    assert all(x.fits for x in completed)
    assert completed[1].projected+completed[1].reserve == 81_788_607_551
    assert credited.te_bytes == original.te_bytes
    assert credited.vae_bytes == original.vae_bytes
    assert credited.activation_bytes == original.activation_bytes


@pytest.mark.parametrize("invalid", ["missing", "wrong-node", "later-list"])
def test_backend_context_is_required_not_a_caller_supplied_prompt_id(invalid, monkeypatch):
    utility = ModuleType("comfy_execution.utils")
    state = SimpleNamespace(prompt_id="backend-id", node_id="29", list_index=0)
    utility.get_executing_context = lambda: state
    monkeypatch.setitem(sys.modules, "comfy_execution", ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", utility)
    monkeypatch.setattr(completion, "_context", REAL_CONTEXT)
    prompt = graph()
    assert record(prompt)
    completion.clear()
    if invalid == "missing":
        utility.get_executing_context = lambda: None
    elif invalid == "wrong-node":
        state.node_id = "2"
    else:
        state.list_index = 1
    assert not record(prompt)
    assert completion.credit(terms(), prompt).ref_bytes == 40*GIB
