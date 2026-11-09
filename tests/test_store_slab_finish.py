"""Slab completion settles quantized-only copies before retiring allocations."""
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor.store_slab_finish import reabsorb_and_trim


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quantized_only_completion_orders_copy_sync_trim_and_reports_strays(
    monkeypatch, caplog, device,
):
    events = []
    model = object()
    stats = {"reabsorbed": 0, "quant_reabsorbed_gib": 2.0,
             "skipped": 1, "quant_stray_gib": 0.25}

    def reabsorb(actual):
        assert actual is model
        events.append("copy")
        return stats

    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = lambda: events.append("trim")
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda actual: events.append(("sync", actual)))
    base = SimpleNamespace(model=SimpleNamespace(diffusion_model=model), load_device=device)
    slab = SimpleNamespace(reabsorb=reabsorb, total_gib=4.0)
    with caplog.at_level(logging.INFO):
        reabsorb_and_trim(base, slab, logging.getLogger("dgx-monarch"), phase="lazy-swap")
    assert events == (["copy", ("sync", "cuda:0"), "trim"] if device == "cuda:0"
                      else ["copy", "trim"])
    assert "quant reabsorbed 2.0 GiB" in caplog.text
    assert "skipped 1 plain strays" in caplog.text
    assert "quant strays 0.25 GiB stay in cudaMalloc" in caplog.text


@pytest.mark.parametrize("stage", ["copy", "sync", "trim"])
def test_completion_failure_never_reports_success(monkeypatch, caplog, stage):
    events = []

    def step(name):
        events.append(name)
        if name == stage:
            raise RuntimeError(f"{stage} failed")
        return {}

    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = lambda: step("trim")
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: step("sync"))
    slab = SimpleNamespace(reabsorb=lambda _model: step("copy"), total_gib=4.0)
    base = SimpleNamespace(model=SimpleNamespace(diffusion_model=object()), load_device="cuda:0")
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        reabsorb_and_trim(base, slab, logging.getLogger("dgx-monarch"), phase="lazy-swap")
    assert events == ["copy", "sync", "trim"][:["copy", "sync", "trim"].index(stage) + 1]
    assert "slab residency (lazy-swap)" not in caplog.text
