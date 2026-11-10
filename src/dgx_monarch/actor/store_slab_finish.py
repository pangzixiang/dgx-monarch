"""Return baked parameters to their owned slab before publishing residency."""
from __future__ import annotations

import logging
from typing import Any


def reabsorb_and_trim(base: Any, slab: Any, logger: logging.Logger, *, phase: str) -> None:
    """Preserve the current weights, settle their copies, and release retired allocations.

    ``reabsorb`` clones quantized wrappers around relocated payloads with their
    existing layout and scales. The pristine-load certificate remains a load
    attestation, not a comparison of these baked weights with the checkpoint.
    Failures propagate to the owning load/swap transaction before adoption.
    """
    import comfy.model_management as mm
    import torch

    stats = slab.reabsorb(base.model.diffusion_model)
    # reabsorb's plain-parameter loop synchronizes when it moves plain weights;
    # a quantized-only migration needs the same completion boundary before trim.
    device = getattr(base, "load_device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    mm.soft_empty_cache()
    logger.info(
        "slab residency (%s): %.2f GiB slab, reabsorbed %s plain strays (%s GiB), "
        "quant reabsorbed %s GiB, skipped %s plain strays, quant strays %s GiB stay in cudaMalloc",
        phase, slab.total_gib, stats.get("reabsorbed", 0), stats.get("reabsorbed_gib", 0.0),
        stats.get("quant_reabsorbed_gib", 0.0), stats.get("skipped", 0),
        stats.get("quant_stray_gib", 0.0),
    )
