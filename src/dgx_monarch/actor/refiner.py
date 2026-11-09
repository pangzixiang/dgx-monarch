"""Independent resident-refiner endpoint, serialized with all GPU work."""
from __future__ import annotations

from typing import Any

from monarch.actor import concurrent_endpoint

from ..loader_options import weight_dtype_from_options


class RefinerMixin:
    @concurrent_endpoint
    async def load_refiner_model(self: Any, unet_name: str, options: dict | None = None,
                                 lora_stack: list | None = None) -> dict:
        weight_dtype_from_options(options)
        async with self._gpu_lock:
            return await self._on_gpu(self._load_model_impl, unet_name, options, lora_stack, "refiner")
