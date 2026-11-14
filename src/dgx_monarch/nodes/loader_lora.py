"""Preserve the live LoRA stack when the H3/LTX readiness barrier reruns its loader.

The complete request is known at sampling. Eagerly loading the empty stack on
every loader invalidation strips an already correct resident LoRA stack and
then requires another bake. Deferring that weight load preserves the resident;
the sampler still prices and admits the actual complete request on every rank.
"""
from __future__ import annotations

from collections.abc import Mapping

from .graph_live import is_link, live_nodes

_LOADERS = frozenset({"DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader",
                      "DGXMonarchRefinerUNETLoader"})


def has_live_lora_successor(prompt: object, unique_id: object, unet_name: str) -> bool:
    """A real native LoRA node consumes a loader downstream of the readiness barrier.

    No predicted LoRA arguments or residency credit comes from the graph. A
    missing backend node identity or unknown transform keeps the eager path.
    """
    if unique_id is None or not isinstance(prompt, Mapping):
        return False
    node_id = str(unique_id)
    node = prompt.get(node_id)
    if (not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str)
            or node.get("class_type") not in _LOADERS
            or not isinstance(node.get("inputs"), Mapping)
            or node["inputs"].get("unet_name") != unet_name):
        return False
    mesh = node["inputs"].get("mesh")
    if not is_link(mesh):
        return False
    barrier = prompt.get(str(mesh[0]))
    if (not isinstance(barrier, Mapping)
            or barrier.get("class_type") != "H3LTXRefinementReady" or mesh[1] != 2):
        return False
    live = live_nodes(prompt)
    for child_id, child in prompt.items():
        if (str(child_id) not in live or not isinstance(child, Mapping)
                or child.get("class_type") != "DGXMonarchLoraLoader"):
            continue
        inputs = child.get("inputs")
        link = inputs.get("model") if isinstance(inputs, Mapping) else None
        if isinstance(link, list | tuple) and is_link(link) and str(link[0]) == node_id and link[1] == 0:
            return True
    return False
