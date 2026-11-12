"""Avoid loading a bare base ahead of its live native LoRA transform.

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
    """A real native LoRA node consumes this loader in the executable graph.

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
