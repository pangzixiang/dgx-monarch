"""Prompt-local evidence that an H3 reference encode ended before a loader.

Only the trusted in-process readiness boundary records evidence, after its
offload/soft-trim completed. No prompt field, global switch, file-size model or
previous job can assert completion. Missing evidence retains the cold price.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .graph_live import is_link, links_in, live_nodes

if TYPE_CHECKING:
    from .loader_graph import DriverStackTerms

_LIMIT = 64
_LOCK = threading.Lock()
# Session lifetime, FIFO-capped; each proof is valid for one exact execution
# object only. Recycle clears the store, and eviction restores the cold price.
_COMPLETED: dict[tuple[str, int | None, int, str, str], tuple[Mapping, set[str]]] = {}
_REFERENCE_CLASS = "MiniMaxH3ReferenceToVideo"
_BOUNDARY_CLASS = "H3LTXRefinementReady"


def clear() -> None:
    """Recycle loses proofs conservatively, just like other loader memos."""
    with _LOCK:
        _COMPLETED.clear()


def _context() -> tuple[str, str, int | None] | None:
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        if (not isinstance(ctx.prompt_id, str) or not ctx.prompt_id
                or not isinstance(ctx.node_id, str) or not ctx.node_id
                or ctx.list_index not in (None, 0)):
            return None
        return ctx.prompt_id, ctx.node_id, ctx.list_index
    except Exception:
        return None


def _source() -> str:
    from ..runtime_provenance import dgx_source_manifest_sha256

    return dgx_source_manifest_sha256(Path(__file__).resolve().parents[1])


def _key(prompt: object) -> tuple[str, int | None, int, str, str] | None:
    context = _context()
    if context is None or not isinstance(prompt, Mapping) or not prompt:
        return None
    # Ignore Comfy's mutable is_changed cache, but bind every class/input,
    # including filenames, references, encoder artifacts and graph rewrites.
    graph = {str(n): {k: v for k, v in node.items() if k in ("class_type", "inputs")}
             for n, node in prompt.items() if isinstance(node, Mapping)}
    digest = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    source = _source()
    if len(source) != 64:
        return None
    return context[0], context[2], id(prompt), digest, source


def _video_links(inputs: Mapping) -> int:
    return sum(len(links_in(value)) for name, value in inputs.items()
               if isinstance(name, str) and name.startswith("ref_video")
               and not name.startswith(("ref_video_audio", "ref_video_audios")))


def _needed(prompt: Mapping) -> set[str] | None:
    """Every live video-bearing node must be a known reference node."""
    result = set()
    live = live_nodes(prompt)
    for node_id, node in prompt.items():
        if not isinstance(node, Mapping) or str(node_id) not in live:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping) or not any(
                links_in(value) for name, value in inputs.items()
                if isinstance(name, str) and name.startswith("ref_video")):
            continue
        if node.get("class_type") != _REFERENCE_CLASS or not _video_links(inputs):
            return None
        result.add(str(node_id))
    return result or None


def _encoded(conditioning: object, count: int) -> bool:
    if not isinstance(conditioning, list | tuple) or not conditioning:
        return False
    for entry in conditioning:
        if not isinstance(entry, list | tuple) or len(entry) < 2 or not isinstance(entry[1], Mapping):
            return False
        blocks = entry[1].get("minimax_refs")
        if not isinstance(blocks, list | tuple):
            return False
        videos = [b for b in blocks if isinstance(b, Mapping) and b.get("kind") in ("video", "video_audio")]
        if len(videos) != count:
            return False
        for block in videos:
            shape = getattr(block.get("latent"), "shape", ())
            dims = tuple(block.get(k) for k in ("latent_t", "latent_h", "latent_w"))
            if (any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims)
                    or len(shape) != 5 or tuple(shape[:2]) != (1, 24) or tuple(shape[2:]) != dims):
                return False
            if block["kind"] == "video_audio" and block.get("audio_latent") is None:
                return False
    return True


def record_completed_reference_encoding(
    *, prompt: object, boundary_node_id: object, conditioning: object, av_latent: object,
) -> bool:
    """Record actual encoded results at the executing readiness boundary only.

    The caller runs this after driver offload and soft worker trim succeed.
    Its hidden PROMPT/UNIQUE_ID inputs are backend-supplied, not public claims.
    """
    try:
        context = _context()
        if context is None or context[1] != str(boundary_node_id) or not isinstance(prompt, Mapping):
            return False
        boundary = prompt.get(str(boundary_node_id))
        if not isinstance(boundary, Mapping) or boundary.get("class_type") != _BOUNDARY_CLASS:
            return False
        inputs = boundary.get("inputs", {})
        link = inputs.get("conditioning")
        if not is_link(link) or link[1] != 0 or inputs.get("av_latent") != [str(link[0]), 1]:
            return False
        node = prompt.get(str(link[0]))
        if not isinstance(node, Mapping) or node.get("class_type") != _REFERENCE_CLASS:
            return False
        count = _video_links(node.get("inputs", {}))
        if (not count or not isinstance(av_latent, Mapping) or av_latent.get("samples") is None
                or not _encoded(conditioning, count)):
            return False
        key = _key(prompt)
        if key is None:
            return False
        with _LOCK:
            # Retain the graph object too: a reused public prompt_id or a
            # recycled Python id cannot turn a different execution into proof.
            _COMPLETED.setdefault(key, (prompt, set()))[1].add(str(link[0]))
            while len(_COMPLETED) > _LIMIT:
                _COMPLETED.pop(next(iter(_COMPLETED)))
        return True
    except Exception:
        return False  # no proof is the unchanged conservative price


def credit(terms: DriverStackTerms, prompt: object) -> DriverStackTerms:
    """Remove only the completed encode transient; preserve every other term."""
    if not terms.ref_bytes or not isinstance(prompt, Mapping):
        return terms
    try:
        key, needed = _key(prompt), _needed(prompt)
        if key is None or not needed:
            return terms
        with _LOCK:
            proof = _COMPLETED.get(key)
            proven = proof is not None and proof[0] is prompt and needed.issubset(proof[1])
        if proven:
            return replace(terms, ref_bytes=0,
                           ref_encode_note="reference encoding completed before this loader in this prompt")
    except Exception:
        pass
    return terms
