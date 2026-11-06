"""Driver-side sequential swap-pair residency admission.

The identity gate proves one model per ceremony, so a dual-model request is
stamped stock and a fleet refuses to price a swap while the resident still
occupies the slot.  This module carries the missing authority for the one
sequence this deployment actually runs: render model A, then load model B into
the same slot on the same fleet.

The proof is evidence-bound, not inferred.  ``record_pass`` accepts a pair only
when the operator supplies the bit-identical SHA-256 of a cold (stock) render of
B and of the render B produced after swapping A out of the slot.  The resulting
PASS is bound to the two exact model requests, the ComfyUI commit, the
dgx-monarch package version, the canonical package-source manifest and this
module's protocol version, exactly like a normal gate verdict.

The resident side is process-local on purpose: after a driver restart no
resident is assumed, so a pair grant cannot fire until the process itself has
dispatched a real render.  Capacity pricing still measures the live slot; this
module only supplies the decision that a swap of a proven pair is safe.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from typing import Any

from . import __version__, runtime_provenance
from .gate_ledger import comfy_commit
from .log import get_logger

log = get_logger(__name__)

PAIR_PROTOCOL_VERSION = 1
PAIR_LEDGER_NAME = "dgxm_pair_ledger.jsonl"

# handle identity -> the last model request this driver process dispatched.
_RESIDENTS: dict[int, dict] = {}


def _request_record(request: dict | None) -> dict:
    """The identity dimensions a pair proves: checkpoint, loader options, LoRA names.

    LoRA strengths are excluded for the same reason the combo key excludes them:
    the strength exercises the same weights and the pair proof does not depend on
    it.  Order is normalized so two equal stacks always hash equally.
    """
    request = request or {}
    loras = []
    for entry in request.get("loras") or []:
        name = entry.get("lora_name") if isinstance(entry, dict) else entry
        if name is None:
            continue
        loras.append(str(name))
    options = request.get("options") or {}
    return {
        "unet_name": str(request.get("unet_name") or ""),
        "options": sorted((str(key), item) for key, item in options.items()),
        "loras": sorted(loras),
    }


def pair_key(resident_request: dict | None, requested_request: dict | None) -> str:
    """The 24-hex key a pair PASS is filed under (resident first, requested second)."""
    payload = json.dumps(
        [_request_record(resident_request), _request_record(requested_request)],
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def note_dispatch(handle: Any, request: dict) -> None:
    """Remember the model this render just asked the fleet to hold."""
    _RESIDENTS[id(handle)] = copy.deepcopy(_request_record(request))


def resident_request(handle: Any) -> dict | None:
    return _RESIDENTS.get(id(handle))


def forget_resident(handle: Any) -> None:
    _RESIDENTS.pop(id(handle), None)


def same_model(first: dict | None, second: dict | None) -> bool:
    return _request_record(first) == _request_record(second)


class PairLedger:
    """Append-only pair PASS journal beside the gate ledger.

    Line-local parsing keeps a torn tail from hiding earlier evidence; the last
    intact row for a key wins, and a row is trusted only when every identity
    dimension matches the live driver.
    """

    def __init__(self, directory: str):
        self.path = os.path.join(directory, PAIR_LEDGER_NAME)

    def entries(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except FileNotFoundError:
            return []
        except OSError as error:
            log.warning("pair ledger unreadable (%r)", error)
            return []
        entries = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            if self._valid(entry):
                entries.append(entry)
        return entries

    @staticmethod
    def _valid(entry: object) -> bool:
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("key"), str) or not entry["key"]:
            return False
        if entry.get("verdict") not in ("PASS", "FAIL"):
            return False
        if not isinstance(entry.get("resident"), dict) or not isinstance(entry.get("requested"), dict):
            return False
        if not isinstance(entry.get("evidence"), dict):
            return False
        if any(field in entry and not isinstance(entry[field], str)
               for field in ("comfy", "dgx_monarch", "dgx_source")):
            return False
        if ("pair_protocol" in entry
                and (isinstance(entry["pair_protocol"], bool)
                     or not isinstance(entry["pair_protocol"], int))):
            return False
        return True

    def lookup(self, key: str) -> dict | None:
        match = None
        for entry in self.entries():
            if entry["key"] == key:
                match = entry
        return match

    def record_pass(self, key: str, resident_request: dict | None,
                    requested_request: dict | None, evidence: dict) -> dict:
        entry = {
            "key": key,
            "verdict": "PASS",
            "resident": _request_record(resident_request),
            "requested": _request_record(requested_request),
            "evidence": dict(evidence),
            "comfy": comfy_commit(),
            "dgx_monarch": __version__,
            "dgx_source": runtime_provenance.cached_dgx_source_manifest_sha256(),
            "pair_protocol": PAIR_PROTOCOL_VERSION,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(directory)
        return entry


def _fsync_directory(directory: str) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def pair_swap_grant(handle: Any, model_request: dict, *, ledger_dir: str) -> dict | None:
    """Return the live pair PASS covering this resident -> requested swap.

    ``None`` means no authority: unknown resident, a reuse (not a swap), a
    missing/stale/failed row, or a runtime identity mismatch.  The caller falls
    back to the ordinary gate path, which prices stock.
    """
    resident = resident_request(handle)
    if resident is None or same_model(resident, model_request):
        return None
    key = pair_key(resident, model_request)
    entry = PairLedger(ledger_dir).lookup(key)
    if entry is None or entry.get("verdict") != "PASS":
        return None
    if entry.get("pair_protocol") != PAIR_PROTOCOL_VERSION:
        return None
    if entry.get("dgx_monarch") != __version__:
        return None
    if entry.get("dgx_source") != runtime_provenance.cached_dgx_source_manifest_sha256():
        return None
    if entry.get("comfy") != comfy_commit():
        return None
    evidence = entry.get("evidence") or {}
    if not evidence.get("reference_sha256") or evidence.get("reference_sha256") != evidence.get("swap_sha256"):
        return None
    return entry
