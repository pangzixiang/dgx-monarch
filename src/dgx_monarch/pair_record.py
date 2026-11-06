"""Record the sequential swap-pair residency proof.

Run this on the driver (ComfyUI output root holds both ledgers) after the two
renders below exist on disk:

1. a COLD (stock residency) render of the requested model, whose SHA-256 is the
   reference; and
2. the render the same requested model produced after the resident model was
   swapped out of the slot, whose SHA-256 must be bit-identical.

The tool never renders, never guesses a SHA and never infers equality: equal
digests are the whole proof.  A pair PASS lets the driver turn the slab and
low-RSS LoRA levers on when it swaps exactly these two requests.

Usage::

    python -m dgx_monarch.pair_record --resident @resident.json --requested @requested.json \
        --reference-sha <sha256> --swap-sha <sha256> [--note text]
    python -m dgx_monarch.pair_record --list
"""
from __future__ import annotations

import argparse
import json
import sys

from . import pair_admission
from .log import get_logger

log = get_logger(__name__)


def _model_request(value: str) -> dict:
    text = value
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as handle:
            text = handle.read()
    request = json.loads(text)
    if not isinstance(request, dict):
        raise SystemExit("model request must be a JSON object")
    if not isinstance(request.get("unet_name"), str) or not request["unet_name"]:
        raise SystemExit("model request must name unet_name")
    return request


def _ledger_dir() -> str:
    import folder_paths

    return folder_paths.get_output_directory()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident", help="JSON (or @file) of the resident model request")
    parser.add_argument("--requested", help="JSON (or @file) of the requested model request")
    parser.add_argument("--reference-sha", help="cold stock render SHA-256")
    parser.add_argument("--swap-sha", help="post-swap render SHA-256")
    parser.add_argument("--note", default="", help="free-form operator note stored with the proof")
    parser.add_argument("--ledger-dir", default=None, help="override the ledger directory")
    parser.add_argument("--list", action="store_true", help="print every intact pair row")
    parser.add_argument("--lookup", nargs=2, metavar=("RESIDENT_JSON", "REQUESTED_JSON"),
                        help="print the live verdict for this pair")
    arguments = parser.parse_args(argv)
    directory = arguments.ledger_dir or _ledger_dir()
    ledger = pair_admission.PairLedger(directory)

    if arguments.list:
        for entry in ledger.entries():
            print(json.dumps(entry, sort_keys=True))
        return 0

    if arguments.lookup:
        resident = _model_request(arguments.lookup[0])
        requested = _model_request(arguments.lookup[1])
        key = pair_admission.pair_key(resident, requested)
        entry = ledger.lookup(key)
        print(json.dumps({"key": key, "entry": entry}, sort_keys=True))
        return 0

    if not (arguments.resident and arguments.requested
            and arguments.reference_sha and arguments.swap_sha):
        parser.error("recording needs --resident, --requested, --reference-sha and --swap-sha")
    reference = arguments.reference_sha.strip().lower()
    swap = arguments.swap_sha.strip().lower()
    for name, digest in (("--reference-sha", reference), ("--swap-sha", swap)):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SystemExit(f"{name} must be a 64-hex SHA-256")
    if reference != swap:
        raise SystemExit(
            "the swap render is not bit-identical to the cold reference; "
            "nothing was recorded")
    resident = _model_request(arguments.resident)
    requested = _model_request(arguments.requested)
    if pair_admission.same_model(resident, requested):
        raise SystemExit("resident and requested are the same model; a pair needs two")
    key = pair_admission.pair_key(resident, requested)
    entry = ledger.record_pass(
        key, resident, requested,
        {"reference_sha256": reference, "swap_sha256": swap,
         "note": str(arguments.note or "")[:500]})
    print(json.dumps({"recorded": True, "key": key, "entry": entry}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
