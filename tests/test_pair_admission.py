"""Sequential swap-pair residency admission: keys, journal, grant and CLI."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from dgx_monarch import pair_admission


def request(name: str, **options) -> dict:
    return {"unet_name": name, "options": dict(options), "loras": []}


class FakeHandle:
    pass


class PairKeyTests(unittest.TestCase):
    def test_key_is_stable_and_order_sensitive(self):
        first = pair_admission.pair_key(request("a"), request("b"))
        self.assertEqual(first, pair_admission.pair_key(request("a"), request("b")))
        self.assertNotEqual(first, pair_admission.pair_key(request("b"), request("a")))
        self.assertEqual(len(first), 24)

    def test_key_ignores_lora_strength_but_not_lora_name(self):
        base = {"unet_name": "a", "options": {}, "loras": [{"lora_name": "x", "strength": 0.5}]}
        other = {"unet_name": "a", "options": {}, "loras": [{"lora_name": "x", "strength": 1.0}]}
        self.assertEqual(pair_admission.pair_key(base, request("b")),
                         pair_admission.pair_key(other, request("b")))
        renamed = {"unet_name": "a", "options": {}, "loras": [{"lora_name": "y", "strength": 1.0}]}
        self.assertNotEqual(pair_admission.pair_key(base, request("b")),
                            pair_admission.pair_key(renamed, request("b")))


class ResidentRegistryTests(unittest.TestCase):
    def test_note_dispatch_tracks_each_handle_independently(self):
        first, second = FakeHandle(), FakeHandle()
        pair_admission.note_dispatch(first, request("a"))
        self.assertEqual(pair_admission.resident_request(first)["unet_name"], "a")
        self.assertIsNone(pair_admission.resident_request(second))
        pair_admission.forget_resident(first)
        self.assertIsNone(pair_admission.resident_request(first))

    def test_same_model_normalizes_order_of_request_fields(self):
        self.assertTrue(pair_admission.same_model(
            {"unet_name": "a", "options": {"b": 2, "a": 1}, "loras": []},
            {"options": {"a": 1, "b": 2}, "unet_name": "a", "loras": []}))
        self.assertFalse(pair_admission.same_model(request("a"), request("b")))


def _pin_provenance() -> mock._patch:
    """Keep the package-source digest out of the test: it walks the live tree."""
    return mock.patch.object(
        pair_admission.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        return_value="1" * 64,
    )


class PairLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="dgxm-pair-")
        self.ledger = pair_admission.PairLedger(self.directory)
        self.resident, self.requested = request("h3"), request("ltx")
        self.key = pair_admission.pair_key(self.resident, self.requested)
        pinned = _pin_provenance()
        pinned.start()
        self.addCleanup(pinned.stop)

    def record(self):
        return self.ledger.record_pass(
            self.key, self.resident, self.requested,
            {"reference_sha256": "a" * 64, "swap_sha256": "a" * 64})

    def test_record_then_lookup_returns_the_row(self):
        row = self.record()
        self.assertEqual(row["verdict"], "PASS")
        self.assertEqual(self.ledger.lookup(self.key)["key"], self.key)
        self.assertIsNone(self.ledger.lookup("missing"))

    def test_rows_append_and_last_writer_wins(self):
        self.record()
        self.ledger.record_pass(self.key, self.resident, self.requested,
                                {"reference_sha256": "b" * 64, "swap_sha256": "b" * 64})
        self.assertEqual(len(self.ledger.entries()), 2)
        self.assertEqual(self.ledger.lookup(self.key)["evidence"]["swap_sha256"], "b" * 64)

    def test_torn_or_foreign_lines_do_not_erase_evidence(self):
        self.record()
        with open(self.ledger.path, "a", encoding="utf-8") as handle:
            handle.write('{"key": "torn"\n')
        self.assertEqual(len(self.ledger.entries()), 1)
        self.assertEqual(self.ledger.lookup(self.key)["key"], self.key)


class PairGrantTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="dgxm-pair-grant-")
        self.handle = FakeHandle()
        self.resident, self.requested = request("h3"), request("ltx")
        pair_admission.note_dispatch(self.handle, self.resident)
        pinned = _pin_provenance()
        pinned.start()
        self.addCleanup(pinned.stop)
        pair_admission.PairLedger(self.directory).record_pass(
            pair_admission.pair_key(self.resident, self.requested),
            self.resident, self.requested,
            {"reference_sha256": "c" * 64, "swap_sha256": "c" * 64})

    def grant(self):
        return pair_admission.pair_swap_grant(
            self.handle, self.requested, ledger_dir=self.directory)

    def test_proven_swap_grants(self):
        self.assertIsNotNone(self.grant())

    def test_reuse_is_not_a_swap(self):
        self.assertIsNone(pair_admission.pair_swap_grant(
            self.handle, self.resident, ledger_dir=self.directory))

    def test_unknown_resident_grants_nothing(self):
        self.assertIsNone(pair_admission.pair_swap_grant(
            FakeHandle(), self.requested, ledger_dir=self.directory))

    def test_runtime_identity_mismatches_do_not_grant(self):
        with mock.patch.object(pair_admission, "__version__", "0.0.0"):
            self.assertIsNone(self.grant())
        with mock.patch.object(pair_admission, "comfy_commit", return_value="deadbeef"):
            self.assertIsNone(self.grant())
        with mock.patch.object(pair_admission.runtime_provenance,
                               "cached_dgx_source_manifest_sha256", return_value="other"):
            self.assertIsNone(self.grant())
        with mock.patch.object(pair_admission, "PAIR_PROTOCOL_VERSION", 999):
            self.assertIsNone(self.grant())

    def test_non_identical_evidence_does_not_grant(self):
        pair_admission.PairLedger(self.directory).record_pass(
            pair_admission.pair_key(self.resident, self.requested),
            self.resident, self.requested,
            {"reference_sha256": "c" * 64, "swap_sha256": "d" * 64})
        self.assertIsNone(self.grant())


class PairRecordCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="dgxm-pair-cli-")
        self.resident = json.dumps(request("h3"))
        self.requested = json.dumps(request("ltx"))
        pinned = _pin_provenance()
        pinned.start()
        self.addCleanup(pinned.stop)

    def arguments(self, reference, swap):
        return ["--resident", self.resident, "--requested", self.requested,
                "--reference-sha", reference, "--swap-sha", swap,
                "--ledger-dir", self.directory]

    def test_mismatched_shas_are_refused(self):
        from dgx_monarch import pair_record

        with self.assertRaises(SystemExit) as caught:
            pair_record.main(self.arguments("a" * 64, "b" * 64))
        self.assertIn("nothing was recorded", str(caught.exception))
        self.assertEqual(pair_admission.PairLedger(self.directory).entries(), [])

    def test_equal_shas_record_a_pass(self):
        from dgx_monarch import pair_record

        self.assertEqual(pair_record.main(self.arguments("e" * 64, "e" * 64)), 0)
        entries = pair_admission.PairLedger(self.directory).entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["evidence"]["swap_sha256"], "e" * 64)

    def test_same_model_pair_is_refused(self):
        from dgx_monarch import pair_record

        with self.assertRaises(SystemExit):
            pair_record.main(["--resident", self.resident, "--requested", self.resident,
                              "--reference-sha", "f" * 64, "--swap-sha", "f" * 64,
                              "--ledger-dir", self.directory])


if __name__ == "__main__":
    unittest.main()
