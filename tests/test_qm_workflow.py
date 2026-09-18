import copy
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from streamlit.testing.v1 import AppTest

from qm_rules import evaluate, new_snapshot
from qm_store import QMStore


def complete_snapshot(document_id="document", count=2):
    snapshot = new_snapshot()
    snapshot["metadata"] = {k: "Documented test evidence" for k in snapshot["metadata"]}
    p = snapshot["protocol"]
    for field in ("name", "version", "authority", "uncertainty_rule", "exclusion_rule", "approved_by", "approval_reference"):
        p[field] = "Approved test protocol"
    p["planned_pairs"] = "\n".join(f"P-{n}" for n in range(count))
    p["minimum_pairs"] = count
    rows = []
    for n in range(count):
        for side in ("in", "out"):
            row = dict(sample_id=f"{side}-{n}", replicate_id="1", pair_id=f"P-{n}", side=side,
                       date="2026-01-01T09:00:00+01:00", received_at="2026-01-01T10:00:00+01:00",
                       analyzed_at="2026-01-01T11:00:00+01:00", location="Zulauf" if side == "in" else "Ablauf",
                       result_type="concentration" if side == "in" else "count", value=100000 if side == "in" else 1,
                       unit="/100 mL" if side == "in" else "count", volume_method="effective", effective_ml=100,
                       recovery_source="Validated SOP", document_id=document_id, source_ref=f"Page 1 row {n}",
                       reported_result="100000 /100 mL" if side == "in" else "1 count", source_verified=True,
                       qc_status="accepted", qc_evidence="Batch controls accepted", disposition="included", exclusion_reason="")
            rows.append(row)
    snapshot["rows"] = rows
    return snapshot


class RulesTests(unittest.TestCase):
    def test_counting_is_not_interpolated_quantile(self):
        snapshot = complete_snapshot(count=20)
        for row in snapshot["rows"][:4]:
            if row["side"] == "out":
                row["value"] = 10
        result = evaluate(snapshot, {"document"})
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["pairs_passed"], 18)
        self.assertEqual(result["decision"], "met")
        self.assertLess(np.quantile([p["guideline_lrv"] for p in result["pairs"]], .1), 5)

    def test_bad_metadata_qc_and_source_block(self):
        for field, value in (("date", "not-a-date"), ("date", "2026-01-01T09:00"),
                             ("location", "wrong"), ("qc_status", "unresolved"), ("qc_evidence", ""),
                             ("document_id", "unknown"), ("source_verified", False), ("source_ref", ""),
                             ("received_at", "2026-01-01T08:00+01:00"),
                             ("analyzed_at", "2026-01-03T11:00+01:00")):
            with self.subTest(field=field, value=value):
                snapshot = complete_snapshot()
                snapshot["rows"][0][field] = value
                self.assertEqual(evaluate(snapshot, {"document"})["decision"], "not_assessable")

    def test_censored_and_missing_not_silently_converted_or_dropped(self):
        for kind in ("missing", "<LOD", "<LOQ", "MPN", "concentration"):
            with self.subTest(kind=kind):
                snapshot = complete_snapshot()
                snapshot["rows"][1]["result_type"] = kind
                result = evaluate(snapshot, {"document"})
                self.assertEqual(result["rows_included"], 4)
                self.assertEqual(result["decision"], "not_assessable")

    def test_exclusions_need_reason_and_do_not_hide_unmatched_pairs(self):
        snapshot = complete_snapshot()
        snapshot["rows"][0]["disposition"] = "excluded"
        result = evaluate(snapshot, {"document"})
        self.assertEqual(result["rows_excluded"], 1)
        self.assertTrue(any("Ausschluss" in issue for issue in result["issues"]))
        snapshot["rows"][0]["exclusion_reason"] = "Damaged sample"
        self.assertTrue(any("vollständiges Paar" in issue for issue in evaluate(snapshot, {"document"})["issues"]))
        snapshot["rows"][1].update(disposition="excluded", exclusion_reason="Paired sample unavailable")
        snapshot["protocol"]["minimum_pairs"] = 1
        self.assertEqual(evaluate(snapshot, {"document"})["decision"], "met")

    def test_planned_pairs_and_protocol_required(self):
        for field, value in (("planned_pairs", "P-0\nP-1\nP-2"), ("approved_by", ""),
                             ("minimum_pairs", 3), ("required_percent", float("nan")), ("minimum_pairs", 1.2)):
            with self.subTest(field=field):
                snapshot = complete_snapshot()
                snapshot["protocol"][field] = value
                self.assertTrue(evaluate(snapshot, {"document"})["issues"])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = QMStore(self.temp.name)
        self.store.provision_user("author", "author password 123", "author")
        self.store.provision_user("reviewer", "reviewer password 123", "reviewer")
        self.author = self.store.login("author", "author password 123")
        self.reviewer = self.store.login("reviewer", "reviewer password 123")

    def ready(self):
        assessment = self.store.create(self.author, "Test assessment")
        document = self.store.add_document(self.author, assessment, 1, "report.txt", b"Original laboratory report")
        self.store.save(self.author, assessment, 2, complete_snapshot(document), "Verified transcription")
        snapshot = self.store.get(self.author, assessment)["snapshot"]
        for row in snapshot["rows"]:
            self.assertFalse(row["source_verified"])
            row["source_verified"] = True
        self.store.save(self.author, assessment, 3, snapshot, "Source verification recorded")
        return assessment

    def test_auth_permissions_and_optimistic_revision(self):
        assessment = self.ready()
        with self.assertRaises(PermissionError):
            self.store.save(self.reviewer, assessment, 4, complete_snapshot(), "Invalid actor")
        with self.assertRaises(ValueError):
            self.store.save(self.author, assessment, 2, complete_snapshot(), "Stale revision")
        with self.assertRaises(PermissionError):
            self.store.get("fake token", assessment)
        self.store.logout(self.author)
        with self.assertRaises(PermissionError):
            self.store.get(self.author, assessment)

    def test_source_check_resets_on_edit(self):
        assessment = self.ready()
        snapshot = self.store.get(self.author, assessment)["snapshot"]
        snapshot["rows"][1]["value"] = 100
        self.store.save(self.author, assessment, 4, snapshot, "Corrected transcription")
        record = self.store.get(self.author, assessment)
        self.assertFalse(record["snapshot"]["rows"][1]["source_verified"])
        with self.assertRaises(ValueError):
            self.store.submit(self.author, assessment, 5)

    def test_renaming_sample_cannot_carry_source_verification(self):
        assessment = self.ready()
        snapshot = self.store.get(self.author, assessment)["snapshot"]
        snapshot["rows"][0]["sample_id"] = "renamed"
        self.store.save(self.author, assessment, 4, snapshot, "Renamed sample")
        self.assertFalse(self.store.get(self.author, assessment)["snapshot"]["rows"][0]["source_verified"])

    def test_account_revocation_and_password_reset(self):
        self.store.disable_user("author")
        with self.assertRaises(PermissionError):
            self.store.actor(self.author)
        with self.assertRaises(ValueError):
            self.store.login("author", "author password 123")
        self.store.reset_password("author", "changed password 456")
        token = self.store.login("author", "changed password 456")
        self.assertEqual(self.store.actor(token)["role"], "author")

    def test_changed_software_blocks_release(self):
        from qm_store import software_record
        assessment = self.ready()
        self.store.submit(self.author, assessment, 4)
        changed = software_record()
        changed["sha256"]["qm_rules.py"] = "different-version"
        with patch("qm_store.software_record", return_value=changed), self.assertRaises(ValueError):
            self.store.review(self.reviewer, assessment, 4, True, "Review")
        self.assertEqual(self.store.get(self.author, assessment)["status"], "submitted")

    def test_backup_restores_and_will_not_overwrite(self):
        assessment = self.ready()
        with tempfile.TemporaryDirectory() as restored:
            path = os.path.join(restored, "qm.sqlite3")
            command = [sys.executable, "qm_store.py", "--directory", self.temp.name, "backup", path]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(QMStore(restored).get(self.author, assessment)["snapshot"],
                             self.store.get(self.author, assessment)["snapshot"])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_release_locks_and_export_survives_restart(self):
        assessment = self.ready()
        self.store.submit(self.author, assessment, 4)
        with self.assertRaises(PermissionError):
            self.store.review(self.author, assessment, 4, True, "Self approval")
        with self.assertRaises(ValueError):
            self.store.save(self.author, assessment, 4, complete_snapshot(), "Edit submitted")
        self.store.review(self.reviewer, assessment, 4, True, "Originals, exclusions and protocol reviewed")
        restarted = QMStore(self.temp.name)
        record = restarted.get(self.author, assessment)
        self.assertEqual(record["status"], "released")
        with self.assertRaises(ValueError):
            restarted.add_document(self.author, assessment, 4, "later.txt", b"Late evidence")
        archive = zipfile.ZipFile(io.BytesIO(restarted.export(self.author, assessment)))
        exported = json.loads(archive.read("assessment.json"))
        self.assertEqual(exported["submitted_result"]["evaluation"]["decision"], "met")
        self.assertIn("software/qm_rules.py", archive.namelist())
        self.assertEqual(len(json.loads(archive.read("revisions.json"))), 4)
        self.assertIn("report.html", archive.namelist())
        self.assertTrue(any(name.endswith("report.txt") for name in archive.namelist()))

    def test_negative_decision_can_be_released(self):
        assessment = self.ready()
        record = self.store.get(self.author, assessment)
        record["snapshot"]["protocol"]["target"] = 6
        self.store.save(self.author, assessment, 4, record["snapshot"], "Approved stricter criterion")
        self.store.submit(self.author, assessment, 5)
        self.store.review(self.reviewer, assessment, 5, True, "Reviewed failure to meet target")
        record = self.store.get(self.author, assessment)
        self.assertEqual(record["submitted_result"]["evaluation"]["decision"], "not_met")
        self.assertEqual(record["status"], "released")

    def test_return_and_amendment_preserve_previous_release(self):
        assessment = self.ready()
        self.store.submit(self.author, assessment, 4)
        self.store.review(self.reviewer, assessment, 4, False, "Please verify report revision")
        self.assertEqual(self.store.get(self.author, assessment)["status"], "draft")
        self.store.submit(self.author, assessment, 4)
        self.store.review(self.reviewer, assessment, 4, True, "Evidence accepted")
        amended = self.store.amend(self.author, assessment, "Laboratory report amendment")
        self.assertEqual(self.store.get(self.author, assessment)["status"], "released")
        record = self.store.get(self.author, amended)
        self.assertEqual(record["parent_id"], assessment)
        for row in record["snapshot"]["rows"]:
            self.assertFalse(row["source_verified"])
            row["source_verified"] = True
        self.store.save(self.author, amended, 1, record["snapshot"], "Originals reverified")
        self.store.submit(self.author, amended, 2)
        self.store.review(self.reviewer, amended, 2, True, "Amendment reviewed")
        self.assertEqual(self.store.get(self.author, assessment)["status"], "superseded")
        self.assertEqual(self.store.get(self.author, amended)["status"], "released")

    def test_database_protects_original_records(self):
        assessment = self.ready()
        for table in ("documents", "events", "revisions"):
            with self.subTest(table=table), self.assertRaises(sqlite3.IntegrityError):
                with self.store.connection() as db:
                    db.execute(f"DELETE FROM {table} WHERE assessment_id=?", (assessment,))

    def test_unresolved_data_cannot_be_submitted(self):
        assessment = self.store.create(self.author, "Incomplete assessment")
        with self.assertRaises(ValueError):
            self.store.submit(self.author, assessment, 1)

    def test_qm_ui_login_draft_and_review_render(self):
        assessment = self.ready()
        with patch.dict(os.environ, {"VALIDREUSE_QM_DIR": self.temp.name}):
            at = AppTest.from_file("app.py", default_timeout=30).run()
            at.sidebar.radio[0].set_value("QM-Bewertungen").run()
            self.assertFalse(at.exception)
            self.assertTrue(any(button.label == "Anmelden" for button in at.button))
            at.session_state["qm_token"] = self.author
            at.run()
            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertTrue(any(button.label == "Revision speichern" for button in at.button))
            next(w for w in at.text_input if w.label == "Grund für diese gespeicherte Revision").set_value("UI save verification")
            next(w for w in at.button if w.label == "Revision speichern").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(all(row["source_verified"] for row in self.store.get(self.author, assessment)["snapshot"]["rows"]))
            self.store.submit(self.author, assessment, 5)
            at.session_state["qm_token"] = self.reviewer
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(any(button.label == "Bewertung freigeben" for button in at.button))
            next(w for w in at.text_area if w.label.startswith("Fachlicher Prüfvermerk")).set_value("Originals and protocol reviewed")
            next(w for w in at.checkbox if w.label.startswith("Quellen, Genehmigungsnachweise")).check()
            next(w for w in at.button if w.label == "Bewertung freigeben").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(self.store.get(self.author, assessment)["status"], "released")
            at.session_state["qm_token"] = self.author
            at.run()
            next(w for w in at.text_input if w.label == "Grund für einen Nachtrag").set_value("Updated report")
            next(w for w in at.button if w.label == "Verknüpften Nachtrag erstellen").click().run()
            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertNotEqual(at.session_state["qm_open_id"], assessment)


if __name__ == "__main__":
    unittest.main()
