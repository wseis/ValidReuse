"""Deterministic suitability checks for the first, count-based QM protocol.

This protocol counts passing paired LRVs. It does not claim to implement every
condition of a regulation or the alternative Bayesian population estimand.
"""
from __future__ import annotations

from datetime import datetime
import math

import pandas as pd

from sample_data import COLUMNS, guideline_lrv_pairs, validate_table

RULE_VERSION = "paired-count-lrv/1"
QM_COLUMNS = COLUMNS + [
    "_row_id", "side", "document_id", "source_ref", "reported_result", "source_verified",
    "received_at", "analyzed_at", "qc_status", "qc_evidence",
    "disposition", "exclusion_reason",
]


def new_snapshot(indicator="E. coli", target=5.0):
    return {
        "metadata": {key: "" for key in (
            "laboratory", "report_number", "report_revision", "method_in", "method_out",
            "accreditation_evidence", "sampling_transport_evidence",
        )},
        "protocol": {
            "name": "", "version": "", "authority": "", "indicator": indicator,
            "target": target, "required_percent": 90, "minimum_pairs": 1,
            "planned_pairs": "", "in_location": "Zulauf", "out_location": "Ablauf",
            "max_holding_hours": 24.0, "max_pair_gap_hours": 24.0,
            "uncertainty_rule": "", "exclusion_rule": "",
            "approved_by": "", "approval_reference": "",
        },
        "rows": [],
    }


def text(value):
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def timestamp(value):
    value = datetime.fromisoformat(text(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Zeitzone fehlt")
    return value


def evaluate(snapshot, document_ids):
    """Return explicit blockers and completeness counts; never silently drop rows."""
    issues, included, excluded = [], [], []
    metadata, protocol = snapshot.get("metadata", {}), snapshot.get("protocol", {})
    for field in new_snapshot()["metadata"]:
        if not text(metadata.get(field)):
            issues.append(f"Bericht: Pflichtangabe {field} fehlt.")
    for field in ("name", "version", "authority", "indicator", "in_location", "out_location",
                  "uncertainty_rule", "exclusion_rule", "approved_by", "approval_reference"):
        if not text(protocol.get(field)):
            issues.append(f"Prüfplan: Pflichtangabe {field} fehlt.")
    numbers = {}
    for field, low, high, integral in (
        ("target", 0, 100, False), ("required_percent", 1, 100, True),
        ("minimum_pairs", 1, 100000, True), ("max_holding_hours", 0, 100000, False),
        ("max_pair_gap_hours", 0, 100000, False),
    ):
        try:
            number = float(protocol.get(field))
            if not math.isfinite(number) or not low <= number <= high or (integral and not number.is_integer()):
                raise ValueError()
            numbers[field] = number
        except (TypeError, ValueError):
            issues.append(f"Prüfplan: {field} ist ungültig.")
    planned = [p.strip() for p in text(protocol.get("planned_pairs")).splitlines() if p.strip()]
    if not planned or len(planned) != len(set(planned)):
        issues.append("Prüfplan: eindeutige geplante Paar-IDs zeilenweise angeben.")
    if protocol.get("in_location") == protocol.get("out_location"):
        issues.append("Prüfplan: Zu- und Ablauf benötigen unterschiedliche Probenahmestellen.")

    identities, observed_pairs = set(), set()
    collected = {"in": {}, "out": {}}
    for index, row in enumerate(snapshot.get("rows", []), 1):
        label = f"Zeile {index} ({text(row.get('sample_id'))})"
        side = row.get("side")
        for field in ("sample_id", "replicate_id", "pair_id", "document_id", "source_ref", "reported_result"):
            if not text(row.get(field)):
                issues.append(f"{label}: {field} fehlt.")
        if row.get("document_id") not in document_ids:
            issues.append(f"{label}: Originaldokument nicht hinterlegt.")
        if row.get("source_verified") is not True:
            issues.append(f"{label}: Übertragung vom Original noch nicht geprüft.")
        identity = (side, text(row.get("sample_id")), text(row.get("replicate_id")))
        if identity in identities:
            issues.append(f"{label}: doppelte Analyse-ID.")
        identities.add(identity)
        observed_pairs.add(text(row.get("pair_id")))
        if text(row.get("pair_id")) not in planned:
            issues.append(f"{label}: Paar-ID steht nicht im Prüfplan.")
        if side not in ("in", "out"):
            issues.append(f"{label}: Seite muss in oder out sein.")
        disposition = row.get("disposition", "included")
        if disposition == "excluded":
            excluded.append(index)
            if not text(row.get("exclusion_reason")):
                issues.append(f"{label}: Ausschluss benötigt eine Begründung.")
            continue
        if disposition != "included":
            issues.append(f"{label}: ungültiger Einschlussstatus.")
        included.append(row)
        if row.get("qc_status") != "accepted" or not text(row.get("qc_evidence")):
            issues.append(f"{label}: QC-Freigabe und Nachweis fehlen oder sind nicht akzeptiert.")
        if side in ("in", "out") and row.get("location") != protocol.get(f"{side}_location"):
            issues.append(f"{label}: Probenahmestelle stimmt nicht mit Prüfplan überein.")
        try:
            sampled, received, analyzed = [timestamp(row.get(k)) for k in ("date", "received_at", "analyzed_at")]
            if not sampled <= received <= analyzed:
                issues.append(f"{label}: Reihenfolge Probenahme → Eingang → Analyse ist ungültig.")
            if (analyzed - sampled).total_seconds() / 3600 > numbers.get("max_holding_hours", 0):
                issues.append(f"{label}: maximale Aufbewahrungszeit überschritten.")
            if side in collected:
                collected[side].setdefault(text(row.get("pair_id")), []).append(sampled)
        except (TypeError, ValueError, OverflowError):
            issues.append(f"{label}: ISO-Zeitpunkte mit Zeitzone erforderlich, z.B. 2026-09-15T09:00+02:00.")
        if row.get("result_type") == "missing":
            issues.append(f"{label}: missing muss ausdrücklich und begründet ausgeschlossen werden.")
    for missing_pair in sorted(set(planned) - observed_pairs):
        issues.append(f"Geplantes Paar {missing_pair}: keine Daten oder dokumentierten Fehlproben vorhanden.")
    tables = {}
    for side in ("in", "out"):
        try:
            tables[side] = validate_table(pd.DataFrame([r for r in included if r.get("side") == side]), side)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f"{side}: {exc}")
    pairs = pd.DataFrame()
    if len(tables) == 2:
        try:
            pairs = guideline_lrv_pairs(tables["in"], tables["out"])
            paired_ids = set() if pairs.empty else set(pairs.pair_id)
            for row in included:
                if text(row.get("pair_id")) not in paired_ids:
                    issues.append(f"Probe {text(row.get('sample_id'))}: eingeschlossen, aber ohne vollständiges Paar.")
            for pair_id in paired_ids:
                times = collected["in"].get(pair_id, []) + collected["out"].get(pair_id, [])
                if times and (max(times) - min(times)).total_seconds() / 3600 > numbers.get("max_pair_gap_hours", 0):
                    issues.append(f"Paar {pair_id}: maximaler Probenahmeabstand überschritten.")
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f"Paarung: {exc}")
    if len(pairs) < numbers.get("minimum_pairs", 1):
        issues.append("Zu wenige vollständige Paare für den festgelegten Prüfplan.")
    passed = 0 if pairs.empty else int((pairs.guideline_lrv >= numbers.get("target", float('inf'))).sum())
    total = len(pairs)
    # Integer arithmetic avoids interpolation and floating point percentage boundaries.
    met = passed * 100 >= int(numbers.get("required_percent", 100)) * total if total else False
    return {
        "rule_version": RULE_VERSION, "issues": list(dict.fromkeys(issues)),
        "suitability": "blocked" if issues else "accepted",
        "decision": "not_assessable" if issues else ("met" if met else "not_met"),
        "rows_total": len(snapshot.get("rows", [])), "rows_included": len(included),
        "rows_excluded": len(excluded), "excluded_rows": excluded,
        "planned_pairs": len(planned), "pairs_evaluated": total, "pairs_passed": passed,
        "percent_passed": passed / total * 100 if total else None,
        "pairs": [] if pairs.empty else pairs.to_dict(orient="records"),
    }
