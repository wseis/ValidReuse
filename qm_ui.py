"""Streamlit interface for persistent, reviewed external-laboratory assessments."""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from qm_rules import QM_COLUMNS, evaluate, new_snapshot
from qm_store import QMStore
from sample_data import NUMERIC_COLUMNS
from ui_theme import inject_css

STATUS = {"draft": "Entwurf", "submitted": "Zur Prüfung", "released": "Freigegeben", "superseded": "Ersetzt"}
DECISION = {"met": "Kriterium erfüllt", "not_met": "Kriterium nicht erfüllt", "not_assessable": "Nicht beurteilbar"}
METADATA_LABELS = {
    "laboratory": "Labor / Standort", "report_number": "Prüfbericht-Nr.",
    "report_revision": "Revision / Ausgabedatum des Laborberichts", "method_in": "Prüfmethode Zulauf (mit Version)",
    "method_out": "Prüfmethode Ablauf (mit Version)",
    "accreditation_evidence": "Nachweis: Geltungsbereich / Status der Akkreditierung und ggf. Zulassung",
    "sampling_transport_evidence": "Nachweis: Probenahme, Konservierung und Transportbedingungen",
}
PROTOCOL_LABELS = {
    "name": "Prüfplan / SOP", "version": "Prüfplan-Version", "authority": "Anwendungszweck / Bescheid / Regelwerk",
    "in_location": "Erwartete Probenahmestelle Zulauf", "out_location": "Erwartete Probenahmestelle Ablauf",
    "uncertainty_rule": "Festgelegter Umgang mit Messunsicherheit und Nullwerten",
    "exclusion_rule": "Festgelegte Ausschlussregel und Umgang mit Fehlproben",
    "approved_by": "Prüfplan fachlich genehmigt durch", "approval_reference": "Nachweis der Prüfplan-Genehmigung",
}


def _frame(rows, side):
    frame = pd.DataFrame([r for r in rows if r.get("side") == side]).reindex(columns=QM_COLUMNS)
    for name in QM_COLUMNS:
        if name == "source_verified":
            frame[name] = frame[name].map(lambda x: x is True)
        elif name in NUMERIC_COLUMNS:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        else:
            frame[name] = frame[name].astype("string")
    return frame


def _editor(rows, side, key, documents):
    from volume_ui import LABELS
    config = {name: st.column_config.TextColumn(label) for name, label in LABELS.items() if name not in NUMERIC_COLUMNS}
    config.update({name: st.column_config.NumberColumn(LABELS.get(name, name)) for name in NUMERIC_COLUMNS})
    config.update({
        "_row_id": None, "side": None,
        "source_verified": st.column_config.CheckboxColumn("Originalabgleich bestätigt", default=False),
        "document_id": st.column_config.SelectboxColumn("Originaldokument-ID", options=[d["id"] for d in documents]),
        "source_ref": st.column_config.TextColumn("Fundstelle: Seite / Zeile"),
        "reported_result": st.column_config.TextColumn("Originalbefund (unverändert)"),
        "received_at": st.column_config.TextColumn("Probeneingang (ISO mit Zeitzone)"),
        "analyzed_at": st.column_config.TextColumn("Analysebeginn (ISO mit Zeitzone)"),
        "qc_status": st.column_config.SelectboxColumn("QC-Status", options=["unresolved", "accepted", "rejected"], default="unresolved"),
        "qc_evidence": st.column_config.TextColumn("QC-Nachweis / Charge / Laborbestätigung"),
        "disposition": st.column_config.SelectboxColumn("Verwendung", options=["included", "excluded"], default="included"),
        "exclusion_reason": st.column_config.TextColumn("Ausschlussgrund"),
    })
    order = ["sample_id", "replicate_id", "pair_id", "reported_result", "value", "unit", "result_type",
             "document_id", "source_ref", "source_verified", "date", "received_at", "analyzed_at", "location",
             "qc_status", "qc_evidence", "disposition", "exclusion_reason"]
    order += [name for name in QM_COLUMNS if name not in order and name not in ("side", "_row_id")]
    edited = st.data_editor(_frame(rows, side), key=key, column_order=order, column_config=config,
                            num_rows="dynamic", hide_index=True, width="stretch")
    # Drop only completely blank editor additions, never result_type=missing rows.
    result = []
    for row in json.loads(edited.to_json(orient="records")):
        if any(value not in (None, "", False) for name, value in row.items()
               if name not in ("_row_id", "side", "source_verified", "disposition", "qc_status")):
            row["side"] = side
            row["disposition"] = row.get("disposition") or "included"
            result.append(row)
    return result


def _documents(store, token, record, editable):
    assessment_id, revision = record["id"], record["revision"]
    documents = store.documents(token, assessment_id)
    st.markdown("#### Originalberichte und Nachweise")
    st.caption("PDF, XLSX, CSV oder TXT. Jede Datei wird unverändert mit Prüfsumme archiviert. "
               "Die Prüfsumme belegt die Unverändertheit; den Aussteller und Inhalt prüft der Bearbeiter.")
    if editable:
        with st.form(f"qm_document_{assessment_id}_{revision}", clear_on_submit=True):
            upload = st.file_uploader("Original / Nachweis", type=["pdf", "xlsx", "csv", "txt"])
            if st.form_submit_button("Original archivieren"):
                if upload is None:
                    st.error("Bitte eine Datei auswählen.")
                else:
                    store.add_document(token, assessment_id, revision, upload.name, upload.getvalue())
                    st.rerun()
    if not documents:
        st.info("Noch keine Originaldokumente hinterlegt.")
        return documents
    st.dataframe(pd.DataFrame(documents).rename(columns={"id": "Dokument-ID", "filename": "Dateiname"}), hide_index=True, width="stretch")
    selected = st.selectbox("Original ansehen / herunterladen", [d["id"] for d in documents],
                            format_func=lambda value: next(d["filename"] + " · " + value[:8] for d in documents if d["id"] == value),
                            key=f"qm_doc_choice_{assessment_id}")
    document = store.document(token, assessment_id, selected)
    st.download_button("Original herunterladen", document["content"], file_name=document["filename"], key=f"qm_doc_download_{assessment_id}")
    if editable and document["filename"].lower().endswith(".csv"):
        with st.form(f"qm_import_{assessment_id}_{revision}"):
            st.caption("CSV mit ValidReuse-Spaltennamen importieren. Bestehende Zeilen bleiben erhalten. "
                       "Fundstelle und ursprüngliche Schreibweise anschließend am Bericht prüfen.")
            side = st.selectbox("CSV enthält", ["in", "out"], format_func=lambda s: "Zulauf" if s == "in" else "Ablauf")
            separator = st.selectbox("Trennzeichen", [",", ";"])
            decimal = st.selectbox("Dezimalzeichen", [".", ","])
            if st.form_submit_button("CSV-Zeilen als ungeprüft ergänzen"):
                frame = pd.read_csv(io.BytesIO(document["content"]), sep=separator, decimal=decimal,
                                    dtype={k: str for k in ("sample_id", "replicate_id", "pair_id", "date")})
                if not {"sample_id", "result_type", "value", "unit"}.issubset(frame.columns) or set(frame.columns) - set(QM_COLUMNS):
                    raise ValueError("CSV benötigt bekannte ValidReuse-Spalten einschließlich sample_id, result_type, value, unit.")
                rows = json.loads(frame.to_json(orient="records"))
                for index, row in enumerate(rows, 2):
                    row.update(side=side, document_id=selected, source_ref=f"CSV-Datensatz {index - 1} (Kopfzeile ausgenommen)",
                               source_verified=False, disposition="included", qc_status="unresolved")
                    row.setdefault("replicate_id", "1")
                snapshot = record["snapshot"]
                snapshot["rows"].extend(rows)
                store.save(token, assessment_id, revision, snapshot, f"CSV-Import aus Dokument {selected}")
                st.rerun()
    return documents


def _draft_form(store, token, record, documents, parameters):
    snapshot = record["snapshot"]
    key = f"qm_edit_{record['id']}_{record['revision']}"
    with st.form(key):
        st.markdown("#### Bericht und Eignungsnachweise")
        metadata = {}
        for field, label in METADATA_LABELS.items():
            metadata[field] = st.text_input(label, value=snapshot["metadata"].get(field, ""))
        st.markdown("#### Versionierter Prüfplan")
        st.caption("Die folgenden Zahlen sind Platzhalter für Ihren genehmigten Prüfplan, keine allgemeinen "
                   "gesetzlichen Mindestanforderungen. Diese Version bewertet gepaarte Count-LRV mit Nullkonvention; "
                   "weitere Genehmigungsbedingungen werden durch die fachliche Prüfung abgedeckt.")
        protocol = dict(snapshot["protocol"])
        names = [p["name"] for p in parameters]
        protocol["indicator"] = st.selectbox("Indikator", names, index=names.index(protocol["indicator"]) if protocol["indicator"] in names else 0)
        for field, label in PROTOCOL_LABELS.items():
            protocol[field] = st.text_input(label, value=protocol.get(field, ""))
        columns = st.columns(3)
        protocol["target"] = columns[0].number_input("LRV-Zielwert", 0.0, 100.0, float(protocol["target"]), step=0.1)
        protocol["required_percent"] = columns[1].number_input("Erforderlicher Anteil (%)", 1, 100, int(protocol["required_percent"]))
        protocol["minimum_pairs"] = columns[2].number_input("Mindestzahl vollständiger Paare", 1, 100000, int(protocol["minimum_pairs"]))
        protocol["planned_pairs"] = st.text_area("Geplante Paar-IDs (eine pro Zeile)", value=protocol.get("planned_pairs", ""))
        protocol["max_holding_hours"] = st.number_input("Maximale Zeit Probenahme bis Analyse (h)", 0.0, 100000.0, float(protocol["max_holding_hours"]))
        protocol["max_pair_gap_hours"] = st.number_input("Maximaler Probenahmeabstand je Paar (h)", 0.0, 100000.0, float(protocol["max_pair_gap_hours"]))
        st.markdown("#### Übertragung und Prüfung der Einzelbefunde")
        st.caption("Zeiten mit Zeitzone eingeben, z.B. 2026-09-15T09:00+02:00. accepted bedeutet dokumentierte QC-Freigabe. "
                   "excluded benötigt eine Begründung. Bei Änderungen an einer gespeicherten Zeile wird ihr Originalabgleich "
                   "zurückgesetzt: zuerst speichern, dann erneut prüfen und bestätigen. Auch Fehlproben bleiben dokumentiert.")
        tabs = st.tabs(["Zulauf", "Ablauf"])
        rows = []
        for tab, side in zip(tabs, ("in", "out")):
            with tab:
                rows.extend(_editor(snapshot["rows"], side, f"{key}_{side}", documents))
        reason = st.text_input("Grund für diese gespeicherte Revision")
        draft = {"metadata": metadata, "protocol": protocol, "rows": rows}
        if st.form_submit_button("Revision speichern", type="primary"):
            store.save(token, record["id"], record["revision"], draft, reason)
            st.rerun()
    st.caption("Prüfergebnis und Einreichung beziehen sich ausschließlich auf die gespeicherte Revision. "
               "Formulareingaben zuerst mit „Revision speichern“ übernehmen.")


def _assessment(store, token, actor, record, parameters):
    st.subheader(record["title"])
    st.write(f"**{STATUS[record['status']]} · Revision {record['revision']}** · Bearbeiter: {record['owner']}")
    st.caption(f"Bewertungs-ID: {record['id']}" + (f" · Nachtrag zu: {record['parent_id']}" if record["parent_id"] else ""))
    editable = record["status"] == "draft" and actor["role"] == "author" and actor["username"] == record["owner"]
    documents = _documents(store, token, record, editable)
    if editable:
        _draft_form(store, token, record, documents, parameters)
    else:
        with st.expander("Gespeicherter Bericht und Prüfplan", expanded=True):
            st.json({"metadata": record["snapshot"]["metadata"], "protocol": record["snapshot"]["protocol"]})
        st.dataframe(pd.DataFrame(record["snapshot"]["rows"]), hide_index=True, width="stretch")
    result = (record["submitted_result"]["evaluation"] if record["submitted_result"] else
              evaluate(record["snapshot"], {d["id"] for d in documents}))
    st.markdown("#### Prüfung der gespeicherten Revision")
    columns = st.columns(3)
    columns[0].metric("Erfasste Analysen", result["rows_total"])
    columns[1].metric("Ausgeschlossene Analysen", result["rows_excluded"])
    columns[2].metric("Vollständige Paare", result["pairs_evaluated"])
    st.write(f"**Daten geeignet:** {'Ja' if result['suitability'] == 'accepted' else 'Offene Prüfpunkte'}")
    st.write(f"**Festgelegtes LRV-Kriterium:** {DECISION[result['decision']]}")
    st.caption("Die Freigabe dokumentiert eine fachlich geprüfte Bewertung. Sie kann auch ein nicht erfülltes "
               "LRV-Kriterium feststellen und ist keine automatische behördliche Genehmigung.")
    if result["pairs_evaluated"]:
        st.write(f"{result['pairs_passed']} / {result['pairs_evaluated']} Paare erreichen den Zielwert "
                 f"({result['percent_passed']:.2f} %). Entscheidung anhand der Anzahl, ohne Quantilinterpolation.")
        st.dataframe(pd.DataFrame(result["pairs"]), hide_index=True, width="stretch")
    if result["issues"]:
        st.dataframe(pd.DataFrame({"Offene Prüfpunkte (blockierend)": result["issues"]}), hide_index=True, width="stretch")
    if editable:
        with st.form(f"qm_submit_{record['id']}_{record['revision']}"):
            confirmed = st.checkbox("Ich reiche die gespeicherte Revision mit allen Originalen, Nachweisen und Ausschlüssen ein.")
            if st.form_submit_button("Gespeicherte Revision zur Prüfung einreichen", disabled=bool(result["issues"])):
                if not confirmed:
                    st.error("Bitte die Einreichung der gespeicherten Revision bestätigen.")
                else:
                    store.submit(token, record["id"], record["revision"])
                    st.rerun()
    if record["status"] == "submitted" and actor["role"] == "reviewer":
        with st.form(f"qm_review_{record['id']}_{record['revision']}"):
            comment = st.text_area("Fachlicher Prüfvermerk (Originalabgleich, Prüfplan, QC, Unsicherheit, Ausschlüsse)")
            confirmed = st.checkbox("Quellen, Genehmigungsnachweise, Datenübertragung, QC und Ausschlüsse fachlich geprüft.")
            release = st.form_submit_button("Bewertung freigeben", type="primary")
            returned = st.form_submit_button("Zur Überarbeitung zurückgeben")
            if release or returned:
                if release and not confirmed:
                    st.error("Bitte den Umfang der fachlichen Prüfung bestätigen.")
                else:
                    store.review(token, record["id"], record["revision"], release, comment)
                    st.rerun()
    if record["status"] == "released" and actor["role"] == "author":
        with st.form(f"qm_amend_{record['id']}"):
            reason = st.text_input("Grund für einen Nachtrag")
            if st.form_submit_button("Verknüpften Nachtrag erstellen"):
                new_id = store.amend(token, record["id"], reason)
                st.session_state["qm_next_id"] = new_id
                st.rerun()
    with st.expander("Revisions- und Prüfarchiv"):
        history = store.history(token, record["id"])
        st.dataframe(pd.DataFrame([{**event, "detail": json.dumps(event["detail"], ensure_ascii=False)} for event in history["events"]]),
                     hide_index=True, width="stretch")
        revision = st.selectbox("Gespeicherte Revision nachlesen", [r["revision"] for r in history["revisions"]],
                               key=f"qm_history_{record['id']}")
        st.json(json.loads(next(r["snapshot"] for r in history["revisions"] if r["revision"] == revision)), expanded=False)
        st.caption("ZIP enthält Originaldateien, alle gespeicherten Revisionen, das Auditprotokoll und bei "
                   "eingereichten Bewertungen den archivierten Rechenstand. Entwürfe sind als solche gekennzeichnet.")
        if st.button("Prüfarchiv vorbereiten", key=f"qm_export_{record['id']}"):
            payload = store.export(token, record["id"])
            st.download_button("Prüfarchiv herunterladen", payload,
                               file_name=f"validreuse_qm_{record['id']}_r{record['revision']}_{record['status']}.zip", mime="application/zip")


def render_qm_workflow(parameters):
    inject_css()
    st.title("ValidReuse · QM-Bewertungen")
    st.write("Originalbefunde sichern, Eignung prüfen und Bewertungen nachvollziehbar freigeben.")
    store = QMStore()
    token = st.session_state.get("qm_token")
    if token:
        try:
            actor = store.actor(token)
        except PermissionError:
            st.session_state.pop("qm_token", None)
            token = None
    if not token:
        with st.form("qm_login", clear_on_submit=True):
            username = st.text_input("Benutzername")
            password = st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden"):
                try:
                    st.session_state["qm_token"] = store.login(username, password)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        st.info("Persönliche Bearbeiter- und Prüferkonten werden durch die lokale Administration eingerichtet. "
                "Einrichtung und Betriebsgrenzen stehen in QM_WORKFLOW.md.")
        return
    with st.sidebar:
        st.write(f"Angemeldet: {actor['username']} · {'Bearbeiter' if actor['role'] == 'author' else 'Prüfer'}")
        if st.button("QM abmelden"):
            store.logout(token)
            st.session_state.pop("qm_token", None)
            st.rerun()
    try:
        if actor["role"] == "author":
            with st.expander("Neue Bewertung anlegen"):
                with st.form("qm_create", clear_on_submit=True):
                    title = st.text_input("Bewertungstitel / Projekt")
                    indicator = st.selectbox("Indikator der neuen Bewertung", [p["name"] for p in parameters])
                    if st.form_submit_button("Entwurf anlegen"):
                        parameter = next(p for p in parameters if p["name"] == indicator)
                        assessment_id = store.create(token, title, new_snapshot(indicator, parameter["target"]))
                        st.session_state["qm_next_id"] = assessment_id
                        st.rerun()
        records = store.list(token)
        if not records:
            st.info("Noch keine QM-Bewertungen vorhanden.")
            return
        ids = [r["id"] for r in records]
        next_id = st.session_state.pop("qm_next_id", None)
        if next_id in ids:
            st.session_state["qm_open_id"] = next_id
        if st.session_state.get("qm_open_id") not in ids:
            st.session_state["qm_open_id"] = ids[0]
        choices = {r["id"]: f"{r['title']} · {STATUS[r['status']]} · {r['id'][:8]}" for r in records}
        assessment_id = st.selectbox("Bewertung öffnen", ids, format_func=choices.get, key="qm_open_id")
        _assessment(store, token, actor, store.get(token, assessment_id), parameters)
    except (ValueError, PermissionError, OSError, pd.errors.ParserError) as exc:
        st.error(str(exc))
