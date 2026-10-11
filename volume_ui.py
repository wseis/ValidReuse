"""Streamlit workflow for raw effluent counts with individual sample volumes."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sample_data import (COLUMNS, RESULT_TYPES, VOLUME_METHODS, empty_table, example_tables,
                         validate_table, model_arrays, editor_table, guideline_lrv_pairs)
from pymc_bayesian import PYMC_AVAILABLE, pymc_q10_lrv_batch
from ui_theme import hero, inject_css, section_head

FAMILIES = {"Poisson-Lognormal": "poisson_lognormal",
            "Negativbinomial (Poisson-Gamma)": "negative_binomial",
            "Bedingt · Poisson-Lognormal": "conditional_poisson_lognormal",
            "Bedingt · Negativbinomial": "conditional_negative_binomial"}
UNPAIRED = [label for label, family in FAMILIES.items() if not family.startswith("conditional_")]
SLOPE_LABELS = {"b frei geschätzt": "free", "b = 1 (vollständig gepaart)": "unit",
                "b = 0 (unabhängig)": "zero"}
TEMPLATE_PATH = Path(__file__).with_name("assets") / "ValidReuse_Datenvorlage.xlsx"
LABELS = {
    "sample_id": "Proben-ID", "replicate_id": "Replikat-ID", "date": "Datum / Uhrzeit",
    "location": "Probenahmestelle", "pair_id": "Paar-ID (Dokumentation)", "result_type": "Ergebnistyp",
    "value": "Messwert (Zahl)", "unit": "Einheit", "volume_method": "Art der Volumenangabe",
    "filtered_ml": "Filtriert (mL)", "eluate_ml": "Gesamtes Eluat (mL)", "assayed_ml": "Analysiert (mL)",
    "dilution_factor": "Verdünnungsfaktor", "recovery": "Wiederfindung (0–1)",
    "effective_ml": "Effektives Volumen (mL)", "recovery_source": "Quelle / Annahme", "notes": "Notizen",
}
FIELD_HELP = {
    "sample_id": "Eindeutige ID der Umweltprobe. Technische Replikate derselben Probe erhalten dieselbe Proben-ID.",
    "replicate_id": "Kennzeichnet einzelne Analysen derselben Umweltprobe, zum Beispiel 1 und 2.",
    "date": "Datum oder Zeitpunkt der Probenahme; ISO-Format wie 2026-01-12 wird empfohlen.",
    "location": "Bezeichnung der Probenahmestelle, zum Beispiel Zulauf oder Ablauf Kläranlage.",
    "pair_id": "Gemeinsamer Schlüssel für zusammengehörige Zu- und Ablaufproben. Die Zeilenposition erzeugt keine Paarung.",
    "result_type": "Art des Ergebnisses. Für das Modell wird im Zulauf concentration und im Ablauf count verwendet.",
    "value": "Numerischer Messwert. Ablauf-Counts müssen ganze Zahlen sein; eine beobachtete Null bleibt 0.",
    "unit": "Einheit des Messwertes. Im Ablauf ist die Einheit count; im Zulauf eine Konzentrationseinheit.",
    "volume_method": "concentrated: Anreicherung; direct: direktes Aliquot; effective: bereits berechnetes effektives Volumen.",
    "filtered_ml": "Ursprünglich filtriertes Wasservolumen in mL; nur bei concentrated.",
    "eluate_ml": "Gesamtvolumen des Eluats in mL; nur bei concentrated.",
    "assayed_ml": "Tatsächlich analysiertes Volumen in mL; bei concentrated oder direct.",
    "dilution_factor": "1 ohne Verdünnung, 10 bei einer Verdünnung von 1:10.",
    "recovery": "Wiederfindung als Anteil von größer 0 bis 1. Der Wert 0,5 bedeutet 50 %.",
    "effective_ml": "Bereits extern berechnetes effektives Originalwasservolumen in mL; nur bei effective.",
    "recovery_source": "Quelle oder begründete Annahme für Wiederfindung beziehungsweise effektives Volumen.",
    "notes": "Freie Hinweise zur Probe oder zur Analyse.",
}


STEP_TITLES = ["Daten eingeben", "Eingaben prüfen", "Messbegrenzte LRV", "Modellbasierte Auswertung"]

def _stepper(states):
    """Progress rail across the four steps; states are done, current or locked."""
    parts = []
    for number, (title, state) in enumerate(zip(STEP_TITLES, states), start=1):
        dot = {"done": "✓", "locked": "🔒"}.get(state, str(number))
        parts.append(f'<div class="vr-step {state}"><span class="vr-dot">{dot}</span>{title}</div>')
        if number < len(STEP_TITLES):
            parts.append('<div class="vr-step-sep"></div>')
    st.markdown(f'<div class="vr-stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def _volume_span(volumes):
    """One value when every analysis shares an exposure, otherwise the range."""
    low, high = float(volumes.min()), float(volumes.max())
    return f"{low:g}" if low == high else f"{low:g}–{high:g}"


def _sidebar_panel(indicator, pid, inflow, outflow, guideline, errors):
    """At-a-glance state plus the percentile that steps 3 and 4 share."""
    rows = [("Indikator", indicator), ("Zulaufproben", str(len(inflow)))]
    if outflow.empty:
        rows.append(("Ablaufproben", "0"))
    else:
        rows += [
            ("Ablaufproben", str(int(outflow.sample_id.nunique()))),
            ("Analysen", str(len(outflow))),
            ("Nullzählungen", str(int((outflow.value == 0).sum()))),
            ("Effektiv je Analyse", f"{_volume_span(outflow.effective_volume_ml)} mL"),
            ("Bezugsfaktor", _volume_span(outflow.exposure)),
        ]
    rows.append(("Gepaarte LRV", str(len(guideline))))

    if errors:
        state = '<span class="vr-bad">Fehler</span>'
    elif inflow.empty and outflow.empty:
        state = '<span class="vr-idle">keine Daten</span>'
    else:
        state = '<span class="vr-ok">formal plausibel</span>'
    rows.append(("Eingabeprüfung", state))

    body = "".join(f'<div class="vr-row"><span>{label}</span><b>{value}</b></div>' for label, value in rows)
    with st.sidebar:
        st.markdown("#### Datenstatus")
        st.markdown(f'<div class="vr-side">{body}</div>', unsafe_allow_html=True)
        st.markdown("#### Auswertung")
        return st.slider("Populationsperzentil (%)", 1, 50, 10, key=f"quantile_{pid}",
                         help="10 bedeutet q10: Der Wert, den 10 % der betrachteten Population unterschreiten. "
                              "Gilt für die Leitfaden-LRV und für die modellbasierte Auswertung.")


def _read_excel_template(content):
    """Read the two data sheets from the documented ValidReuse workbook."""
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=2, dtype={
        "sample_id": str, "replicate_id": str, "pair_id": str, "date": str,
    })
    missing = {"Zulauf", "Ablauf"} - set(sheets)
    if missing:
        raise ValueError("Die Excel-Datei benötigt die Tabellenblätter 'Zulauf' und 'Ablauf'.")
    result = []
    for sheet_name in ["Zulauf", "Ablauf"]:
        table = sheets[sheet_name].dropna(how="all")
        unknown = set(table.columns) - set(COLUMNS)
        required = {"sample_id", "result_type", "value", "unit"}
        if unknown or not required.issubset(table.columns):
            raise ValueError(f"Im Blatt '{sheet_name}' wurden Spalten verändert oder Pflichtspalten entfernt.")
        result.append(editor_table(table))
    return tuple(result)


def _excel_controls(parameter_id):
    """Offer the self-documenting workbook and import both sheets together."""
    st.markdown("**Excel-Vorlage: herunterladen, ausfüllen, wieder importieren**")
    st.caption("Die Vorlage enthält ein eigenes Blatt **Anleitung**, erklärte Beispielzeilen und getrennte Blätter für Zu- und Ablauf. Die englischen Spaltennamen dienen dem sicheren Import und dürfen nicht geändert werden.")
    left, right = st.columns(2, vertical_alignment="center")
    if TEMPLATE_PATH.exists():
        left.download_button(
            "Excel-Vorlage herunterladen",
            TEMPLATE_PATH.read_bytes(),
            file_name="ValidReuse_Datenvorlage.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_download_{parameter_id}",
            width="stretch",
            help="Arbeitsmappe mit Anleitung, Spaltenlexikon und Beispielen für beide Probenahmestellen.",
        )
    upload = right.file_uploader(
        "Ausgefüllte Excel-Vorlage importieren",
        type="xlsx",
        key=f"xlsx_upload_{parameter_id}",
        help="ValidReuse übernimmt die Blätter 'Zulauf' und 'Ablauf' gemeinsam.",
    )
    if upload is not None and st.button("Excel-Daten übernehmen", key=f"xlsx_import_{parameter_id}"):
        try:
            inflow, outflow = _read_excel_template(upload.getvalue())
            st.session_state[f"vol_{parameter_id}_in"] = inflow
            st.session_state[f"vol_{parameter_id}_out"] = outflow
            st.session_state[f"vol_version_{parameter_id}"] = st.session_state.get(f"vol_version_{parameter_id}", 0) + 1
            st.session_state.pop(f"vol_result_{parameter_id}", None)
            st.success("Zu- und Ablaufdaten wurden aus der Excel-Datei übernommen.")
            st.rerun()
        except (ValueError, ImportError, OSError) as exc:
            st.error(f"Excel-Import fehlgeschlagen: {exc}")


def _reset(parameter_id, example):
    a, b = example_tables() if example else (empty_table(), empty_table())
    for side, table in [("in",a),("out",b)]:
        st.session_state[f"vol_{parameter_id}_{side}"] = table
    st.session_state[f"vol_version_{parameter_id}"] = st.session_state.get(f"vol_version_{parameter_id}",0)+1
    st.session_state.pop(f"vol_result_{parameter_id}",None)


def _editor(side, parameter_id):
    key = f"vol_{parameter_id}_{side}"
    if key not in st.session_state:
        st.session_state[key] = empty_table()
    if side == "in":
        st.info("**Pflicht für die Auswertung:** Proben-ID, positiver Messwert, Ergebnistyp `concentration` und passende Konzentrationseinheit. Für die gepaarte Leitfaden-LRV zusätzlich eine Paar-ID eintragen.")
    else:
        st.info("**Pflicht für die Auswertung:** Proben-ID, ganzer Count einschließlich 0, Ergebnistyp `count`, Einheit `count`, Art der Volumenangabe und die dazugehörigen Volumenfelder. Die Quelle oder Annahme ebenfalls dokumentieren.")
    with st.expander("Spalten in dieser Tabelle erklärt"):
        visible = (["sample_id", "value", "unit", "result_type", "date", "location", "pair_id", "notes"] if side == "in" else
                   ["sample_id", "replicate_id", "value", "volume_method", "filtered_ml", "eluate_ml", "assayed_ml",
                    "dilution_factor", "recovery", "effective_ml", "recovery_source", "result_type", "unit", "pair_id", "notes"])
        st.dataframe(pd.DataFrame({"Spalte": [LABELS[x] for x in visible], "Bedeutung": [FIELD_HELP[x] for x in visible]}),
                     hide_index=True, width="stretch")
    with st.expander("Nur dieses Tabellenblatt als CSV importieren"):
        upload = st.file_uploader("Optional: dieses Tabellenblatt als CSV importieren", type="csv", key=f"upload_{parameter_id}_{side}",
                                  label_visibility="collapsed",
                                  help="Für den gemeinsamen Import von Zu- und Ablauf verwenden Sie oben die Excel-Vorlage.")
    if upload is not None and st.button("CSV übernehmen", key=f"import_{parameter_id}_{side}"):
        try:
            table = pd.read_csv(io.BytesIO(upload.getvalue()), dtype={k:str for k in ["sample_id","replicate_id","pair_id","date"]})
            unknown = set(table.columns) - set(COLUMNS)
            required = {"sample_id", "result_type", "value", "unit"}
            if unknown or not required.issubset(table.columns):
                raise ValueError("Bitte die CSV-Vorlage verwenden; unbekannte oder fehlende Spalten.")
            st.session_state[key] = editor_table(table)
            st.session_state[f"vol_version_{parameter_id}"] = st.session_state.get(f"vol_version_{parameter_id}",0)+1
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Import fehlgeschlagen: {exc}")
    config = {name:st.column_config.TextColumn(label, help=FIELD_HELP[name]) for name,label in LABELS.items()}
    for name in ["value","filtered_ml","eluate_ml","assayed_ml","dilution_factor","recovery","effective_ml"]:
        config[name] = st.column_config.NumberColumn(LABELS[name], help=FIELD_HELP[name], min_value=0., format="%.6g")
    config["result_type"] = st.column_config.SelectboxColumn(LABELS["result_type"], help=FIELD_HELP["result_type"], options=RESULT_TYPES, default="concentration" if side == "in" else "count")
    config["unit"] = st.column_config.SelectboxColumn(LABELS["unit"], help=FIELD_HELP["unit"], options=["/100 mL","/mL","/L","count"], default="/100 mL" if side == "in" else "count")
    config["volume_method"] = st.column_config.SelectboxColumn(LABELS["volume_method"], help=FIELD_HELP["volume_method"], options=VOLUME_METHODS)
    config["replicate_id"] = st.column_config.TextColumn(LABELS["replicate_id"], help=FIELD_HELP["replicate_id"], default="1")
    columns = (["sample_id", "value", "volume_method", "filtered_ml", "eluate_ml", "assayed_ml",
                "dilution_factor", "recovery", "effective_ml", "recovery_source", "replicate_id",
                "result_type", "unit", "date", "location", "pair_id", "notes"] if side == "out" else
               ["sample_id", "value", "unit", "result_type", "date", "location", "pair_id", "notes"])
    widget_key = f"vol_editor_{parameter_id}_{side}_{st.session_state.get(f'vol_version_{parameter_id}',0)}"
    base_key = f"{key}_editor_base"
    if widget_key not in st.session_state:
        st.session_state[base_key] = editor_table(st.session_state[key])
    table = st.data_editor(st.session_state[base_key], column_order=columns, column_config=config,
                           hide_index=True, num_rows="dynamic", width="stretch",
                           key=widget_key)
    st.session_state[key] = table
    st.download_button("Dieses Tabellenblatt als CSV herunterladen", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{parameter_id}_{side}.csv", mime="text/csv", key=f"csv_{parameter_id}_{side}")
    return table


def _fingerprint(in_table, out_table, settings):
    content = in_table.to_csv(index=False)+out_table.to_csv(index=False)+json.dumps(settings,sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def render_volume_workflow(parameters):
    inject_css()
    hero("ValidReuse · Counts und Probenvolumina",
         "Hier werten Sie Rohzählungen im Ablauf zusammen mit dem tatsächlich untersuchten Volumen "
         "jeder Probe aus. Das berücksichtigt, dass bei großen und wechselnden Probenvolumina derselbe "
         "Count unterschiedliche Konzentrationen bedeutet.")

    # Filled in once the editors below have been validated.
    stepper_slot = st.empty()

    with st.container(border=True, key="vr_step1"):
        section_head(1, "Indikator wählen und Daten eingeben")
        # The indicator picker must stay the first selectbox in render order.
        selected_indicator = st.selectbox("Indikator", [p["name"] for p in parameters],
                                          help="Für jeden Indikator werden Eingaben und Ergebnisse getrennt gespeichert.")
        parameter = next(p for p in parameters if p["name"] == selected_indicator)
        pid = parameter["id"]
        _excel_controls(pid)
        c1,c2 = st.columns(2)
        c1.button("Beispieldaten in der Oberfläche laden", on_click=_reset, args=(pid,True), width="stretch",
                  help="Ersetzt die aktuellen Eingaben durch ein vollständiges Beispiel.")
        c2.button("Eingaben leeren", on_click=_reset, args=(pid,False), width="stretch",
                  help="Entfernt die aktuellen Tabellenwerte für diesen Indikator.")
        with st.expander("Welche Eingabeart passt zu meiner Ablaufprobe?"):
            st.markdown("""**Zulauf:** positive Konzentrationen; Dezimalwerte sind erlaubt. **Ablauf:** ganze CFU/PFU-Rohzählungen, einschließlich 0.

- **concentrated – Probe angereichert:** Verwenden, wenn Wasser filtriert oder anderweitig angereichert und nur ein Teil des Eluats analysiert wurde.
- **direct – direktes Aliquot:** Verwenden, wenn ein Wasseraliquot ohne vorherige Anreicherung direkt analysiert wurde.
- **effective – Volumen schon berechnet:** Verwenden, wenn das effektive Originalwasservolumen bereits nachvollziehbar außerhalb der App berechnet wurde.

Alle Volumina in **mL**. Verdünnungsfaktor 10 bedeutet 1:10; ohne Verdünnung 1. Wiederfindung 0,5 bedeutet 50 %. Eine Quelle oder ausdrückliche Annahme ist erforderlich.

15 000 mL filtriert, 400 mL Eluat, 100 mL analysiert und 50 % Wiederfindung ergeben **1 875 mL effektives Volumen** (Leitfaden Gl. 4-1 und 4-2). Daraus folgt der **Bezugsfaktor 18,75**: Die Zulaufkonzentration wird für den Vergleich mit 18,75 multipliziert (Gl. 4-3), nicht der Ablauf heruntergerechnet. Die Anreicherung macht so rund **1,3 log zusätzliche Reduktion** nachweisbar — auch dann, wenn im Ablauf null gezählt wird.

**Technische Replikate:** gleiche Proben-ID, unterschiedliche Replikat-ID und getrennte analysierte Aliquots. Sie teilen im Modell eine Konzentration. Fehlende Ergebnisse als `missing` dokumentieren; `<LOD`, `<LOQ`, MPN und Konzentrationen im Ablauf werden gespeichert, aber für diese Count-Auswertung nicht automatisch umgerechnet.""")
        t1,t2 = st.tabs(["Zulaufkonzentrationen", "Ablauf: Counts und Volumina"])
        with t1:
            raw_in = _editor("in",pid)
        with t2:
            raw_out = _editor("out",pid)

    # Validate once, up front: every later step is gated on the outcome.
    errors = []
    try:
        inflow = validate_table(raw_in,"in")
    except ValueError as exc:
        inflow = pd.DataFrame()
        errors.append(f"Zulauf: {exc}")
    try:
        outflow = validate_table(raw_out,"out")
    except ValueError as exc:
        outflow = pd.DataFrame()
        errors.append(f"Ablauf: {exc}")
    pairing_error = None
    try:
        guideline = guideline_lrv_pairs(inflow, outflow)
    except ValueError as exc:
        guideline = pd.DataFrame()
        pairing_error = str(exc)
    for error in errors:
        st.error(error)

    out_samples = 0 if outflow.empty else int(outflow.sample_id.nunique())
    inputs_clean = not errors and not inflow.empty and not outflow.empty
    # The model needs at least two independent samples on each side (model_arrays).
    model_ready = inputs_clean and len(inflow) >= 2 and out_samples >= 2
    result_key = f"vol_result_{pid}"
    has_result = bool(st.session_state.get(result_key))

    with stepper_slot:
        _stepper([
            "done" if inputs_clean else "current",
            "done" if not outflow.empty else ("current" if inputs_clean else "locked"),
            "done" if not guideline.empty else ("current" if inputs_clean else "locked"),
            "done" if has_result else ("current" if model_ready else "locked"),
        ])
    q = _sidebar_panel(selected_indicator, pid, inflow, outflow, guideline, errors)

    with st.container(border=True, key="vr_step2"):
        section_head(2, "Eingaben und effektive Volumina prüfen", locked=outflow.empty)
        if outflow.empty:
            st.caption("Sobald eine gültige Ablaufzeile mit Count und Volumenangabe vorliegt, erscheinen hier die berechneten effektiven Volumina.")
        else:
            tiles = st.columns(5)
            tiles[0].metric("Ablaufproben", out_samples, border=True,
                            help="Unabhängige Umweltproben; technische Replikate zählen nicht doppelt.")
            tiles[1].metric("Analysen", len(outflow), border=True)
            tiles[2].metric("Nullzählungen", int((outflow.value == 0).sum()), border=True)
            tiles[3].metric("Effektiv je Analyse (mL)", _volume_span(outflow.effective_volume_ml), border=True,
                            help="Effektives Originalwasservolumen je Analyse nach Anreicherung, Verdünnung und Wiederfindung (Leitfaden Gl. 4-2).")
            tiles[4].metric("Bezugsfaktor", _volume_span(outflow.exposure), border=True,
                            help="V_eff / 100 mL. Mit diesem Faktor wird die Zulaufkonzentration auf das repräsentative Ablaufvolumen angehoben (Leitfaden Gl. 4-3).")
            st.info("**So wird gerechnet (Leitfaden Kap. 4.6.1):** Der Bezugsfaktor V_eff/100 mL wird auf die **Zulaufkonzentration** angewandt (Gl. 4-3), nicht auf den Ablauf. Der Ablaufcount bleibt die gezählte Zahl – bei null gezählten Einheiten nach Gl. 5-2 die 1, und zwar eine Einheit im effektiven Volumen der Probe, nicht eine je 100 mL.")
            st.markdown("**Berechnete Ablaufgrößen**")
            preview = outflow[["sample_id","replicate_id","value","effective_volume_ml","exposure","concentration_100ml","zero_convention_100ml"]].rename(columns={
                "sample_id":"Probe", "replicate_id":"Replikat", "value":"Count", "effective_volume_ml":"Effektiv V_eff (mL)",
                "exposure":"Bezugsfaktor V_eff/100 mL", "concentration_100ml":"Count je 100 mL (nur Bericht)",
                "zero_convention_100ml":"Gl. 5-2 Nenner (je 100 mL)"})
            st.dataframe(preview, hide_index=True, width="stretch")
            st.caption("Kontrollieren Sie besonders das effektive Volumen und den daraus folgenden Bezugsfaktor. Die Spalte »Count je 100 mL« ist eine reine Berichtsgröße: Bei Count 0 steht dort 0, und das sagt nichts über die Leistung der Anlage – die Aussage steckt im Bezugsfaktor. Der Nenner nach Gl. 5-2 setzt nur für die Leitfadenrechnung einen Count von 0 auf 1. Im statistischen Modell bleibt eine Null eine beobachtete Null.")

    with st.container(border=True, key="vr_step3"):
        section_head(3, "Messbegrenzte LRV nach Leitfaden", locked=guideline.empty)
        st.write("Diese Ausgabe verwendet zusammengehörige Zu- und Ablaufproben. Tragen Sie dafür bei beiden Proben dieselbe Paar-ID ein.")
        st.markdown("Gerechnet wird in der Laufrichtung des Leitfadens: Der Zulauf wird mit dem Bezugsfaktor auf das repräsentative Ablaufvolumen angehoben (Gl. 4-3), der Ablauf bleibt die gezählte Zahl – bei null gezählten Einheiten nach Gl. 5-2 die 1.")
        st.latex(r"\mathrm{LRV} = \log_{10}\!\left(\frac{c_{\mathrm{Zulauf}} \cdot V_{\mathrm{eff}}/100\,\mathrm{mL}}{\max(k,\,1)}\right)")
        if pairing_error:
            st.error(f"Paarung: {pairing_error}")
        if guideline.empty:
            st.info("Noch keine vollständigen Paare: Bei der zugehörigen Zu- und Ablaufprobe dieselbe Paar-ID eintragen. Die Position einer Zeile wird nicht als Paarung verwendet.")
        else:
            shown = guideline.rename(columns={
                "pair_id":"Paar-ID", "inflow_sample_id":"Zulaufprobe", "outflow_sample_id":"Ablaufprobe",
                "count":"Gesamt-Count", "effective_volume_ml":"Effektiv gesamt (mL)",
                "reference_factor":"Bezugsfaktor V_eff/100 mL",
                "inflow_concentration_100ml":"Zulauf (/100 mL)",
                "inflow_concentration_reference":"Zulauf bezogen auf V_eff, Gl. 4-3",
                "effluent_denominator_100ml":"Ablaufnenner Gl. 5-2 (/100 mL)",
                "guideline_lrv":"Messbegrenzte LRV",
            })
            descriptive_q = float(np.quantile(guideline.guideline_lrv, q / 100.))
            headline, table = st.columns([0.28, 0.72], gap="medium")
            headline.metric(f"Deskriptives q{q}", f"{descriptive_q:.2f} log", border=True,
                            help=f"Perzentil über die {len(guideline)} gepaarten Leitfaden-LRV.")
            table.dataframe(shown, hide_index=True, width="stretch",
                            column_config={"Messbegrenzte LRV": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Technische Ablaufreplikate werden je Umweltprobe aus Counts und getrennten effektiven Volumina zusammengefasst; Bezugsfaktor und angehobener Zulauf beziehen sich auf das gepoolte Volumen. Die Ausgabe ist messbegrenzt und enthält kein Unsicherheitsintervall: Ohne Nachweis im Ablauf begrenzt der Bezugsfaktor, wie viel Reduktion überhaupt belegbar ist.")
            st.download_button("Gepaarte Leitfaden-LRV als CSV herunterladen",
                               guideline.to_csv(index=False).encode("utf-8"),
                               file_name=f"validreuse_{pid}_leitfaden_lrv.csv", mime="text/csv")

    with st.container(border=True, key="vr_step4"):
        unlocked = model_ready or has_result
        section_head(4, "Modellbasierte Auswertung", locked=not unlocked)
        st.write("Die Modelle schätzen Konzentrationen aus Counts und individuellen Volumina. So werden Zählunsicherheit und Unterschiede zwischen Umweltproben getrennt berücksichtigt.")
        if unlocked:
            _model_section(parameter, pid, raw_in, raw_out, inflow, outflow, guideline, q, errors, result_key)
        else:
            missing = []
            if errors:
                missing.append("fehlerfreie Eingaben in beiden Tabellen")
            if len(inflow) < 2:
                missing.append("mindestens zwei Zulaufkonzentrationen")
            if out_samples < 2:
                missing.append("mindestens zwei unabhängige Ablaufproben")
            st.info("Für die modellbasierte Auswertung fehlt noch: " + " sowie ".join(missing) + ".")


RECOVERY_GRID = [.01, .02, .05, .10, .20, .50, .80, 1.]


def _shared_recovery(outflow):
    """The one recovery every analysis shares, or None when they differ."""
    if outflow.empty or (outflow.volume_method == "effective").any():
        return None
    values = pd.to_numeric(outflow.recovery, errors="coerce")
    return None if values.isna().any() or values.nunique() != 1 else float(values.iloc[0])


def _recovery_sensitivity(outflow, results, target, alpha):
    """V_eff is linear in the recovery, so every LRV moves by log10(R/R0).

    The curve is an analytic shift of the fitted bounds, not a refit: exact for the
    guideline LRV and accurate to well under 0.1 log for the model candidates.
    """
    st.markdown("#### Empfindlichkeit gegenüber der Wiederfindung")
    base = _shared_recovery(outflow)
    if base is None:
        st.caption("Die Ablaufanalysen teilen keine gemeinsame Wiederfindung, oder mindestens eine Zeile "
                   "verwendet ein extern berechnetes Volumen. Eine gemeinsame Wiederfindungsachse ist dann "
                   "nicht definiert. Allgemein gilt: Ein gemeinsamer Faktor f auf alle effektiven Volumina "
                   "verschiebt jede LRV um log10(f).")
        return
    st.caption(f"Angesetzt sind {base*100:g} % Wiederfindung. Das effektive Volumen ist linear in der "
               f"Wiederfindung, die LRV also um log10(R/{base:g}) verschoben. Die Kurven sind eine "
               "analytische Verschiebung der angepassten Grenzen, keine Neuanpassung.")

    grid = np.logspace(-2., 0., 161)
    curves = pd.concat([pd.DataFrame({"Wiederfindung": grid*100., "Modell": label,
                                      "Untere Grenze": result["L_alpha"] + np.log10(grid/base)})
                        for label, result in results.items()], ignore_index=True)
    axis = alt.Axis(title="Wiederfindung (%)", values=[1, 2, 5, 10, 20, 50, 100], format="g")
    lines = alt.Chart(curves).mark_line().encode(
        x=alt.X("Wiederfindung:Q", scale=alt.Scale(type="log", domain=[1., 100.]), axis=axis),
        y=alt.Y("Untere Grenze:Q", scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(title=f"Untere {(1-alpha)*100:g}%-Grenze (log)")),
        color=alt.Color("Modell:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Modell:N", alt.Tooltip("Wiederfindung:Q", format=".1f"),
                 alt.Tooltip("Untere Grenze:Q", format=".2f")])
    goal = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[6, 4], color="#b0392b").encode(y="y:Q")
    here = alt.Chart(pd.DataFrame({"x": [base*100.]})).mark_rule(strokeDash=[2, 3], color="#6b7686").encode(x="x:Q")
    st.altair_chart(lines + goal + here, width="stretch")
    st.caption(f"Gestrichelt rot: Zielwert {target:g} log. Gestrichelt grau: angesetzte Wiederfindung.")

    table = pd.DataFrame({"Wiederfindung": [f"{r*100:g} %" for r in RECOVERY_GRID]})
    for label, result in results.items():
        table[label] = [result["L_alpha"] + np.log10(r/base) for r in RECOVERY_GRID]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={label: st.column_config.NumberColumn(label, format="%.2f")
                                for label in results})

    tiles = st.columns(len(results))
    for column, (label, result) in zip(tiles, results.items()):
        critical = base * 10.**(target - result["L_alpha"])
        shown = ("ueber 100 %" if critical > 1. else f"{critical*100:.2g} %" if critical >= .0001 else "unter 0,01 %")
        column.metric(f"Grenzwiederfindung {label}", shown, border=True,
                      help=f"Unterhalb dieser Wiederfindung unterschreitet die untere Grenze den Zielwert "
                           f"{target:g} log. 'ueber 100 %' bedeutet, dass der Zielwert auch bei vollstaendiger "
                           "Wiederfindung nicht erreicht wird.")
    note = ("Die Verschiebung setzt voraus, dass sich die Posteriorverteilung starr mitbewegt. Bei "
            "Datensaetzen mit Nachweisen im Ablauf stimmt das bis unter 0,05 log; sind alle Counts null, "
            "traegt der Prior einen Teil des Ergebnisses und die Abweichung kann rund 0,08 log erreichen.")
    st.caption(note)


def _model_section(parameter, pid, raw_in, raw_out, inflow, outflow, guideline, q, errors, result_key):
    """Model choice, assumptions, the fit itself and its results."""
    selected = st.multiselect("Ablaufmodelle vor der Auswertung festlegen", list(FAMILIES), default=UNPAIRED,
                              help="Poisson-Lognormal ist der Hauptkandidat; die Negativbinomialverteilung zeigt die Empfindlichkeit gegenüber einer anderen Mischverteilung. Die bedingten Kandidaten schätzen zusätzlich die Kopplung an den Zulauf.")
    conditional = [label for label in selected if FAMILIES[label].startswith("conditional_")]
    unpaired = [label for label in selected if label not in conditional]
    st.caption("Ungepaart ist der Standard: Zu- und Ablaufverteilung werden unabhängig verglichen, historische Zulaufwerte bleiben nutzbar. Die bedingten Kandidaten setzen je Ablaufprobe eine über die Paar-ID zugeordnete Zulaufkonzentration voraus. Die App wählt nicht automatisch das Modell mit dem höheren Ergebnis.")
    assumptions, advanced = st.columns([0.58, 0.42], gap="large")
    with assumptions:
        independent = True
        if unpaired:
            independent = st.checkbox("Ich bestätige: Eine ungepaarte Auswertung mit unabhängigen Zu- und Ablaufverteilungen ist für diese Fragestellung angemessen.", value=False,
                                      help="Gilt nur für die ungepaarten Kandidaten. Bestätigen Sie dies nur, wenn die Verteilungen unabhängig verglichen werden sollen.")
            st.caption("Bei den ungepaarten Kandidaten bleiben Paar-IDs dokumentiert, werden aber nicht als Korrelation verwendet. Bei zeitlich gepaarten Proben muss die Unabhängigkeitsannahme fachlich begründet sein.")
        slope_mode = "free"
        if conditional:
            slope_mode = SLOPE_LABELS[st.radio("Kopplung b der bedingten Kandidaten", list(SLOPE_LABELS),
                                               horizontal=True, key=f"slope_{pid}",
                                               help="b steuert, wie stark die Ablaufkonzentration der Zulaufkonzentration folgt.")]
            st.caption("b = 0 reproduziert das ungepaarte Modell, b = 1 eine feste Logreduktion je Probe. Frei geschätzt liegt die App dazwischen und weist das Intervall von b aus; nur der Anteil (1 − b) der Zulaufstreuung erreicht die LRV.")
        target = st.number_input("LRV-Zielwert", min_value=0., value=float(parameter["target"]), step=.1, key=f"target_{pid}",
                                 help="Die untere Unsicherheitsgrenze wird mit diesem fachlich festgelegten Zielwert verglichen.")
    with advanced:
        with st.expander("Erweiterte Recheneinstellungen und Annahmen"):
            st.write("Mehr Ziehungen können stabilere Schätzungen liefern, verlängern aber die Rechenzeit. Die Standardwerte sind für eine erste vollständige Auswertung vorgesehen.")
            draws = st.number_input("Ziehungen je Kette", min_value=200,max_value=10000,value=1000,step=100,
                                    help="Ausgewertete Zufallsziehungen je Rechenkette nach dem Warmup.")
            warmup = st.number_input("Warmup je Kette", min_value=200,max_value=10000,value=1000,step=100,
                                     help="Anpassungsphase jeder Rechenkette; diese Ziehungen fließen nicht in das Ergebnis ein.")
            chains = st.number_input("Anzahl Ketten",min_value=2,max_value=4,value=2,
                                     help="Unabhängige Rechenketten zur Prüfung, ob die Schätzung stabil zusammenläuft.")
            alpha = st.selectbox("Alpha der einseitigen Untergrenze",[.05,.1,.2],
                                 help="Alpha 0,05 ergibt eine einseitige 95-%-Kredibilitätsuntergrenze.")
            seed = st.number_input("Zufalls-Seed",min_value=0,max_value=2**32-1,value=42,
                                   help="Gleiche Daten, Einstellungen und Seed ermöglichen eine reproduzierbare Rechnung.")
            st.caption("Feste Priors auf ln(Konzentration /100 mL): Zulaufmittel N(0,10), Zulaufstreuung HalfNormal(2); Ablaufmittel N(0,5). PLN-Streuung HalfNormal(2); NB log(shape) N(0,1.5). Wiederfindung und Volumina gelten in der Anpassung als bekannt; die Empfindlichkeit gegenueber der Wiederfindung wird unterhalb der Ergebnisse ausgewiesen. Prior-Sensitivitaet ist nicht automatisiert.")
    settings = dict(models=selected, target=target, q=q, alpha=alpha, draws=draws, warmup=warmup,
                    chains=chains, seed=seed, independent=independent, slope_mode=slope_mode)
    fingerprint = _fingerprint(raw_in,raw_out,settings)
    if not PYMC_AVAILABLE:
        st.error("Diese Auswertung benoetigt PyMC in einer unterstuetzten Python-Umgebung (z.B. Python 3.13).")
    if st.button("Modellbasierte LRV berechnen",type="primary",disabled=not PYMC_AVAILABLE,width="stretch",
                 help="Startet beide ausgewählten Modelle mit den geprüften Eingaben."):
        st.session_state.pop(result_key,None)
        if errors or not selected or not independent:
            st.error("Eingaben korrigieren, mindestens ein Modell waehlen und die Unabhaengigkeitsannahme bestaetigen.")
        else:
            try:
                # Conditional candidates need the pair-matched arrays; unpaired ones may
                # use every influent value, including historical samples without a pair.
                arrays = model_arrays(inflow,outflow) if unpaired else None
                paired_arrays = model_arrays(inflow,outflow,paired=True) if conditional else None
                payload = {label:dict(analysis_kind="volume_counts", family=FAMILIES[label],
                                      **(paired_arrays if label in conditional else arrays),
                                      **({"slope_mode":slope_mode} if label in conditional else {}),
                                      q=q,alpha=alpha,draws=draws,warmup=warmup,chains=chains,seed=seed)
                           for label in selected}
                with st.spinner("Modelle werden angepasst und geprueft …"):
                    result = pymc_q10_lrv_batch(payload)
                st.session_state[result_key] = dict(fingerprint=fingerprint, result=result, settings=settings,
                    created=datetime.now(timezone.utc).isoformat(), input_csv=raw_in.to_csv(index=False),
                    output_csv=raw_out.to_csv(index=False), indicator=parameter["name"], payload=payload,
                    guideline_lrv=guideline.to_dict(orient="records"))
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    saved = st.session_state.get(result_key)
    if not saved:
        return
    if saved["fingerprint"] != fingerprint:
        st.info("Eingaben oder Einstellungen wurden geaendert. Fuer aktuelle Ergebnisse erneut berechnen.")
        return
    st.divider()
    st.markdown("#### Ergebnisse: modellbasierte Konzentrations-LRV")
    estimands = {result.get("estimand") for result in saved["result"]["results"].values()}
    basis = " bzw. ".join(text for key, text in
                          [("latent_independent_population_lrv", "bei unabhaengigem Zu-/Ablauf"),
                           ("latent_conditional_population_lrv", "bei geschaetzter Kopplung an den Zulauf")]
                          if key in estimands) or "der Population"
    st.caption(f"q{q} der Population {basis}. Untergrenze: einseitige {(1-alpha)*100:g}%-Kredibilitaetsgrenze. Experimentelle Kandidaten; keine Gleichsetzung mit der messbegrenzten Leitfaden-LRV oder einer regulatorischen Freigabe.")
    summary_container = st.container()
    with st.expander("So lesen Sie die Ergebnistabelle"):
        st.markdown("""
- **Untere Grenze:** vorsichtige, einseitige Unsicherheitsgrenze für das gewählte Populationsperzentil. Dieser Wert wird mit dem Zielwert verglichen.
- **Median q:** mittlere Lage der posterioren Schätzung des Populationsperzentils; nicht als alleiniger Freigabewert verwenden.
- **Status:** Das Modellkriterium ist erreicht, wenn die untere Grenze mindestens dem Zielwert entspricht und die Diagnostik keine Warnung enthält.
- **Modellspanne:** Unterschied zwischen den Kandidatenmodellen. Eine große Spanne weist auf Modellabhängigkeit hin und sollte fachlich geprüft werden.
""")
    rows = []
    for label,error in saved["result"]["errors"].items():
        st.error(f"{label}: {error}")
    for label,result in saved["result"]["results"].items():
        status = "Analyse nicht belastbar / pruefen" if result["warnings"] else ("Modellkriterium erreicht" if result["L_alpha"] >= target else "Modellkriterium nicht nachgewiesen")
        coupling = result.get("slope")
        if not coupling:
            shown_b = "ungepaart"
        elif coupling["fixed"] is not None:
            shown_b = f"{coupling['fixed']:g} (fest)"
        else:
            shown_b = f"{coupling['median']:.2f} ({coupling['low']:.2f}–{coupling['high']:.2f})"
        rows.append({"Modell":label,"Kopplung b":shown_b,"Untere Grenze":round(result["L_alpha"],2),
                     "Median q":round(result["median"],2),"Status":status})
        with st.expander(f"{label}: Diagnostik und praediktive Pruefung",expanded=True):
            if coupling and coupling["fixed"] is None:
                st.metric("Kopplung b (Median)", f"{coupling['median']:.2f}", border=True,
                          help=f"90%-Intervall {coupling['low']:.2f} bis {coupling['high']:.2f}. "
                               "Nahe 0 verhaelt sich die Anlage wie im ungepaarten Modell, nahe 1 folgt der "
                               "Ablauf dem Zulauf und nur (1 - b) der Zulaufstreuung erreicht die LRV.")
            elif coupling:
                st.caption(f"Kopplung b war fest auf {coupling['fixed']:g} gesetzt; sie wurde nicht aus den Daten geschaetzt.")
            d = result["diagnostics"]
            diag = st.columns(4)
            diag[0].metric("Max. R-hat", "—" if d["max_rhat"] is None else f"{d['max_rhat']:.3f}", border=True,
                           help="Werte bis etwa 1,01 sprechen für stabil zusammengelaufene Ketten.")
            diag[1].metric("Min. Tail-ESS", "—" if d["min_tail_ess"] is None else f"{round(d['min_tail_ess']):d}", border=True,
                           help="Effektive Stichprobengröße in den Verteilungsrändern.")
            diag[2].metric("Divergenzen", d["divergences"], border=True,
                           help="Divergente Übergänge weisen auf eine unzuverlässige Anpassung hin.")
            diag[3].metric("MC-Fehler Untergrenze", "—" if d["lower_bound_mcse"] is None else f"{d['lower_bound_mcse']:.3f}", border=True,
                           help="Monte-Carlo-Fehler der Untergrenze in log-Einheiten.")
            for warning in result["warnings"]:
                st.warning(warning)
            st.dataframe(pd.DataFrame(result["predictive_checks"]).rename(columns={"statistic":"Statistik","observed":"Beobachtet","low":"Praediktiv 5%","median":"Praediktiv 50%","high":"Praediktiv 95%"}),hide_index=True,width="stretch")
            st.caption("Neue Umweltproben bei den eingegebenen Volumina; Replikate teilen eine Konzentration. Die Intervalle dienen der Modellpruefung, nicht als formaler Anpassungstest.")
    if rows:
        with summary_container:
            headline = st.columns(len(rows))
            for column, row in zip(headline, rows):
                column.metric(row["Modell"], f"{row['Untere Grenze']:.2f} log", border=True,
                              delta=f"{row['Untere Grenze'] - target:+.2f} gegenüber Zielwert {target:g}")
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
        if len(rows)>1:
            bounds = [r["L_alpha"] for r in saved["result"]["results"].values()]
            st.write(f"Spanne der unteren Grenzen ueber die Kandidaten: **{np.ptp(bounds):.2f} log**.")
            st.caption("Die Modellspanne ist ein Sensitivitaetsmass, kein Unsicherheitsintervall. Diagnostisch auffaellige Kandidaten zuerst pruefen.")
        st.divider()
        _recovery_sensitivity(outflow, saved["result"]["results"], target, alpha)
        st.divider()
        st.download_button("Ergebnisse und Eingaben als JSON herunterladen",json.dumps(saved,ensure_ascii=False,indent=2,
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),file_name=f"validreuse_{pid}.json",mime="application/json")
