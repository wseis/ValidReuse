"""Streamlit workflow for raw effluent counts with individual sample volumes."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sample_data import (COLUMNS, RESULT_TYPES, VOLUME_METHODS, empty_table, example_tables,
                         validate_table, model_arrays, editor_table, guideline_lrv_pairs)
from pymc_bayesian import PYMC_AVAILABLE, pymc_q10_lrv_batch

FAMILIES = {"Poisson-Lognormal": "poisson_lognormal", "Negativbinomial (Poisson-Gamma)": "negative_binomial"}
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
    st.markdown("#### Excel-Vorlage: herunterladen, ausfüllen, wieder importieren")
    st.write("Die Vorlage enthält ein eigenes Blatt **Anleitung**, erklärte Beispielzeilen und getrennte Blätter für Zu- und Ablauf. Die englischen Spaltennamen dienen dem sicheren Import und dürfen nicht geändert werden.")
    left, right = st.columns(2)
    if TEMPLATE_PATH.exists():
        left.download_button(
            "Excel-Vorlage herunterladen",
            TEMPLATE_PATH.read_bytes(),
            file_name="ValidReuse_Datenvorlage.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_download_{parameter_id}",
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
    upload = st.file_uploader("Optional: dieses Tabellenblatt als CSV importieren", type="csv", key=f"upload_{parameter_id}_{side}",
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
    st.title("ValidReuse · Counts und Probenvolumina")
    st.write("Hier werten Sie Rohzählungen im Ablauf zusammen mit dem tatsächlich untersuchten Volumen jeder Probe aus. Das berücksichtigt, dass bei großen und wechselnden Probenvolumina derselbe Count unterschiedliche Konzentrationen bedeutet.")
    st.markdown("### 1. Indikator wählen und Daten eingeben")
    selected_indicator = st.selectbox("Indikator", [p["name"] for p in parameters],
                                      help="Für jeden Indikator werden Eingaben und Ergebnisse getrennt gespeichert.")
    parameter = next(p for p in parameters if p["name"] == selected_indicator)
    pid = parameter["id"]
    _excel_controls(pid)
    c1,c2 = st.columns(2)
    c1.button("Beispieldaten in der Oberfläche laden", on_click=_reset, args=(pid,True),
              help="Ersetzt die aktuellen Eingaben durch ein vollständiges Beispiel.")
    c2.button("Eingaben leeren", on_click=_reset, args=(pid,False),
              help="Entfernt die aktuellen Tabellenwerte für diesen Indikator.")
    with st.expander("Welche Eingabeart passt zu meiner Ablaufprobe?", expanded=True):
        st.markdown("""**Zulauf:** positive Konzentrationen; Dezimalwerte sind erlaubt. **Ablauf:** ganze CFU/PFU-Rohzählungen, einschließlich 0.

- **concentrated – Probe angereichert:** Verwenden, wenn Wasser filtriert oder anderweitig angereichert und nur ein Teil des Eluats analysiert wurde.
- **direct – direktes Aliquot:** Verwenden, wenn ein Wasseraliquot ohne vorherige Anreicherung direkt analysiert wurde.
- **effective – Volumen schon berechnet:** Verwenden, wenn das effektive Originalwasservolumen bereits nachvollziehbar außerhalb der App berechnet wurde.

Alle Volumina in **mL**. Verdünnungsfaktor 10 bedeutet 1:10; ohne Verdünnung 1. Wiederfindung 0,5 bedeutet 50 %. Eine Quelle oder ausdrückliche Annahme ist erforderlich.

15 000 mL filtriert, 400 mL Eluat, 100 mL analysiert und 50 % Wiederfindung ergeben **1 875 mL effektives Volumen**. Drei Counts entsprechen dann **0,16 /100 mL**.

**Technische Replikate:** gleiche Proben-ID, unterschiedliche Replikat-ID und getrennte analysierte Aliquots. Sie teilen im Modell eine Konzentration. Fehlende Ergebnisse als `missing` dokumentieren; `<LOD`, `<LOQ`, MPN und Konzentrationen im Ablauf werden gespeichert, aber für diese Count-Auswertung nicht automatisch umgerechnet.""")
    t1,t2 = st.tabs(["Zulaufkonzentrationen", "Ablauf: Counts und Volumina"])
    with t1:
        raw_in = _editor("in",pid)
    with t2:
        raw_out = _editor("out",pid)
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
    for error in errors:
        st.error(error)
    st.markdown("### 2. Eingaben und effektive Volumina prüfen")
    q = st.slider("Populationsperzentil (%)", 1, 50, 10, key=f"quantile_{pid}",
                  help="10 bedeutet q10: Der Wert, den 10 % der betrachteten Population unterschreiten.")
    if not outflow.empty:
        st.subheader("Berechnete Ablaufkonzentrationen")
        preview = outflow[["sample_id","replicate_id","value","effective_volume_ml","concentration_100ml","zero_convention_100ml"]].rename(columns={
            "sample_id":"Probe", "replicate_id":"Replikat", "value":"Count", "effective_volume_ml":"Effektiv (mL)",
            "concentration_100ml":"Count / effektives Volumen (je 100 mL)", "zero_convention_100ml":"Gl. 5-2 Nenner (je 100 mL)"})
        st.dataframe(preview, hide_index=True, width="stretch")
        st.caption("Kontrollieren Sie besonders das effektive Volumen. Der Nenner nach Gleichung 5-2 setzt nur für die Leitfadenrechnung einen Count von 0 auf 1. Im statistischen Modell bleibt eine Null eine beobachtete Null.")
        st.write(f"{outflow.sample_id.nunique()} unabhängige Ablaufproben · {len(outflow)} Analysen · {int((outflow.value == 0).sum())} Nullzählungen · {outflow.effective_volume_ml.min():g}–{outflow.effective_volume_ml.max():g} mL je Analyse")
    st.markdown("### 3. Messbegrenzte LRV nach Leitfaden")
    st.write("Diese Ausgabe verwendet zusammengehörige Zu- und Ablaufproben. Tragen Sie dafür bei beiden Proben dieselbe Paar-ID ein.")
    try:
        guideline = guideline_lrv_pairs(inflow, outflow)
    except ValueError as exc:
        guideline = pd.DataFrame()
        st.error(f"Paarung: {exc}")
    if guideline.empty:
        st.info("Noch keine vollständigen Paare: Bei der zugehörigen Zu- und Ablaufprobe dieselbe Paar-ID eintragen. Die Position einer Zeile wird nicht als Paarung verwendet.")
    else:
        shown = guideline.rename(columns={
            "pair_id":"Paar-ID", "inflow_sample_id":"Zulaufprobe", "outflow_sample_id":"Ablaufprobe",
            "count":"Gesamt-Count", "effective_volume_ml":"Effektiv gesamt (mL)",
            "inflow_concentration_100ml":"Zulauf (/100 mL)",
            "effluent_denominator_100ml":"Ablaufnenner Gl. 5-2 (/100 mL)",
            "guideline_lrv":"Messbegrenzte LRV",
        })
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"Messbegrenzte LRV": st.column_config.NumberColumn(format="%.2f")})
        descriptive_q = float(np.quantile(guideline.guideline_lrv, q / 100.))
        st.write(f"Deskriptives q{q} der {len(guideline)} gepaarten Leitfaden-LRV: **{descriptive_q:.2f} log**")
        st.caption("Technische Ablaufreplikate werden je Umweltprobe aus Counts und getrennten effektiven Volumina zusammengefasst. Die Ausgabe ist messbegrenzt und enthält kein Unsicherheitsintervall.")
        st.download_button("Gepaarte Leitfaden-LRV als CSV herunterladen",
                           guideline.to_csv(index=False).encode("utf-8"),
                           file_name=f"validreuse_{pid}_leitfaden_lrv.csv", mime="text/csv")
    st.markdown("### 4. Modellbasierte Auswertung")
    st.write("Die Modelle schätzen Konzentrationen aus Counts und individuellen Volumina. So werden Zählunsicherheit und Unterschiede zwischen Umweltproben getrennt berücksichtigt.")
    selected = st.multiselect("Ablaufmodelle vor der Auswertung festlegen", list(FAMILIES), default=list(FAMILIES),
                              help="Poisson-Lognormal ist der Hauptkandidat; die Negativbinomialverteilung zeigt die Empfindlichkeit gegenüber einer anderen Mischverteilung.")
    st.caption("Poisson-Lognormal ist der erste Kandidat; Negativbinomial dient als Vergleich. Die App wählt nicht automatisch das Modell mit dem höheren Ergebnis.")
    independent = st.checkbox("Ich bestätige: Eine ungepaarte Auswertung mit unabhängigen Zu- und Ablaufverteilungen ist für diese Fragestellung angemessen.", value=False,
                              help="Bestätigen Sie dies nur, wenn die Verteilungen unabhängig verglichen werden sollen.")
    st.caption("Paar-IDs bleiben dokumentiert, werden in dieser Modellrechnung aber nicht als Korrelation verwendet. Bei zeitlich gepaarten Proben muss die Unabhängigkeitsannahme fachlich begründet sein.")
    target = st.number_input("LRV-Zielwert", min_value=0., value=float(parameter["target"]), step=.1, key=f"target_{pid}",
                             help="Die untere Unsicherheitsgrenze wird mit diesem fachlich festgelegten Zielwert verglichen.")
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
        st.caption("Feste Priors auf ln(Konzentration /100 mL): Zulaufmittel N(0,10), Zulaufstreuung HalfNormal(2); Ablaufmittel N(0,5). PLN-Streuung HalfNormal(2); NB log(shape) N(0,1.5). Wiederfindung und Volumina gelten als bekannt. Prior- und Wiederfindungssensitivitaet sind noch nicht automatisiert.")
    settings = dict(models=selected, target=target, q=q, alpha=alpha, draws=draws, warmup=warmup, chains=chains, seed=seed, independent=independent)
    fingerprint = _fingerprint(raw_in,raw_out,settings)
    result_key = f"vol_result_{pid}"
    if not PYMC_AVAILABLE:
        st.error("Diese Auswertung benoetigt PyMC in einer unterstuetzten Python-Umgebung (z.B. Python 3.13).")
    if st.button("Modellbasierte LRV berechnen",type="primary",disabled=not PYMC_AVAILABLE,
                 help="Startet beide ausgewählten Modelle mit den geprüften Eingaben."):
        st.session_state.pop(result_key,None)
        if errors or not selected or not independent:
            st.error("Eingaben korrigieren, mindestens ein Modell waehlen und die Unabhaengigkeitsannahme bestaetigen.")
        else:
            try:
                arrays = model_arrays(inflow,outflow)
                payload = {label:dict(analysis_kind="volume_counts", family=FAMILIES[label], **arrays,
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
    st.subheader("Ergebnisse: modellbasierte Konzentrations-LRV")
    st.caption(f"q{q} der Population bei unabhaengigem Zu-/Ablauf. Untergrenze: einseitige {(1-alpha)*100:g}%-Kredibilitaetsgrenze. Experimentelle Kandidaten; keine Gleichsetzung mit der messbegrenzten Leitfaden-LRV oder einer regulatorischen Freigabe.")
    with st.expander("So lesen Sie die Ergebnistabelle", expanded=True):
        st.markdown("""
- **Untere Grenze:** vorsichtige, einseitige Unsicherheitsgrenze für das gewählte Populationsperzentil. Dieser Wert wird mit dem Zielwert verglichen.
- **Median q:** mittlere Lage der posterioren Schätzung des Populationsperzentils; nicht als alleiniger Freigabewert verwenden.
- **Status:** Das Modellkriterium ist erreicht, wenn die untere Grenze mindestens dem Zielwert entspricht und die Diagnostik keine Warnung enthält.
- **Modellspanne:** Unterschied zwischen den Kandidatenmodellen. Eine große Spanne weist auf Modellabhängigkeit hin und sollte fachlich geprüft werden.
""")
    summary_container = st.container()
    rows = []
    for label,error in saved["result"]["errors"].items():
        st.error(f"{label}: {error}")
    for label,result in saved["result"]["results"].items():
        status = "Analyse nicht belastbar / pruefen" if result["warnings"] else ("Modellkriterium erreicht" if result["L_alpha"] >= target else "Modellkriterium nicht nachgewiesen")
        rows.append({"Modell":label,"Untere Grenze":round(result["L_alpha"],2),"Median q":round(result["median"],2),"Status":status})
        with st.expander(f"{label}: Diagnostik und praediktive Pruefung",expanded=True):
            d = result["diagnostics"]
            st.write({"Max. R-hat": None if d["max_rhat"] is None else round(d["max_rhat"], 3),
                      "Min. Tail-ESS": None if d["min_tail_ess"] is None else round(d["min_tail_ess"]),
                      "Divergenzen": d["divergences"],
                      "Monte-Carlo-Fehler der Untergrenze (log)": None if d["lower_bound_mcse"] is None else round(d["lower_bound_mcse"], 3)})
            for warning in result["warnings"]:
                st.warning(warning)
            st.dataframe(pd.DataFrame(result["predictive_checks"]).rename(columns={"statistic":"Statistik","observed":"Beobachtet","low":"Praediktiv 5%","median":"Praediktiv 50%","high":"Praediktiv 95%"}),hide_index=True)
            st.caption("Neue Umweltproben bei den eingegebenen Volumina; Replikate teilen eine Konzentration. Die Intervalle dienen der Modellpruefung, nicht als formaler Anpassungstest.")
    if rows:
        summary_container.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
        if len(rows)>1:
            bounds = [r["L_alpha"] for r in saved["result"]["results"].values()]
            st.write(f"Spanne der unteren Grenzen ueber die Kandidaten: **{np.ptp(bounds):.2f} log**.")
            st.caption("Die Modellspanne ist ein Sensitivitaetsmass, kein Unsicherheitsintervall. Diagnostisch auffaellige Kandidaten zuerst pruefen.")
        st.download_button("Ergebnisse und Eingaben als JSON herunterladen",json.dumps(saved,ensure_ascii=False,indent=2,
            default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),file_name=f"validreuse_{pid}.json",mime="application/json")
