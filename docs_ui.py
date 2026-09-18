"""Documentation workspace: every quantity the app reports, with its formula.

Each section names the module and function that implements it, so the
documentation can be checked against the code rather than drifting from it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sample_data import VOLUME_METHODS, effective_volume
from ui_theme import hero, inject_css, note, section_head

UNIT_FACTORS = {"/100 mL": 1.0, "/mL": 100.0, "/L": 0.1}

SYMBOLS = [
    ("$V_{\\mathrm{filt}}$", "filtered_ml", "Filtriertes oder anderweitig angereichertes Originalwasser", "mL"),
    ("$V_{\\mathrm{el}}$", "eluate_ml", "Gesamtvolumen des Eluats", "mL"),
    ("$V_{\\mathrm{an}}$", "assayed_ml", "Tatsächlich analysiertes Volumen", "mL"),
    ("$f$", "dilution_factor", "Verdünnungsfaktor; 1 ohne Verdünnung, 10 bei 1:10", "–"),
    ("$R$", "recovery", "Wiederfindung als Anteil, $0 < R \\le 1$", "–"),
    ("$V_{\\mathrm{eff}}$", "effective_volume_ml", "Effektives Originalwasservolumen einer Analyse", "mL"),
    ("$E$", "exposure", "Exposition, $V_{\\mathrm{eff}}$ in Einheiten von 100 mL", "100 mL"),
    ("$k$", "value (Ablauf)", "Rohzählung CFU/PFU einer Analyse, ganzzahlig $\\ge 0$", "Counts"),
    ("$c_{\\mathrm{Zulauf}}$", "concentration_100ml", "Zulaufkonzentration nach Einheitenumrechnung", "je 100 mL"),
    ("$c_{\\mathrm{Ablauf}}$", "concentration_100ml", "Count je effektivem Volumen", "je 100 mL"),
    ("$C_j$", "concentration", "Latente Konzentration der Umweltprobe $j$ im Modell", "je 100 mL"),
    ("$g(a)$", "sample_groups", "Zuordnung der Analyse $a$ zu ihrer Umweltprobe", "–"),
    ("$q$", "q", "Populationsperzentil in Prozent, Standard 10", "%"),
    ("$\\alpha$", "alpha", "Irrtumswahrscheinlichkeit der einseitigen Untergrenze", "–"),
    ("$L_\\alpha$", "L_alpha", "Einseitige untere Kredibilitätsgrenze der LRV", "log"),
]


def _table(rows, columns):
    """Static reference table as markdown, so the browser search finds the text."""
    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    st.markdown("\n".join([head, rule, *body]))


def _source(text):
    st.caption(f"Implementierung: {text}")


def _overview():
    with st.container(border=True, key="vr_doc_ways"):
        section_head("A", "Zwei Auswertungswege, zwei verschiedene Größen")
        st.write(
            "Die App berechnet zwei LRV-Größen, die sich nicht ineinander überführen lassen. "
            "Beide werden getrennt ausgewiesen und dürfen nicht gegeneinander ausgetauscht werden."
        )
        _table(
            [
                ["**Messbegrenzte Leitfaden-LRV**", "Schritt 3 im Count-Arbeitsbereich",
                 "Counts und effektive Volumina gepaarter Proben",
                 "Deskriptiv, Gl. 5-1/5-2 auf Count-Skala", "Kein Unsicherheitsintervall"],
                ["**Modellbasierte Populations-LRV**", "Schritt 4 im Count-Arbeitsbereich",
                 "Counts, Expositionen, Zulaufkonzentrationen",
                 "Quantil der latenten Konzentrations-LRV", "Einseitige Kredibilitätsuntergrenze"],
                ["**Count-prädiktive LRV**", "Bisheriger Methodenvergleich",
                 "Zu- und Ablaufwerte auf gleicher Volumenbasis",
                 "Quantil simulierter Count-LRV mit Nullkonvention", "Perzentile der Simulationsverteilung"],
            ],
            ["Größe", "Wo", "Eingang", "Definition", "Unsicherheit"],
        )
        note(
            "**Wichtig:** Die messbegrenzte LRV ist durch das analysierte Volumen nach oben begrenzt – "
            "ein Count von 0 wird nach Gl. 5-2 wie 1 behandelt. Die modellbasierte LRV schätzt dagegen die "
            "latente Konzentration; eine beobachtete Null bleibt dort eine echte Null in der Likelihood."
        )

    with st.container(border=True, key="vr_doc_chain"):
        section_head("B", "Rechenkette des Count-Arbeitsbereichs")
        st.markdown(
            "1. **Rohdaten je Analyse:** Proben-ID, Replikat-ID, Count $k$, Volumenangaben, Wiederfindung.\n"
            "2. **Effektives Volumen** $V_{\\mathrm{eff}}$ aus der gewählten Volumenmethode.\n"
            "3. **Exposition** $E = V_{\\mathrm{eff}}/100\\,\\mathrm{mL}$ und Konzentration $c = k/E$.\n"
            "4. **Leitfaden-LRV** je Paar-ID aus gepoolten Replikaten, mit Nullkonvention nach Gl. 5-2.\n"
            "5. **Modellanpassung:** Poisson-Likelihood auf $E_a \\cdot C_{g(a)}$, lognormale oder "
            "gammaverteilte latente Konzentrationen.\n"
            "6. **Populationsquantil** $R_q$ je Posterior-Ziehung, daraus Median und Untergrenze $L_\\alpha$.\n"
            "7. **Statusentscheid:** Vergleich von $L_\\alpha$ mit dem Zielwert, zusammen mit der Diagnostik."
        )
        _source("`sample_data.py` (Schritte 1–4), `count_models.py` (Schritte 5–6), `volume_ui.py` (Schritt 7)")


def _volumes():
    with st.container(border=True, key="vr_doc_veff"):
        section_head("1", "Effektives Volumen einer Analyse")
        st.write(
            "Alle drei Volumenmethoden führen auf dasselbe Ziel: das Volumen an **Originalwasser**, "
            "das die einzelne Analyse tatsächlich repräsentiert. Zuerst wird das analysierte Volumen "
            "um die Verdünnung korrigiert:"
        )
        st.latex(r"V_{\mathrm{an}}^{*} = \frac{V_{\mathrm{an}}}{f}")
        st.markdown("**`concentrated` – Probe angereichert:** nur ein Anteil des Eluats wurde analysiert.")
        st.latex(r"V_{\mathrm{eff}} = V_{\mathrm{filt}} \cdot \frac{V_{\mathrm{an}}^{*}}{V_{\mathrm{el}}} \cdot R")
        st.markdown("**`direct` – direktes Aliquot:** ohne Anreicherung analysiertes Wasservolumen.")
        st.latex(r"V_{\mathrm{eff}} = V_{\mathrm{an}}^{*} \cdot R")
        st.markdown("**`effective` – extern berechnet:** der Wert wird unverändert übernommen.")
        st.latex(r"V_{\mathrm{eff}} = V_{\mathrm{eff}}^{\mathrm{extern}}")
        st.markdown("**Bedingungen, die die App erzwingt:**")
        _table(
            [
                ["$0 < R \\le 1$", "Wiederfindung als Anteil, nicht in Prozent; 0,5 bedeutet 50 %"],
                ["$f \\ge 1$", "Verdünnungsfaktor; 10 steht für 1:10"],
                ["$V_{\\mathrm{an}}^{*} \\le V_{\\mathrm{el}}$", "Nur bei `concentrated`: nicht mehr analysieren als Eluat vorhanden"],
                ["Quelle / Annahme gesetzt", "`recovery_source` ist in allen drei Methoden Pflicht"],
                ["$V_{\\mathrm{eff}}$ endlich und $> 0$", "Sonst wird die Zeile als Fehler zurückgewiesen"],
            ],
            ["Bedingung", "Bedeutung"],
        )
        note(
            "Bei technischen Replikaten derselben Probe aus einem Eluat müssen $V_{\\mathrm{filt}}$, "
            "$V_{\\mathrm{el}}$ und $R$ identisch sein, und die Summe der $V_{\\mathrm{an}}^{*}$ darf "
            "$V_{\\mathrm{el}}$ nicht überschreiten – die Aliquots müssen disjunkt sein."
        )
        _source("`sample_data.effective_volume` und `sample_data.validate_table`")

    with st.container(border=True, key="vr_doc_conc"):
        section_head("2", "Exposition, Konzentration und Nullkonvention")
        st.write("Die Exposition ist das effektive Volumen in Einheiten von 100 mL:")
        st.latex(r"E = \frac{V_{\mathrm{eff}}}{100\,\mathrm{mL}}")
        st.write("Daraus folgt die berichtete Ablaufkonzentration je 100 mL:")
        st.latex(r"c_{\mathrm{Ablauf}} = \frac{k}{E} = \frac{100\,k}{V_{\mathrm{eff}}}")
        st.write(
            "Für die Leitfadenrechnung – und **nur** dort – wird ein Count von 0 im Nenner wie 1 behandelt "
            "(Gl. 5-2). Im statistischen Modell bleibt die Null eine beobachtete Null:"
        )
        st.latex(r"c^{(5\text{-}2)} = \frac{\max(k,\,1)}{E}")
        st.write("Zulaufwerte werden anhand ihrer Einheit auf 100 mL umgerechnet:")
        st.latex(r"c_{\mathrm{Zulauf}} = v \cdot u")
        _table(
            [[f"`{unit}`", f"$u = {factor:g}$"] for unit, factor in UNIT_FACTORS.items()],
            ["Einheit im Formular", "Faktor"],
        )
        _source("`sample_data.validate_table`")

    with st.container(border=True, key="vr_doc_calc"):
        section_head("3", "Rechner: Volumen und Konzentration nachvollziehen")
        st.write(
            "Dieser Rechner ruft dieselbe Funktion auf wie die Auswertung. Die Werte sind mit dem "
            "Beispiel aus der Excel-Vorlage vorbelegt."
        )
        method = st.selectbox("Art der Volumenangabe", VOLUME_METHODS, index=VOLUME_METHODS.index("concentrated"))
        left, right = st.columns(2, gap="large")
        with left:
            filtered = st.number_input("Filtriert $V_{\\mathrm{filt}}$ (mL)", min_value=0., value=15000., step=100.,
                                       disabled=method != "concentrated")
            eluate = st.number_input("Gesamtes Eluat $V_{\\mathrm{el}}$ (mL)", min_value=0., value=400., step=10.,
                                     disabled=method != "concentrated")
            assayed = st.number_input("Analysiert $V_{\\mathrm{an}}$ (mL)", min_value=0., value=100., step=10.,
                                      disabled=method == "effective")
        with right:
            dilution = st.number_input("Verdünnungsfaktor $f$", min_value=1., value=1., step=1.,
                                       disabled=method == "effective")
            recovery = st.number_input("Wiederfindung $R$", min_value=0., max_value=1., value=0.5, step=0.05,
                                       disabled=method == "effective")
            external = st.number_input("Externes $V_{\\mathrm{eff}}$ (mL)", min_value=0., value=1875., step=25.,
                                       disabled=method != "effective")
        count = st.number_input("Rohzählung $k$", min_value=0, value=3, step=1)

        row = pd.Series({
            "volume_method": method, "filtered_ml": filtered, "eluate_ml": eluate, "assayed_ml": assayed,
            "dilution_factor": dilution, "recovery": recovery, "effective_ml": external,
            "recovery_source": "Dokumentationsbeispiel",
        })
        try:
            v_eff = effective_volume(row)
        except ValueError as exc:
            st.warning(f"Diese Kombination wird zurückgewiesen: {exc}")
        else:
            exposure = v_eff / 100.
            tiles = st.columns(4)
            tiles[0].metric("$V_{\\mathrm{eff}}$ (mL)", f"{v_eff:g}", border=True)
            tiles[1].metric("Exposition $E$", f"{exposure:g}", border=True, help="In Einheiten von 100 mL.")
            tiles[2].metric("$c_{\\mathrm{Ablauf}}$ (je 100 mL)", f"{count / exposure:.4g}", border=True)
            tiles[3].metric("Gl. 5-2 Nenner", f"{max(count, 1) / exposure:.4g}", border=True,
                            help="Nur für die Leitfadenrechnung; im Modell bleibt eine Null eine Null.")
            if method == "concentrated":
                st.latex(
                    r"V_{\mathrm{eff}} = %g \cdot \frac{%g / %g}{%g} \cdot %g = %g\ \mathrm{mL}"
                    % (filtered, assayed, dilution, eluate, recovery, v_eff)
                )
            elif method == "direct":
                st.latex(r"V_{\mathrm{eff}} = \frac{%g}{%g} \cdot %g = %g\ \mathrm{mL}"
                         % (assayed, dilution, recovery, v_eff))


def _guideline():
    with st.container(border=True, key="vr_doc_eq"):
        section_head("4", "Gleichungen 5-1 und 5-2 des Validierungsleitfadens")
        st.write(
            "Der FlexTreat-Validierungsleitfaden (Kapitel 5.1.1) definiert die Logreduktion und die "
            "Behandlung von Ablaufwerten unter 1:"
        )
        st.latex(r"\text{(5-1)}\quad \mathrm{LRV} = \log_{10}\!\left(\frac{c_{\mathrm{Zulauf}}}{c_{\mathrm{Ablauf}}}\right)")
        st.latex(r"\text{(5-2)}\quad \mathrm{LRV} = \log_{10}\!\left(c_{\mathrm{Zulauf}}\right) \quad \text{für } c_{\mathrm{Ablauf}} < 1")
        st.write(
            "Mikrobiologische Indikatoren sind diskrete Zählungen. $c_{\\mathrm{Ablauf}} < 1$ bedeutet hier "
            "eine exakte Null, keinen zensierten Wert. Gl. 5-2 wirkt deshalb als Untergrenze von 1, die "
            "**ausschließlich** auf Nullwerte angewandt wird; Werte über null folgen weiter Gl. 5-1. Das Ersetzen "
            "durch die halbe Bestimmungsgrenze lehnt der Leitfaden ausdrücklich ab, weil es die LRV ohne "
            "Datengrundlage um 0,3 anheben würde."
        )
        _source("`lrv.py` (Modul-Docstring und `effluent_denominator`)")

    with st.container(border=True, key="vr_doc_pairs"):
        section_head("5", "Gepaarte messbegrenzte LRV")
        st.write(
            "Technische Replikate einer Umweltprobe werden zuerst zusammengefasst. Counts und disjunkte "
            "effektive Volumina sind additiv:"
        )
        st.latex(r"k_{\mathrm{ges}} = \sum_{r} k_r \qquad\qquad V_{\mathrm{eff,ges}} = \sum_{r} V_{\mathrm{eff},r}")
        st.write("Der Ablaufnenner nach Gl. 5-2 verwendet den gepoolten Count und die gepoolte Exposition:")
        st.latex(r"c_{\mathrm{Ablauf}}^{(5\text{-}2)} = \frac{\max\!\left(k_{\mathrm{ges}},\,1\right)}{V_{\mathrm{eff,ges}} / 100}")
        st.latex(r"\mathrm{LRV}_{\mathrm{Leitfaden}} = \log_{10}\!\left(\frac{c_{\mathrm{Zulauf}}}{c_{\mathrm{Ablauf}}^{(5\text{-}2)}}\right)")
        st.markdown("**Paarungsregeln:**")
        st.markdown(
            "- Die Zuordnung erfolgt **nur** über die Paar-ID, nie über die Zeilenposition.\n"
            "- Eine Paar-ID darf im Zulauf höchstens einmal vorkommen.\n"
            "- Eine Paar-ID darf im Ablauf auf nur eine unabhängige Umweltprobe verweisen.\n"
            "- Zeilen ohne Paar-ID gehen nicht in diese Auswertung ein."
        )
        st.write("Über alle Paare wird anschließend das deskriptive Perzentil gebildet:")
        st.latex(r"\mathrm{LRV}_{q} = \mathrm{Quantil}_{q/100}\!\left(\left\{\mathrm{LRV}_{\mathrm{Leitfaden}}\right\}\right)")
        note(
            "Diese Größe ist **messbegrenzt und deskriptiv**: Sie hat kein Unsicherheitsintervall und "
            "wird durch das analysierte Volumen nach oben begrenzt."
        )
        _source("`sample_data.guideline_lrv_pairs`, Perzentil in `volume_ui.render_volume_workflow`")


def _model():
    with st.container(border=True, key="vr_doc_lik"):
        section_head("6", "Modell: Likelihood und Priors")
        st.write("Die Zulaufkonzentrationen werden als lognormalverteilt behandelt:")
        st.latex(r"\ln c_{\mathrm{Zulauf},i} \sim \mathcal{N}\!\left(\mu_{\mathrm{in}},\ \sigma_{\mathrm{in}}\right)")
        st.write(
            "Jede Ablaufanalyse $a$ ist eine Poisson-Zählung ihrer Exposition und der latenten Konzentration "
            "ihrer Umweltprobe. Technische Replikate teilen sich $C_j$, haben aber je eigene Exposition:"
        )
        st.latex(r"k_a \sim \mathrm{Poisson}\!\left(E_a \cdot C_{g(a)}\right)")
        st.markdown("**Kandidat 1 – Poisson-Lognormal (PLN):**")
        st.latex(r"C_j = \exp\!\left(\mu_{\mathrm{out}} + s\,z_j\right), \qquad z_j \sim \mathcal{N}(0,1)")
        st.markdown("**Kandidat 2 – Negativbinomial (Poisson-Gamma):**")
        st.latex(r"C_j \sim \mathrm{Gamma}\!\left(\text{shape}=s,\ \text{rate}=\frac{s}{e^{\mu_{\mathrm{out}}}}\right)")
        st.markdown("**Feste Priors** auf natürlich-logarithmierte Konzentration je 100 mL:")
        _table(
            [
                ["$\\mu_{\\mathrm{in}}$", "$\\mathcal{N}(0,\\ 10)$", "Lage der Zulaufverteilung"],
                ["$\\sigma_{\\mathrm{in}}$", "$\\mathrm{HalfNormal}(2)$", "Streuung der Zulaufverteilung"],
                ["$\\mu_{\\mathrm{out}}$", "$\\mathcal{N}(0,\\ 5)$", "Lage der Ablaufverteilung"],
                ["$s$ (PLN)", "$\\mathrm{HalfNormal}(2)$", "Streuung der latenten Ablaufkonzentration"],
                ["$\\ln s$ (NB)", "$\\mathcal{N}(0,\\ 1{,}5)$", "Formparameter der Gammaverteilung"],
            ],
            ["Parameter", "Prior", "Bedeutung"],
        )
        note(
            "Volumina und Wiederfindungen gelten als **bekannt**. Ihre Unsicherheit wird nicht propagiert. "
            "Die Priors sind fest verdrahtet und werden nicht aus den Daten gesetzt."
        )
        _source("`count_models.build_model`")

    with st.container(border=True, key="vr_doc_quant"):
        section_head("7", "Populationsquantil der latenten LRV")
        st.write(
            "Ausgewertet wird die LRV eines **neuen, unabhängigen** Paares aus latenter Zulauf- und "
            "Ablaufkonzentration – nicht ein simulierter Count:"
        )
        st.latex(r"R = \log_{10}\!\left(\frac{C_{\mathrm{in}}}{C_{\mathrm{out}}}\right)")
        st.markdown("**Poisson-Lognormal – geschlossene Form.** Die Differenz zweier Normalverteilungen ist normal:")
        st.latex(r"\ln C_{\mathrm{in}} - \ln C_{\mathrm{out}} \sim \mathcal{N}\!\left(\mu_{\mathrm{in}} - \mu_{\mathrm{out}},\ \sqrt{\sigma_{\mathrm{in}}^{2} + s^{2}}\right)")
        st.latex(r"R_q = \frac{\mu_{\mathrm{in}} - \mu_{\mathrm{out}} + z_q\,\sqrt{\sigma_{\mathrm{in}}^{2} + s^{2}}}{\ln 10}")
        st.caption("$z_q$ ist das Standardnormal-Quantil zu $q/100$.")
        st.markdown("**Negativbinomial – numerisch.** Für die Gammaverteilung existiert keine geschlossene Form:")
        st.latex(r"F_R(r) = \mathbb{E}_{C_{\mathrm{in}}}\!\left[\Pr\!\left(C_{\mathrm{out}} \ge C_{\mathrm{in}}\,10^{-r}\right)\right]"
                 r" = \int \varphi(x)\; Q\!\left(s,\; s\,e^{\,\mu_{\mathrm{in}} + \sigma_{\mathrm{in}}x - \mu_{\mathrm{out}} - r\ln 10}\right) \mathrm{d}x")
        st.caption("$Q(a, z)$ ist die regularisierte obere unvollständige Gammafunktion.")
        st.markdown(
            "Das Integral wird mit **Gauß-Hermite-Quadratur** ausgewertet (64 Knoten, zur Kontrolle erneut mit 128), "
            "und $R_q$ mit $F_R(R_q) = q/100$ über **56 Bisektionsschritte** bestimmt. Die Differenz zwischen "
            "64 und 128 Knoten wird als `quadrature_error` mitgeführt."
        )
        _source("`count_models.population_quantiles`")

    with st.container(border=True, key="vr_doc_bound"):
        section_head("8", "Untergrenze, Median und Modellspanne")
        st.write(
            "$R_q$ wird für **jede** Posterior-Ziehung $d$ berechnet. Die berichteten Kennzahlen sind "
            "Quantile dieser Verteilung über $D$ Ziehungen:"
        )
        st.latex(r"L_\alpha = \mathrm{Quantil}_{\alpha}\!\left(\left\{R_q^{(d)}\right\}_{d=1}^{D}\right)")
        st.latex(r"\mathrm{Median} = \mathrm{Quantil}_{0{,}5}\!\left(\left\{R_q^{(d)}\right\}\right) \qquad "
                 r"\mathrm{Obergrenze} = \mathrm{Quantil}_{1-\alpha}\!\left(\left\{R_q^{(d)}\right\}\right)")
        st.write(
            "$L_\\alpha$ ist die **einseitige** $(1-\\alpha)$-Kredibilitätsuntergrenze. Bei $\\alpha = 0{,}05$ "
            "ist das die 95-%-Untergrenze; sie wird mit dem Zielwert verglichen."
        )
        st.write("Werden mehrere Kandidaten gerechnet, wird zusätzlich deren Spanne ausgewiesen:")
        st.latex(r"\Delta = \max_{m} L_\alpha^{(m)} - \min_{m} L_\alpha^{(m)}")
        note(
            "$\\Delta$ ist ein **Sensitivitätsmaß** gegenüber der Modellwahl, kein Unsicherheitsintervall. "
            "Eine große Spanne bedeutet Modellabhängigkeit und muss fachlich geprüft werden. Die App wählt "
            "nicht automatisch den Kandidaten mit dem höheren Ergebnis."
        )
        _source("`count_models.fit_count_model`, Spanne in `volume_ui._model_section`")


def _diagnostics():
    with st.container(border=True, key="vr_doc_diag"):
        section_head("9", "Konvergenz- und Genauigkeitsdiagnostik")
        st.write(
            "Die Diagnostik wird unrundet über **alle** Modellvariablen gebildet, einschließlich der latenten "
            "Probenkonzentrationen. Wird eine Schwelle verletzt, erhält das Ergebnis eine Warnung und gilt "
            "nicht als belastbar."
        )
        _table(
            [
                ["Max. $\\hat{R}$", "`az.rhat` über alle Variablen", "Warnung bei $> 1{,}01$ oder nicht endlich"],
                ["Min. Tail-ESS", "`az.ess(method=\"tail\")`", "Warnung bei $< 400$"],
                ["Divergenzen", "Divergente NUTS-Übergänge", "Warnung bei $> 0$"],
                ["MC-Fehler der Untergrenze", "`az.mcse(method=\"quantile\", prob=`$\\alpha$`)`", "Warnung bei $> 0{,}05$ log oder nicht bestimmbar"],
                ["Quadraturfehler", "64 gegen 128 Knoten, nur NB", "Warnung bei $> 0{,}02$ log"],
                ["Alle Counts null", "$\\sum_a k_a = 0$", "Warnung: Streuung und Quantil sind stark priorabhängig"],
            ],
            ["Kennzahl", "Berechnung", "Schwelle"],
        )
        st.caption("Gesampelt wird mit NUTS, `target_accept = 0.95`, `init = adapt_diag`, standardmäßig 2 Ketten "
                   "mit je 1000 Warmup- und 1000 Auswertungsziehungen.")
        _source("`count_models.fit_count_model`")

    with st.container(border=True, key="vr_doc_ppc"):
        section_head("10", "Prädiktive Prüfung")
        st.write(
            "Simuliert werden **neue Umweltproben** bei den beobachteten Expositionen; technische Replikate "
            "teilen sich dabei ihre latente Konzentration. Aus bis zu 500 Posterior-Ziehungen entstehen "
            "prädiktive Counts, deren Randstatistiken mit den beobachteten verglichen werden."
        )
        _table(
            [
                ["Nullanteil", "Anteil der Analysen mit $k = 0$"],
                ["Maximum (Count)", "Größter beobachteter Count"],
                ["q90 (Count)", "90-%-Perzentil der Counts"],
                ["Varianz/Mittel (Count)", "Dispersionsmaß; 1 entspricht reiner Poisson-Streuung"],
            ],
            ["Statistik", "Bedeutung"],
        )
        st.write(
            "Berichtet werden der beobachtete Wert sowie das 5-, 50- und 95-%-Perzentil der prädiktiven "
            "Verteilung. Liegt eine beobachtete Statistik außerhalb des 90-%-Bereichs, wird gewarnt."
        )
        note("Die Intervalle dienen der **Modellprüfung**, sie sind kein formaler Anpassungstest.")
        _source("`count_models.fit_count_model`")

    with st.container(border=True, key="vr_doc_status"):
        section_head("11", "Statuslogik")
        st.markdown(
            "Im Count-Arbeitsbereich gilt je Kandidat:\n\n"
            "- **Analyse nicht belastbar / prüfen** – sobald mindestens eine Warnung vorliegt. "
            "Dieser Status hat Vorrang, unabhängig vom Zahlenwert.\n"
            "- **Modellkriterium erreicht** – keine Warnung **und** $L_\\alpha \\ge$ Zielwert.\n"
            "- **Modellkriterium nicht nachgewiesen** – keine Warnung, aber $L_\\alpha <$ Zielwert."
        )
        st.write("Im bisherigen Methodenvergleich gilt eine Kombination aus Parameter und Methode als erfüllt, wenn:")
        st.latex(r"L_\alpha \ge \mathrm{Zielwert} \quad \text{und} \quad \mathrm{Median} \ge \mathrm{Zielwert}")
        note(
            "Kein Status der App ist eine regulatorische Freigabe. Die modellbasierten Kandidaten sind "
            "ausdrücklich als **experimentell** gekennzeichnet."
        )
        _source("`volume_ui._model_section` und `app.py`")


def _legacy():
    with st.container(border=True, key="vr_doc_zero"):
        section_head("12", "Nullwertbehandlung im bisherigen Methodenvergleich")
        st.write(
            "Der Legacy-Arbeitsbereich simuliert Ablauf**counts** und braucht deshalb eine Regel für den "
            "Nenner $D(k)$. Alle drei Methoden dort verwenden dieselbe Regel:"
        )
        st.latex(r"D(k) = \begin{cases}"
                 r"\max(k,\,1) & \text{Gl. 5-2: nur Nullwerte auf 1 setzen} \\[2pt]"
                 r"k + 1 & \text{1 zu allen Ablaufwerten addieren} \\[2pt]"
                 r"k & \text{keine Anpassung}"
                 r"\end{cases}")
        st.latex(r"\mathrm{LRV} = \log_{10}\!\left(\frac{c_{\mathrm{Zulauf}}}{D(k_{\mathrm{Ablauf}})}\right)")
        note(
            "Ohne Anpassung sind Nullwerte im Ablauf unzulässig, weil die LRV dann nicht definiert ist. "
            "Die App bricht in diesem Fall mit einer Fehlermeldung ab."
        )
        _source("`lrv.effluent_denominator`")

    with st.container(border=True, key="vr_doc_boot"):
        section_head("13", "Empirischer Bootstrap")
        st.markdown(
            "Für jede der $B$ Wiederholungen:\n\n"
            "1. Ziehe $n_{\\mathrm{zu}}$ Zulauf- und $n_{\\mathrm{ab}}$ Ablaufwerte **mit Zurücklegen** "
            "aus den Originaldaten.\n"
            "2. Ziehe daraus je $n_{\\mathrm{sim}}$ Werte mit Zurücklegen.\n"
            "3. Bilde $\\mathrm{LRV}_j = \\log_{10}\\!\\left(z_j / D(a_j)\\right)$ und daraus das "
            "$q$-Perzentil $\\hat{q}_b$."
        )
        st.write("Berichtet werden Perzentile über die $B$ Werte $\\hat{q}_b$:")
        st.latex(r"L_\alpha = \mathrm{Perzentil}_{100\alpha}\!\left(\left\{\hat{q}_b\right\}\right)")
        note(
            "Zu- und Ablauf werden hier **unabhängig** gezogen. Das Ergebnis ist nicht das empirische "
            "Quantil beobachteter gepaarter LRV-Werte."
        )
        _source("`bootstrap.bootstrap_q10_lrv`")

    with st.container(border=True, key="vr_doc_approx"):
        section_head("14", "Bayessche Approximation")
        st.markdown("**Zulauf** – konjugierte Normal-Inverse-Gamma-Posterior auf $\\ln c_{\\mathrm{Zulauf}}$ "
                    "mit $\\bar{y} = \\overline{\\ln c}$ und $\\mathrm{SSQ} = \\sum_i (\\ln c_i - \\bar{y})^2$:")
        st.latex(r"\kappa_n = \kappa_0 + n, \qquad \alpha_n = \alpha_0 + \tfrac{n}{2}, \qquad "
                 r"\mu_n = \frac{\kappa_0\mu_0 + n\bar{y}}{\kappa_n}")
        st.latex(r"\beta_n = \beta_0 + \tfrac{1}{2}\mathrm{SSQ} + \frac{\kappa_0\,n\,(\bar{y}-\mu_0)^2}{2\kappa_n}")
        st.caption("Mit $\\mu_0 = \\bar{y}$, $\\kappa_0 = 0{,}25$, $\\alpha_0 = 2{,}5$ und "
                   "$\\beta_0 = \\max\\!\\left(\\mathrm{Var}(\\ln c),\\ 10^{-6}\\right)$. "
                   "Da $\\mu_0$ auf den Datenmittelwert gesetzt wird, verschwindet der letzte Term.")
        st.markdown("**Ablauf** – Negativbinomialmodell, gesampelt mit Random-Walk-Metropolis auf "
                    "$(\\ln\\mu,\\ \\ln s)$ und im Warmup adaptierter Schrittweite:")
        st.latex(r"\ln\mu \sim \mathcal{N}\!\left(\ln(\bar{k}+0{,}5),\ 3\right), \qquad \ln s \sim \mathcal{N}(0,\ 2{,}5)")
        st.write(
            "Je Posterior-Ziehung werden $n_{\\mathrm{sim}}$ Zulaufwerte lognormal und $n_{\\mathrm{sim}}$ "
            "Ablaufcounts negativbinomial simuliert; daraus folgt wieder das $q$-Perzentil der LRV."
        )
        _source("`bayesian.bayesian_q10_lrv`")

    with st.container(border=True, key="vr_doc_pymc"):
        section_head("15", "Bayessch (PyMC), Legacy-Variante")
        st.write("Dasselbe Ziel wie die Approximation, aber mit NUTS statt Metropolis:")
        st.latex(r"\ln c_{\mathrm{Zulauf},i} \sim \mathcal{N}(\mu_{\mathrm{in}},\ \sigma_{\mathrm{in}}), \qquad "
                 r"k_{\mathrm{Ablauf},i} \sim \mathrm{NegBinomial}(\mu_{\mathrm{out}},\ \alpha_{\mathrm{out}})")
        _table(
            [
                ["$\\mu_{\\mathrm{in}}$", "$\\mathcal{N}(0,\\ 5)$"],
                ["$\\sigma_{\\mathrm{in}}$", "$\\mathrm{HalfNormal}(2)$"],
                ["$\\ln\\mu_{\\mathrm{out}}$", "$\\mathcal{N}(0,\\ 5)$"],
                ["$\\alpha_{\\mathrm{out}}$", "$\\mathrm{HalfNormal}(10)$"],
            ],
            ["Parameter", "Prior"],
        )
        note(
            "Dieses Legacy-Modell kennt **keine** Expositionen. Es setzt voraus, dass alle Werte auf "
            "derselben Volumenbasis erhoben wurden. Für wechselnde Probenvolumina ist der "
            "Count-Arbeitsbereich zu verwenden."
        )
        _source("`pymc_worker.py` (`get_model`, `run_single`)")


def _symbols():
    with st.container(border=True, key="vr_doc_sym"):
        section_head("16", "Symbolverzeichnis")
        _table([[symbol, f"`{field}`", meaning, unit] for symbol, field, meaning, unit in SYMBOLS],
               ["Symbol", "Feld / Schlüssel", "Bedeutung", "Einheit"])

    with st.container(border=True, key="vr_doc_scope"):
        section_head("17", "Geltungsbereich und bekannte Grenzen")
        st.markdown(
            "- Die modellbasierte Auswertung setzt **unabhängige** Zu- und Ablaufverteilungen voraus. "
            "Paar-IDs bleiben dokumentiert, werden aber nicht als Korrelation modelliert; die Annahme "
            "muss aktiv bestätigt und fachlich begründet werden.\n"
            "- **Wiederfindung und Volumina gelten als bekannt.** Ihre Unsicherheit wird nicht propagiert.\n"
            "- Zensierte Ergebnisse (`<LOD`, `<LOQ`), `interval` und `MPN` werden gespeichert, aber **nicht** "
            "in Counts umgedeutet. Sie blockieren die Count-Auswertung, bis geeignete Rohzählungen vorliegen.\n"
            "- `missing` bleibt im Export erhalten und wird von der Anpassung ausgeschlossen.\n"
            "- Prior-Sensitivität, Wiederfindungssensitivität, zensierte Likelihoods, zeitliche Modelle und "
            "eine vollständige Simulationskalibrierung sind **nicht** automatisiert.\n"
            "- Es findet **keine** automatische Ausreißerentfernung oder Abwertung statt.\n"
            "- Mindestanforderung der Modellrechnung: zwei Zulaufkonzentrationen und zwei unabhängige "
            "Ablaufproben."
        )
        st.markdown("**Quellen**")
        st.markdown(
            "- FlexTreat Validierungsleitfaden, Kapitel 5.1.1 – Zenodo "
            "[10.5281/zenodo.13756832](https://doi.org/10.5281/zenodo.13756832)\n"
            "- `ROBUSTNESS_REVIEW_NOTES.md` im Projektverzeichnis: Einordnung des Robustness Review "
            "und bewusst gesetzte Grenzen\n"
            "- `README.md`: Bedienung, Einheiten und Exportformate"
        )


def render_documentation(parameters):
    inject_css()
    hero(
        "ValidReuse · Dokumentation und Formeln",
        "Vollständige Herleitung aller Größen, die die App berechnet – von der Volumenkorrektur einer "
        "einzelnen Analyse bis zur unteren Kredibilitätsgrenze der Logreduktion. Jeder Abschnitt nennt "
        "die Funktion, die ihn implementiert.",
    )

    with st.sidebar:
        st.markdown("#### Zielwerte")
        rows = "".join(
            f'<div class="vr-row"><span>{p["name"]}</span><b>{float(p["target"]):g} log</b></div>'
            for p in parameters
        )
        st.markdown(f'<div class="vr-side">{rows}</div>', unsafe_allow_html=True)
        st.caption("Voreingestellte Validierungszielwerte. Sie sind in beiden Arbeitsbereichen anpassbar.")

    tabs = st.tabs([
        "Überblick",
        "Volumen und Konzentration",
        "Leitfaden-LRV",
        "Modellbasierte LRV",
        "Diagnostik und Status",
        "Bisheriger Methodenvergleich",
        "Symbole und Grenzen",
    ])
    for tab, section in zip(tabs, [_overview, _volumes, _guideline, _model, _diagnostics, _legacy, _symbols]):
        with tab:
            section()
