"""Assay-level records and explicit original-water exposure accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["sample_id", "replicate_id", "date", "location", "pair_id", "result_type",
           "value", "unit", "volume_method", "filtered_ml", "eluate_ml", "assayed_ml",
           "dilution_factor", "recovery", "effective_ml", "recovery_source", "notes"]
RESULT_TYPES = ["count", "concentration", "<LOD", "<LOQ", "interval", "MPN", "missing"]
VOLUME_METHODS = ["effective", "direct", "concentrated"]


NUMERIC_COLUMNS = ["value", "filtered_ml", "eluate_ml", "assayed_ml", "dilution_factor", "recovery", "effective_ml"]


def editor_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.reindex(columns=COLUMNS).copy()
    for name in COLUMNS:
        if name in NUMERIC_COLUMNS:
            table[name] = pd.to_numeric(table[name].replace("", np.nan), errors="raise").astype(float)
        else:
            table[name] = table[name].astype("string")
    return table


def empty_table() -> pd.DataFrame:
    return editor_table(pd.DataFrame(columns=COLUMNS))


def example_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ins, outs = [], []
    for i, k in enumerate([3, 9, 8, 2, 6, 7, 4, 6, 10, 2, 18, 5]):
        ins.append(dict(sample_id=f"IN-{i+1}", replicate_id="1", pair_id=f"P-{i+1}", location="Zulauf",
                        result_type="concentration", value=8_000_000 + i*210_000,
                        unit="/100 mL"))
        outs.append(dict(sample_id=f"OUT-{i+1}", replicate_id="1", pair_id=f"P-{i+1}", location="Ablauf",
                         result_type="count", value=k, unit="count", volume_method="concentrated",
                         filtered_ml=15000., eluate_ml=400., assayed_ml=100.,
                         dilution_factor=1., recovery=0.5, recovery_source="Beispielannahme"))
    return (editor_table(pd.DataFrame(ins)), editor_table(pd.DataFrame(outs)))


def _number(row, name, positive=True):
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        raise ValueError(f"{name}: Zahl erforderlich.") from None
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name}: endliche {'positive' if positive else 'nichtnegative'} Zahl erforderlich.")
    return value


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def effective_volume(row) -> float:
    method = _text(row.get("volume_method"))
    if method == "effective":
        # An externally calculated exposure already includes all corrections.
        volume = _number(row, "effective_ml")
        if not _text(row.get("recovery_source")):
            raise ValueError("Quelle/Annahme fuer das effektive Volumen fehlt (recovery_source).")
        return volume
    recovery = _number(row, "recovery")
    if recovery > 1:
        raise ValueError("Wiederfindung als Anteil > 0 bis 1 eingeben, nicht als Prozentzahl.")
    if not _text(row.get("recovery_source")):
        raise ValueError("Quelle/Annahme der Wiederfindung fehlt (recovery_source).")
    assayed = _number(row, "assayed_ml")
    dilution = _number(row, "dilution_factor")
    if dilution < 1:
        raise ValueError("Verduennungsfaktor muss mindestens 1 sein (z.B. 10 fuer 1:10).")
    equivalent = assayed / dilution
    if method == "direct":
        volume = equivalent * recovery
    elif method == "concentrated":
        filtered = _number(row, "filtered_ml")
        eluate = _number(row, "eluate_ml")
        if equivalent > eluate:
            raise ValueError("Analysiertes Original-Eluatvolumen ist groesser als das gesamte Eluat.")
        volume = filtered * (equivalent / eluate) * recovery
    else:
        raise ValueError("Volumenmethode effective, direct oder concentrated waehlen.")
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Berechnetes effektives Volumen ist ungueltig.")
    return volume


def validate_table(table: pd.DataFrame, side: str) -> pd.DataFrame:
    """Retain row identity; missing results are not zeros. Reject unsupported assays."""
    table = table.reindex(columns=COLUMNS)
    records, errors = [], []
    for index, row in table.iterrows():
        if all(not _text(v) for v in row):
            continue
        if _text(row.get("result_type")) == "missing":
            continue
        try:
            sample = _text(row.get("sample_id"))
            if not sample:
                raise ValueError("Proben-ID fehlt.")
            kind = _text(row.get("result_type"))
            if kind not in (["concentration"] if side == "in" else ["count"]):
                raise ValueError("Zulauf: positive Konzentration erforderlich." if side == "in" else
                                 "Ablauf: exakte Rohzaehlung erforderlich. MPN, Konzentration und zensierte Ergebnisse werden nicht als Counts umgedeutet.")
            value = _number(row, "value", positive=side == "in")
            record = row.to_dict()
            record.update(sample_id=sample, value=value, source_row=str(index))
            if side == "in":
                unit = _text(row.get("unit"))
                if unit not in ["/100 mL", "/mL", "/L"]:
                    raise ValueError("Konzentrationseinheit /100 mL, /mL oder /L angeben.")
                record["concentration_100ml"] = value * {"/100 mL": 1., "/mL": 100., "/L": 0.1}[unit]
            else:
                if not value.is_integer() or value > 2**53-1:
                    raise ValueError("Rohzaehlung muss eine nichtnegative, exakt darstellbare ganze Zahl sein.")
                if _text(row.get("unit")) != "count":
                    raise ValueError("Rohzaehlung benoetigt Einheit count.")
                record["value"] = int(value)
                record["effective_volume_ml"] = effective_volume(row)
                record["exposure"] = record["effective_volume_ml"] / 100.
                record["concentration_100ml"] = value / record["exposure"]
                record["zero_convention_100ml"] = max(value, 1.) / record["exposure"]
                record["replicate_id"] = _text(row.get("replicate_id"))
                if not record["replicate_id"]:
                    raise ValueError("Replikat-ID fehlt (fuer Einzelmessung z.B. 1).")
            if not np.isfinite(record["concentration_100ml"]):
                raise ValueError("Konzentration nach Einheitenumrechnung ist nicht endlich.")
            records.append(record)
        except ValueError as exc:
            errors.append(f"Zeile {index + 1}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    result = pd.DataFrame(records)
    if result.empty:
        return result
    keys = ["sample_id"] if side == "in" else ["sample_id", "replicate_id"]
    if result.duplicated(keys).any():
        raise ValueError("Doppelte Proben-ID" + ("/Replikat-ID." if side == "out" else ": Zulaufreplikate vorher fachlich zusammenfassen."))
    if side == "out":
        for sample, group in result.groupby("sample_id", sort=False):
            for field in ["date", "location", "pair_id"]:
                if len({_text(v) for v in group[field]}) > 1:
                    raise ValueError(f"Probe {sample}: Replikate haben widerspruechliche {field}-Angaben.")
            if len(group) > 1 and set(group.volume_method) == {"concentrated"}:
                for field in ["filtered_ml", "eluate_ml", "recovery"]:
                    if group[field].astype(float).nunique() != 1:
                        raise ValueError(f"Probe {sample}: Replikate desselben Eluats benoetigen identisches {field}.")
                total = (group.assayed_ml.astype(float)/group.dilution_factor.astype(float)).sum()
                if total > float(group.eluate_ml.iloc[0]) * (1 + 1e-12):
                    raise ValueError(f"Probe {sample}: Replikate verbrauchen mehr als das gesamte Eluat.")
    return result


def model_arrays(inflow, outflow, paired=False):
    """Arrays for the count candidates.

    Unpaired (the default) passes every influent concentration, so historical
    influent data stay usable. ``paired=True`` returns exactly one influent
    concentration per environmental effluent sample, matched by pair ID and
    ordered by the effluent sample grouping, as the conditional candidates in
    ``count_models`` require.
    """
    if len(inflow) < 2 or outflow.empty or outflow.sample_id.nunique() < 2:
        raise ValueError("Mindestens zwei unabhaengige Zulauf- und Ablaufproben erforderlich.")
    if not paired:
        groups, _ = pd.factorize(outflow.sample_id, sort=False)
        return dict(vals_zu=inflow.concentration_100ml.tolist(), vals_ab=outflow.value.tolist(),
                    exposures=outflow.exposure.tolist(), sample_groups=groups.tolist())

    in_pairs = inflow.assign(pair_id=inflow.pair_id.map(_text))
    out_pairs = outflow.assign(pair_id=outflow.pair_id.map(_text))
    in_pairs = in_pairs[in_pairs.pair_id != ""]
    out_pairs = out_pairs[out_pairs.pair_id != ""]
    if in_pairs.empty or out_pairs.empty:
        raise ValueError("Bedingte Kandidaten benoetigen Paar-IDs in beiden Tabellen.")
    if in_pairs.duplicated("pair_id").any():
        duplicates = sorted(in_pairs.loc[in_pairs.duplicated("pair_id", keep=False), "pair_id"].unique())
        raise ValueError("Paar-ID im Zulauf nicht eindeutig: " + ", ".join(duplicates))
    ambiguous = out_pairs.groupby("pair_id").sample_id.nunique()
    if (ambiguous > 1).any():
        raise ValueError("Paar-ID verweist auf mehrere unabhaengige Ablaufproben: "
                         + ", ".join(sorted(ambiguous.index[ambiguous > 1])))
    matched = out_pairs[out_pairs.pair_id.isin(set(in_pairs.pair_id))]
    if matched.sample_id.nunique() < 2:
        raise ValueError("Bedingte Kandidaten benoetigen mindestens zwei vollstaendige Paare "
                         "aus Zulaufkonzentration und Ablaufprobe.")
    groups, keys = pd.factorize(matched.sample_id, sort=False)
    # One influent concentration per effluent sample, in the grouping's own order.
    pair_of_sample = matched.drop_duplicates("sample_id").set_index("sample_id").pair_id
    concentration = in_pairs.set_index("pair_id").concentration_100ml
    return dict(vals_zu=[float(concentration[pair_of_sample[key]]) for key in keys],
                vals_ab=matched.value.tolist(), exposures=matched.exposure.tolist(),
                sample_groups=groups.tolist())


def guideline_lrv_pairs(inflow: pd.DataFrame, outflow: pd.DataFrame) -> pd.DataFrame:
    """Calculate the guide's count-scale LRV for explicitly matched samples.

    Technical effluent replicates are pooled by environmental sample. Their raw
    counts and disjoint effective volumes are additive. Pair IDs, rather than row
    order or sample IDs, define the influent/effluent match.

    The guide (Kap. 4.6.1) raises the *influent* onto the representative effluent
    volume ``V_r`` by the factor ``V_r/100 mL`` (Eq. 4-3), because a negative
    effluent finding must not be scaled up to a larger volume. Dividing the count
    down to 100 mL instead, as ``effluent_denominator_100ml`` does, is the same
    ratio and is kept for reporting; ``guideline_lrv`` follows the guide's own
    direction so the code mirrors the document.
    """
    columns = ["pair_id", "inflow_sample_id", "outflow_sample_id", "count",
               "effective_volume_ml", "reference_factor", "inflow_concentration_100ml",
               "inflow_concentration_reference", "effluent_denominator_100ml", "guideline_lrv"]
    if inflow.empty or outflow.empty:
        return pd.DataFrame(columns=columns)

    in_pairs = inflow.assign(pair_id=inflow.pair_id.map(_text))
    out_pairs = outflow.assign(pair_id=outflow.pair_id.map(_text))
    in_pairs = in_pairs[in_pairs.pair_id != ""]
    out_pairs = out_pairs[out_pairs.pair_id != ""]
    if in_pairs.empty or out_pairs.empty:
        return pd.DataFrame(columns=columns)
    if in_pairs.duplicated("pair_id").any():
        duplicates = sorted(in_pairs.loc[in_pairs.duplicated("pair_id", keep=False), "pair_id"].unique())
        raise ValueError("Paar-ID im Zulauf nicht eindeutig: " + ", ".join(duplicates))

    grouped = (out_pairs.groupby(["pair_id", "sample_id"], sort=False, as_index=False)
               .agg(count=("value", "sum"), effective_volume_ml=("effective_volume_ml", "sum")))
    if grouped.duplicated("pair_id").any():
        duplicates = sorted(grouped.loc[grouped.duplicated("pair_id", keep=False), "pair_id"].unique())
        raise ValueError("Paar-ID verweist auf mehrere unabhaengige Ablaufproben: " + ", ".join(duplicates))

    paired = grouped.merge(
        in_pairs[["pair_id", "sample_id", "concentration_100ml"]],
        on="pair_id", how="inner", suffixes=("_out", "_in"), validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame(columns=columns)
    floored_count = np.maximum(paired["count"].to_numpy(dtype=float), 1.)
    # Eq. 4-3: the influent is expressed per representative effluent volume.
    paired["reference_factor"] = paired["effective_volume_ml"].to_numpy(dtype=float) / 100.
    paired["inflow_concentration_reference"] = (
        paired["concentration_100ml"].to_numpy(dtype=float) * paired["reference_factor"]
    )
    paired["effluent_denominator_100ml"] = floored_count / paired["reference_factor"]
    paired["guideline_lrv"] = np.log10(
        paired["inflow_concentration_reference"] / floored_count
    )
    return paired.rename(columns={
        "sample_id_in": "inflow_sample_id", "sample_id_out": "outflow_sample_id",
        "concentration_100ml": "inflow_concentration_100ml",
    })[columns]
