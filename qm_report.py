"""Self-contained, printable assessment report; all record content is escaped."""
from html import escape
import json


def _cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return escape(str(value if value is not None else ""))


def _table(rows):
    if not rows:
        return "<p>Keine Einträge.</p>"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return ("<table><thead><tr>" + "".join(f"<th>{_cell(k)}</th>" for k in columns) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_cell(row.get(k))}</td>" for k in columns) +
                                          "</tr>" for row in rows) + "</tbody></table>")


def report_html(record, audit):
    snapshot = record["snapshot"]
    package = record.get("submitted_result")
    result = package["evaluation"] if package else None
    statuses = {"draft": "ENTWURF", "submitted": "ZUR PRÜFUNG", "released": "FREIGEGEBEN", "superseded": "ERSETZT"}
    decision = {"met": "Kriterium erfüllt", "not_met": "Kriterium nicht erfüllt", "not_assessable": "Nicht beurteilbar"}
    events = [json.loads(event["body"]) for event in audit]
    summary = (f"<p>{_cell(decision[result['decision']])}: {result['pairs_passed']} von "
               f"{result['pairs_evaluated']} Paaren erreichen den festgelegten Zielwert. "
               f"Ausgeschlossene Analysen: {result['rows_excluded']}.</p>" if result else
               "<p>Entwurf: keine eingereichte Bewertung.</p>")
    row_details = "".join(f"<h3>Analyse {index}</h3>" + _table([{"Feld": k, "Wert": v} for k, v in row.items()])
                          for index, row in enumerate(snapshot["rows"], 1))
    return ("<!doctype html><html lang='de'><meta charset='utf-8'><title>ValidReuse QM-Bewertung</title>"
            "<style>body{font:14px system-ui,sans-serif;max-width:1100px;margin:36px auto;color:#172033}"
            "table{border-collapse:collapse;width:100%;margin:12px 0;overflow-wrap:anywhere}"
            "th,td{border:1px solid #ccd3dd;padding:7px;text-align:left;vertical-align:top}"
            "th{background:#eef2f7}h2{margin-top:30px}h3{break-after:avoid}"
            "@media print{body{margin:0;font-size:10pt}thead{display:table-header-group}tr{break-inside:avoid}}"
            "</style><body><h1>ValidReuse · QM-Bewertung</h1>"
            f"<h2>{_cell(record['title'])}</h2><p><strong>{statuses[record['status']]}</strong> · "
            f"Revision {record['revision']} · ID {_cell(record['id'])}</p>"
            f"<p>Bearbeiter: {_cell(record['owner'])} · Erstellt: {_cell(record['created'])}<br>"
            f"Vorgänger: {_cell(record.get('parent_id'))}<br>Datensatz-SHA-256: {_cell(record['sha256'])}</p>"
            "<p>Diese Bewertung betrifft das dokumentierte gepaarte Count-LRV-Kriterium. "
            "Sie ersetzt keine Prüfung weiterer Genehmigungsbedingungen.</p><h2>Ergebnis</h2>" + summary +
            (_table(result["pairs"]) if result else "") +
            "<h2>Bericht und Eignungsnachweise</h2>" + _table([{"Feld": k, "Wert": v} for k, v in snapshot["metadata"].items()]) +
            "<h2>Prüfplan</h2>" + _table([{"Feld": k, "Wert": v} for k, v in snapshot["protocol"].items()]) +
            "<h2>Bearbeitung und Freigabe</h2>" + _table(events) +
            "<h2>Einzelbefunde einschließlich Ausschlüssen</h2>" + row_details +
            "<p>Originaldateien, Revisionshistorie und archivierter Rechenstand befinden sich im zugehörigen Prüfarchiv.</p></body></html>")
