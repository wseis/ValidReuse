"""Local QM records, authenticated transitions, immutable revisions and audit export.

Filesystem administrators remain trusted. Hash chains detect accidental changes;
they are not a substitute for independent backups or an external audit service.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import hmac
import importlib.metadata
import io
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import time
import uuid
import zipfile

from qm_rules import evaluate, new_snapshot


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def software_record():
    root = Path(__file__).parent
    sources = {name: (root / name).read_text() for name in (
        "qm_rules.py", "qm_store.py", "qm_ui.py", "qm_report.py", "sample_data.py", "requirements.txt",
    )}
    return {"sources": sources, "sha256": {k: digest(v.encode()) for k, v in sources.items()}, "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "streamlit")}}


class QMStore:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get("VALIDREUSE_QM_DIR", Path(__file__).parent / ".qm_data"))
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "qm.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, salt TEXT NOT NULL, password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('author','reviewer')),
                    failures INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES users,
                    expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, owner TEXT NOT NULL REFERENCES users,
                    status TEXT NOT NULL, revision INTEGER NOT NULL, created TEXT NOT NULL,
                    parent_id TEXT REFERENCES assessments, submitted_result TEXT
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    assessment_id TEXT NOT NULL REFERENCES assessments, revision INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY(assessment_id, revision)
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, assessment_id TEXT NOT NULL REFERENCES assessments,
                    filename TEXT NOT NULL, sha256 TEXT NOT NULL, content BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, assessment_id TEXT NOT NULL REFERENCES assessments,
                    body TEXT NOT NULL, previous_hash TEXT NOT NULL, sha256 TEXT NOT NULL
                );
            """)
            if "enabled" not in {row[1] for row in db.execute("PRAGMA table_info(users)")}:
                db.execute("ALTER TABLE users ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
            for table in ("revisions", "documents", "events"):
                for operation in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} "
                               "BEGIN SELECT RAISE(ABORT, 'Immutable QM record'); END")
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def provision_user(self, username, password, role):
        """Host-admin operation, exposed only via the local CLI, never the web UI."""
        username = username.strip().casefold()
        if not username or len(password) < 12 or role not in ("author", "reviewer"):
            raise ValueError("Benutzername, mindestens 12 Passwortzeichen und author/reviewer erforderlich.")
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
        with self.transaction() as db:
            db.execute("INSERT INTO users(username,salt,password_hash,role) VALUES(?,?,?,?)", (username, salt, key, role))

    def login(self, username, password):
        username = username.strip().casefold()
        token = None
        with self.transaction() as db:
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            salt = user["salt"] if user else "00" * 16
            key = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
            if user and user["enabled"] and user["locked_until"] <= time.time() and hmac.compare_digest(key, user["password_hash"]):
                token = secrets.token_urlsafe(32)
                db.execute("INSERT INTO sessions VALUES(?,?,?)", (digest(token.encode()), username, time.time() + 8 * 3600))
                db.execute("UPDATE users SET failures=0,locked_until=0 WHERE username=?", (username,))
            elif user:
                failures = user["failures"] + 1
                locked = max(user["locked_until"], time.time() + 300) if failures >= 5 else 0
                db.execute("UPDATE users SET failures=?,locked_until=? WHERE username=?", (failures, locked, username))
        if token is None:
            raise ValueError("Anmeldung fehlgeschlagen oder Konto vorübergehend gesperrt.")
        return token

    def _actor(self, db, token):
        row = db.execute("SELECT u.username,u.role FROM sessions s JOIN users u USING(username) "
                         "WHERE s.token_hash=? AND s.expires>? AND u.enabled=1", (digest(token.encode()), time.time())).fetchone()
        if row is None:
            raise PermissionError("Bitte erneut anmelden.")
        return dict(row)

    def actor(self, token):
        with self.connection() as db:
            return self._actor(db, token)

    def logout(self, token):
        with self.transaction() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (digest(token.encode()),))

    def disable_user(self, username):
        """Trusted host-admin operation: immediately revoke existing sessions."""
        username = username.strip().casefold()
        with self.transaction() as db:
            if not db.execute("UPDATE users SET enabled=0 WHERE username=?", (username,)).rowcount:
                raise ValueError("Konto nicht gefunden.")
            db.execute("DELETE FROM sessions WHERE username=?", (username,))

    def reset_password(self, username, password):
        """Trusted host-admin operation; resetting also re-enables a disabled account."""
        if len(password) < 12:
            raise ValueError("Mindestens 12 Passwortzeichen erforderlich.")
        username, salt = username.strip().casefold(), secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
        with self.transaction() as db:
            if not db.execute("UPDATE users SET salt=?,password_hash=?,enabled=1,failures=0,locked_until=0 WHERE username=?",
                              (salt, key, username)).rowcount:
                raise ValueError("Konto nicht gefunden.")
            db.execute("DELETE FROM sessions WHERE username=?", (username,))

    def _event(self, db, assessment_id, actor, action, detail):
        previous = db.execute("SELECT sha256 FROM events WHERE assessment_id=? ORDER BY seq DESC LIMIT 1", (assessment_id,)).fetchone()
        previous_hash = previous[0] if previous else ""
        body = canonical(dict(at=now(), actor=actor, action=action, detail=detail))
        sha = digest((previous_hash + body).encode())
        db.execute("INSERT INTO events(assessment_id,body,previous_hash,sha256) VALUES(?,?,?,?)",
                   (assessment_id, body, previous_hash, sha))

    def _revision(self, db, assessment_id, revision, snapshot):
        body = canonical(snapshot)
        db.execute("INSERT INTO revisions VALUES(?,?,?,?)", (assessment_id, revision, body, digest(body.encode())))

    def _get(self, db, assessment_id):
        row = db.execute("SELECT a.*,r.snapshot,r.sha256 FROM assessments a JOIN revisions r "
                         "ON a.id=r.assessment_id AND a.revision=r.revision WHERE a.id=?", (assessment_id,)).fetchone()
        if row is None:
            raise ValueError("Bewertung nicht gefunden.")
        result = dict(row)
        result["snapshot"] = json.loads(result["snapshot"])
        result["submitted_result"] = json.loads(result["submitted_result"]) if result["submitted_result"] else None
        return result

    def _editable(self, actor, record, revision):
        if actor["role"] != "author" or actor["username"] != record["owner"]:
            raise PermissionError("Nur der verantwortliche Bearbeiter darf den Entwurf ändern.")
        if record["status"] != "draft":
            raise ValueError("Bewertung ist gesperrt. Rückgabe oder Nachtrag erforderlich.")
        if record["revision"] != revision:
            raise ValueError("Zwischenzeitliche Änderung: Bewertung neu laden.")

    def create(self, token, title, snapshot=None):
        if not title.strip():
            raise ValueError("Titel fehlt.")
        with self.transaction() as db:
            actor = self._actor(db, token)
            if actor["role"] != "author":
                raise PermissionError("Bearbeiterkonto erforderlich.")
            assessment_id = uuid.uuid4().hex
            db.execute("INSERT INTO assessments VALUES(?,?,?,?,?,?,?,?)", (assessment_id, title.strip(), actor["username"], "draft", 1, now(), None, None))
            initial = json.loads(canonical(snapshot or new_snapshot()))
            for row in initial["rows"]:
                row.update(_row_id=uuid.uuid4().hex, source_verified=False)
            self._revision(db, assessment_id, 1, initial)
            self._event(db, assessment_id, actor["username"], "created", {"revision": 1})
            return assessment_id

    def list(self, token):
        with self.connection() as db:
            self._actor(db, token)
            return [dict(r) for r in db.execute("SELECT id,title,owner,status,revision,parent_id FROM assessments ORDER BY created DESC")]

    def get(self, token, assessment_id):
        with self.connection() as db:
            self._actor(db, token)
            return self._get(db, assessment_id)

    def save(self, token, assessment_id, revision, snapshot, reason):
        if not reason.strip():
            raise ValueError("Änderungsgrund erforderlich.")
        with self.transaction() as db:
            actor, record = self._actor(db, token), self._get(db, assessment_id)
            self._editable(actor, record, revision)
            # A source check must be repeated after any change to a previously saved row.
            old_rows = record["snapshot"].get("rows", [])
            old_by_id = {r["_row_id"]: r for r in old_rows}
            snapshot = json.loads(canonical(snapshot))
            metadata_changed = snapshot["metadata"] != record["snapshot"]["metadata"]
            used_ids = set()
            for row in snapshot.get("rows", []):
                old = old_by_id.get(row.get("_row_id"))
                if not old or row.get("_row_id") in used_ids:
                    row["_row_id"] = uuid.uuid4().hex
                    row["source_verified"] = False
                elif metadata_changed or any(old.get(k) != row.get(k) for k in set(old) | set(row) if k != "source_verified"):
                    row["source_verified"] = False
                used_ids.add(row["_row_id"])
            self._revision(db, assessment_id, revision + 1, snapshot)
            db.execute("UPDATE assessments SET revision=? WHERE id=?", (revision + 1, assessment_id))
            self._event(db, assessment_id, actor["username"], "saved", {"revision": revision + 1, "reason": reason.strip(), "sha256": digest(snapshot)})

    def add_document(self, token, assessment_id, revision, filename, content):
        filename = Path(filename.replace("\\", "/")).name
        if not filename or Path(filename).suffix.lower() not in (".pdf", ".csv", ".xlsx", ".txt") or not 0 < len(content) <= 20 * 1024 * 1024:
            raise ValueError("PDF, CSV, XLSX oder TXT mit 1 Byte bis 20 MiB erforderlich.")
        with self.transaction() as db:
            actor, record = self._actor(db, token), self._get(db, assessment_id)
            self._editable(actor, record, revision)
            document_id = uuid.uuid4().hex
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (document_id, assessment_id, filename, digest(content), content))
            self._revision(db, assessment_id, revision + 1, record["snapshot"])
            db.execute("UPDATE assessments SET revision=? WHERE id=?", (revision + 1, assessment_id))
            self._event(db, assessment_id, actor["username"], "document_added", {"document_id": document_id, "filename": filename, "sha256": digest(content), "revision": revision + 1})
            return document_id

    def documents(self, token, assessment_id):
        with self.connection() as db:
            self._actor(db, token)
            return [dict(r) for r in db.execute("SELECT id,filename,sha256 FROM documents WHERE assessment_id=?", (assessment_id,))]

    def document(self, token, assessment_id, document_id):
        with self.connection() as db:
            self._actor(db, token)
            row = db.execute("SELECT * FROM documents WHERE id=? AND assessment_id=?", (document_id, assessment_id)).fetchone()
            if row is None or digest(row["content"]) != row["sha256"]:
                raise ValueError("Dokument fehlt oder Integritätsprüfung fehlgeschlagen.")
            return dict(row)

    def _verify(self, db, assessment_id):
        previous = ""
        for row in db.execute("SELECT * FROM events WHERE assessment_id=? ORDER BY seq", (assessment_id,)):
            if row["previous_hash"] != previous or digest((previous + row["body"]).encode()) != row["sha256"]:
                raise ValueError("Audit-Integritätsprüfung fehlgeschlagen.")
            previous = row["sha256"]
        for row in db.execute("SELECT snapshot,sha256 FROM revisions WHERE assessment_id=?", (assessment_id,)):
            if digest(row["snapshot"].encode()) != row["sha256"]:
                raise ValueError("Revisions-Integritätsprüfung fehlgeschlagen.")
        for row in db.execute("SELECT content,sha256 FROM documents WHERE assessment_id=?", (assessment_id,)):
            if digest(row["content"]) != row["sha256"]:
                raise ValueError("Dokument-Integritätsprüfung fehlgeschlagen.")

    def submit(self, token, assessment_id, revision):
        with self.transaction() as db:
            actor, record = self._actor(db, token), self._get(db, assessment_id)
            self._editable(actor, record, revision)
            self._verify(db, assessment_id)
            docs = {r[0] for r in db.execute("SELECT id FROM documents WHERE assessment_id=?", (assessment_id,))}
            result = evaluate(record["snapshot"], docs)
            if result["issues"]:
                raise ValueError("Offene Prüfpunkte verhindern die Einreichung: " + " / ".join(result["issues"][:3]))
            package = {"evaluation": result, "software": software_record(), "snapshot_sha256": record["sha256"]}
            db.execute("UPDATE assessments SET status='submitted',submitted_result=? WHERE id=?", (canonical(package), assessment_id))
            self._event(db, assessment_id, actor["username"], "submitted", {"revision": revision, "result_sha256": digest(package)})

    def review(self, token, assessment_id, revision, release, comment):
        if not comment.strip():
            raise ValueError("Prüfvermerk erforderlich; Quellen, Prüfplan und Ausschlüsse ausdrücklich beurteilen.")
        with self.transaction() as db:
            actor, record = self._actor(db, token), self._get(db, assessment_id)
            if actor["role"] != "reviewer" or actor["username"] == record["owner"]:
                raise PermissionError("Unabhängiges Prüferkonto erforderlich.")
            if record["status"] != "submitted" or record["revision"] != revision:
                raise ValueError("Nur die aktuelle eingereichte Revision kann geprüft werden.")
            self._verify(db, assessment_id)
            if release:
                docs = {r[0] for r in db.execute("SELECT id FROM documents WHERE assessment_id=?", (assessment_id,))}
                package = record["submitted_result"]
                if (package["snapshot_sha256"] != record["sha256"] or
                    package["software"]["sha256"] != software_record()["sha256"] or
                    package["software"]["packages"] != software_record()["packages"] or
                    package["software"]["python"] != sys.version or
                    package["evaluation"] != evaluate(record["snapshot"], docs)):
                    raise ValueError("Daten oder Rechenversion geändert: zurückgeben und erneut einreichen.")
                if record["parent_id"]:
                    parent = self._get(db, record["parent_id"])
                    if parent["status"] != "released":
                        raise ValueError("Vorgänger wurde bereits ersetzt; Nachtrag vom aktuellen Stand erstellen.")
                    db.execute("UPDATE assessments SET status='superseded' WHERE id=?", (record["parent_id"],))
                    self._event(db, record["parent_id"], actor["username"], "superseded", {"successor": assessment_id})
            db.execute("UPDATE assessments SET status=?,submitted_result=? WHERE id=?",
                       ("released" if release else "draft", canonical(record["submitted_result"]) if release else None, assessment_id))
            self._event(db, assessment_id, actor["username"], "released" if release else "returned", {"revision": revision, "comment": comment.strip()})

    def amend(self, token, assessment_id, reason):
        if not reason.strip():
            raise ValueError("Nachtragsgrund erforderlich.")
        with self.transaction() as db:
            actor, record = self._actor(db, token), self._get(db, assessment_id)
            if actor["role"] != "author" or record["status"] != "released":
                raise PermissionError("Nachtrag nur durch Bearbeiter aus einer aktuellen Freigabe.")
            self._verify(db, assessment_id)
            new_id, snapshot = uuid.uuid4().hex, record["snapshot"]
            db.execute("INSERT INTO assessments VALUES(?,?,?,?,?,?,?,?)", (new_id, record["title"], actor["username"], "draft", 1, now(), assessment_id, None))
            doc_map = {}
            for doc in db.execute("SELECT * FROM documents WHERE assessment_id=?", (assessment_id,)).fetchall():
                new_doc = uuid.uuid4().hex
                doc_map[doc["id"]] = new_doc
                db.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (new_doc, new_id, doc["filename"], doc["sha256"], doc["content"]))
            for row in snapshot["rows"]:
                row["document_id"] = doc_map.get(row.get("document_id"), "")
                row["source_verified"] = False
            self._revision(db, new_id, 1, snapshot)
            self._event(db, new_id, actor["username"], "amendment_created", {"parent": assessment_id, "reason": reason.strip()})
            return new_id

    def history(self, token, assessment_id):
        with self.connection() as db:
            self._actor(db, token)
            self._verify(db, assessment_id)
            return {
                "events": [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE assessment_id=? ORDER BY seq", (assessment_id,))],
                "revisions": [dict(r) for r in db.execute("SELECT revision,snapshot,sha256 FROM revisions WHERE assessment_id=? ORDER BY revision", (assessment_id,))],
            }

    def export(self, token, assessment_id):
        from qm_report import report_html
        with self.connection() as db:
            db.execute("BEGIN")
            self._actor(db, token)
            self._verify(db, assessment_id)
            record = self._get(db, assessment_id)
            events = [dict(r) for r in db.execute("SELECT * FROM events WHERE assessment_id=? ORDER BY seq", (assessment_id,))]
            revisions = [dict(r) for r in db.execute("SELECT * FROM revisions WHERE assessment_id=? ORDER BY revision", (assessment_id,))]
            documents = [dict(r) for r in db.execute("SELECT * FROM documents WHERE assessment_id=?", (assessment_id,))]
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("assessment.json", json.dumps(record, ensure_ascii=False, indent=2))
                archive.writestr("report.html", report_html(record, events))
                archive.writestr("audit.json", json.dumps(events, ensure_ascii=False, indent=2))
                archive.writestr("revisions.json", json.dumps(revisions, ensure_ascii=False, indent=2))
                archive.writestr("documents.json", json.dumps([{k: v for k, v in d.items() if k != "content"} for d in documents], indent=2))
                for doc in documents:
                    archive.writestr(f"documents/{doc['id']}/{doc['filename']}", doc["content"])
                if record["submitted_result"]:
                    for name, source in record["submitted_result"]["software"]["sources"].items():
                        archive.writestr(f"software/{name}", source)
            return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description="ValidReuse QM: lokale Konten und konsistente Sicherung")
    parser.add_argument("--directory", help="QM-Verzeichnis (Standard: VALIDREUSE_QM_DIR oder .qm_data)")
    commands = parser.add_subparsers(dest="command", required=True)
    user = commands.add_parser("add-user")
    user.add_argument("username")
    user.add_argument("--role", required=True, choices=["author", "reviewer"])
    commands.add_parser("disable-user").add_argument("username")
    commands.add_parser("set-password").add_argument("username")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", help="Neue SQLite-Datei; vorhandene Dateien werden nicht überschrieben")
    args = parser.parse_args()
    store = QMStore(args.directory)
    if args.command in ("add-user", "set-password"):
        password = getpass.getpass("Passwort (mindestens 12 Zeichen): ")
        if password != getpass.getpass("Passwort wiederholen: "):
            raise SystemExit("Passwörter stimmen nicht überein.")
        if args.command == "add-user":
            store.provision_user(args.username, password, args.role)
        else:
            store.reset_password(args.username, password)
        print("Konto eingerichtet.")
    elif args.command == "disable-user":
        store.disable_user(args.username)
        print("Konto und bestehende Sitzungen gesperrt.")
    else:
        destination = Path(args.destination)
        with destination.open("xb"):
            pass
        destination.chmod(0o600)
        with store.connection() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        print(f"Sicherung einschließlich Originaldokumenten: {destination}")


if __name__ == "__main__":
    main()
