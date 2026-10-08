# QM assessment workspace — first milestone

## Intended use

The **QM-Bewertungen** workspace manages reviewed assessments of external laboratory
reports for one trusted team. It preserves source documents, structured findings,
approved-protocol references, data revisions and release history in a local SQLite
database. The existing count/model and legacy workspaces remain available for
exploration; they do not issue QM releases.

The first calculation protocol is **paired-count-lrv/1**:

- Positive influent concentrations are normalised to /100 mL.
- Effluent observations must be exact integer counts with known effective volumes.
- Technical replicates are pooled using the existing `guideline_lrv_pairs` function.
- A pooled zero uses one count over the pooled exposure for the guideline denominator.
- Each paired LRV is compared with the unrounded target.
- The criterion is met if `passing_pairs * 100 >= required_percent * evaluated_pairs`.
- Sampling-plan completeness, minimum pairs, source checks, QC, timestamps, holding
  time and pairing must also pass before submission.

This is a configurable, reviewed assessment criterion. It does not implement every
condition or alternative route in Regulation (EU) 2020/741, a permit, or the FlexTreat
guide. The generic defaults for holding times, sample numbers and targets require a
documented, appropriate protocol. The original regulation describes the proportion
of passing validation samples in Annex I; that proportion is not an interpolated
empirical quantile or the Bayesian lower bound on a population quantile.

## Local setup

Use the same environment as the app:

```bash
cd /home/wseis/validreuse
/home/wseis/miniconda3/envs/validreuse/bin/python qm_store.py add-user analyst --role author
/home/wseis/miniconda3/envs/validreuse/bin/python qm_store.py add-user reviewer --role reviewer
```

Each command prompts twice for a password (minimum 12 characters); passwords are
not passed on the command line. Assign accounts to **different named people**.
Host administrators provision accounts; the web UI has no self-registration or role
editor. There are no default passwords or production test accounts.

Select **QM-Bewertungen** in the app sidebar and log in. Authentication lasts up to
eight hours and can be revoked with logout. Passwords use salted PBKDF2-HMAC-SHA256
with 600,000 iterations. Five failed attempts impose a five-minute account lock.

Local administrators can use `qm_store.py disable-user analyst` to disable an
account and revoke all its sessions, or `qm_store.py set-password analyst` to set a
new password interactively and re-enable it. A password reset also revokes sessions.

The database defaults to `.qm_data/qm.sqlite3` beside the source files. `.qm_data/`
is gitignored. Set `VALIDREUSE_QM_DIR` for an alternative persistent directory and
use the same setting for the server and administrative commands. The CLI also
supports `--directory /absolute/path` before its subcommand.

## Workflow

1. **Create an assessment.** Choose a project title and indicator. Each assessment
   covers one indicator and a documented laboratory-report set.
2. **Archive originals.** Upload PDF, XLSX, CSV or TXT files, at most 20 MiB each.
   The original bytes and SHA-256 digest are retained; filenames are normalised.
   Every analysis must reference an archived document ID and its page/row.
3. **Enter or import findings.** CSV import accepts ValidReuse column names,
   selectable delimiter and decimal separator. It appends unverified rows and never
   overwrites existing data. PDFs and XLSX files are retained as evidence; this
   milestone does not automatically extract their measurements. Transcribe them in
   the editor or prepare a compatible CSV and retain both sources.
4. **Complete metadata and the protocol.** Record lab, report/revision, methods,
   accreditation-scope evidence, sampling/transport evidence, planned pair IDs,
   sampling points, permitted time intervals, decision settings, uncertainty and
   exclusion rules, and the protocol's approval reference. Reference archived
   supporting documents in these fields. The app checks completeness; the reviewer
   verifies the substantive evidence and its applicability.
5. **Verify every source transcription.** Preserve the original reported expression
   separately from its numeric input. Enter sample, receipt and analysis times as
   ISO timestamps with timezone, e.g. `2026-09-15T09:00+02:00`. Record accepted QC and
   its evidence for included analyses. Save data first, then perform and record the
   source check. New rows, changed rows (including renamed samples), changed report
   metadata and amendments require fresh source checks.
6. **Document exclusions.** Mark a row `excluded` and give a reason; never replace an
   unfavourable result with `missing`. Planned missing samples must be recorded.
   Excluded rows remain in the assessment, archive and report. An included row with
   no matching pair blocks submission. The reviewer must evaluate the suitability
   of exclusions and the resulting sample set under the approved protocol.
7. **Save and submit.** Every saved revision requires a reason. Submission assesses
   the saved revision and locks it. Unsaved editor changes are not submitted.
   Outstanding issues prevent submission. The exact result, source-code files,
   Python/package versions and input hash are archived at submission.
8. **Independent review.** A reviewer account can return a submitted record with a
   comment or release it after confirming review of originals, protocol, QC and
   exclusions. Author accounts cannot release. A valid assessment whose LRV
   criterion is *not met* can be released: release means the assessment was reviewed,
   not that the water meets its requirements.
9. **Export.** The ZIP contains `report.html` (printable in a browser),
   `assessment.json`, `audit.json`, all `revisions.json`, document metadata and original
   files. Submitted/released records also include the archived calculation source.
   Draft, submitted, released and superseded states are clearly identified.
10. **Amend.** A released assessment cannot be edited. An author can create a linked
    draft amendment with a reason. The previous release remains current until the
    amendment is independently released, when it becomes superseded. Competing
    amendments cannot both supersede the same current release.

## Data protection, backup and restoration

Database updates use transactions and optimistic revision checks. Immutable-table
triggers prevent normal updates/deletes of originals, revisions and audit events.
Hash checks are run before submission, review and export. These protections do not
make records tamper-proof against a privileged filesystem/database administrator.
The digest establishes file integrity, not authenticity of the lab or report issuer.

This milestone assumes a **single trusted laboratory team**: authenticated accounts
can read all assessments; authors can edit only their own drafts. There is no
tenant or project-level read isolation. Run on localhost or a protected network.
For remote access, configure HTTPS, an authenticated reverse proxy as appropriate,
restricted host access and protected backups. Database contents are not encrypted
by the app; apply filesystem/volume encryption if required by the deployment.

Create a consistent backup, including document BLOBs, to a **new** file:

```bash
/home/wseis/miniconda3/envs/validreuse/bin/python qm_store.py backup /absolute/backup/path/qm-2026-09-15.sqlite3
```

Existing destinations are not overwritten. Configure scheduled backups, independent
retention and access controls operationally. An assessment ZIP is useful for review
but is not a replacement for a full database backup.

To test recovery, copy a backup to a separate protected directory as `qm.sqlite3`,
point an isolated app instance at that directory using `VALIDREUSE_QM_DIR`, and check
login, archived-document downloads, assessment history and released exports. Do not
restore over a running production database. Backups contain password hashes and
session records as well as laboratory data and must be protected accordingly.

## Validation evidence and remaining scope

Run the automated suite:

```bash
/home/wseis/miniconda3/envs/validreuse/bin/python -m unittest discover -s tests -v
```

`tests/test_qm_workflow.py` covers the count-vs-quantile boundary, invalid metadata,
unresolved QC, censored/missing data, exclusions, sampling-plan completeness,
permissions, stale revisions, source re-verification, submission, independent
release, negative conclusions, immutable records, restart persistence, amendments,
archives and the Streamlit workflow. Existing volume and model tests remain relevant.

Before routine official use, the laboratory must validate its configured intended
use with representative real reports and approved acceptance criteria, document
user acceptance, approve procedures/training and verify its deployment and recovery
arrangements. This implementation is not an accreditation claim or a completed
laboratory software-validation dossier.

Remaining work includes automatic lab/accreditation registry verification, PDF
extraction with verified mappings, validated censored/MPN or reported-concentration
analysis, measurement/recovery uncertainty propagation, a fuller regulatory rule
catalogue, internal-laboratory batch QC and equipment/LIMS integration, enterprise
identity management, externally anchored audit records and automated retention.

References for protocol owners:

- [DAkkS accreditation scope](https://www.dakks.de/de/schritt-1-grundlagen.html)
- [ILAC G8 decision rules and G17 measurement uncertainty](https://ilac.org/publications-and-resources/ilac-guidance-series/)
- [Regulation (EU) 2020/741](https://eur-lex.europa.eu/eli/reg/2020/741/oj)
