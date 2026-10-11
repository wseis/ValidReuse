# ValidReuse Method Explorer

Streamlit app for comparing q10(LRV) evaluation methods on the same editable integer input table:

- Empirical bootstrap
- Bayesian approximation
- Bayesian (PyMC)

## Use the local Conda environment

```zsh
conda activate validreuse
```

## Install dependencies

```zsh
pip install -r requirements.txt
```

Recommended:
- Use Python 3.13 for the full app, including PyMC.
- The older `validreuse/` Python 3.14 environment is still present, but PyMC is not reliable there.

## Run the app

```zsh
streamlit run app.py
```

## Deploy With Docker

Build the image:

```zsh
docker build -t validreuse-app .
```

Run it locally:

```zsh
docker run --rm -p 8501:8501 validreuse-app
```

Then open `http://localhost:8501`.

## Deploy On A Server

1. Copy the project to the server.
2. Install Docker.
3. Build the image:

```zsh
docker build -t validreuse-app .
```

4. Run the container in the background:

```zsh
docker run -d --name validreuse -p 8501:8501 --restart unless-stopped validreuse-app
```

5. Put Nginx or Caddy in front of port `8501` if you want HTTPS and a public domain.

## Streamlit Cloud

This app can also be deployed from a GitHub repo on Streamlit Community Cloud with `app.py` as the entrypoint.

Note:
- PyMC makes deployment heavier than a simple Streamlit app.
- If you want the lightest deployment, we can make the PyMC method optional or disable it in cloud builds.

## Quality-managed assessments

The **QM-Bewertungen** workspace adds persistent assessments, original-report
retention, authenticated author/reviewer roles, documented suitability checks,
versioned paired-LRV decision protocols, controlled exclusions, immutable revision
history, independent release and linked amendments. Exports include a printable
HTML report, originals and the audit/calculation archive.

See [QM_WORKFLOW.md](QM_WORKFLOW.md) for personal account setup, the first
milestone's scope, backup instructions and validation requirements before routine
official use. Local accounts must be provisioned before signing into this workspace;
no default passwords are provided.

## Documentation workspace

The sidebar workspace **Dokumentation und Formeln** documents every quantity the app
reports, with its formula: the three volume methods and the constraints on them,
Eq. 4-1 to 4-3 with the reason the influent rather than the effluent is rescaled,
unit conversion, Eq. 5-1/5-2 and the paired guideline LRV, the model
likelihood and priors, the closed-form and numerical population quantiles, the
diagnostic thresholds and the status logic, plus the legacy methods and a symbol
table. Each section names the function that implements it, and the volume section
includes a live calculator that calls `sample_data.effective_volume` directly, so
the worked example cannot drift from the code.

## Counts and individual sample volumes

The default workspace now accepts assay-level effluent counts and sample-specific
volumes. Select an indicator, enter rows or import the provided Excel template, and
inspect the effective-volume preview before fitting. Data for each indicator are
kept separately in the session. Download CSV/JSON to retain them outside the session.
The old calculator is available from the sidebar as **Bisheriger Methodenvergleich**.

`assets/ValidReuse_Datenvorlage.xlsx` contains an **Anleitung** sheet with the full
workflow, a column glossary and all three volume methods. Its documented **Zulauf**
and **Ablauf** sheets can be imported together. Keep their technical column names
unchanged so that the app can validate the workbook safely.

For effluent, enter raw integer CFU/PFU counts (including zero), a sample ID and a
replicate ID. For influent, enter positive concentrations in `/100 mL`, `/mL` or
`/L`; decimal values are accepted. Result type `missing` is retained in export and
excluded from fitting. Censored, MPN and normalised effluent results can be recorded
but block the count fit until a suitable raw-count dataset is supplied.

Volume modes, all in mL:

- `concentrated`: filtered × assayed / dilution factor / total eluate × recovery.
- `direct`: assayed / dilution factor × recovery.
- `effective`: already corrected original-water volume, used once. Other volume
  fields are ignored in this mode; document the source in `recovery_source`.

The effective volume in units of 100 mL is the **reference factor** `V_eff/100`.
Following the guide (Kap. 4.6.1, Eq. 4-1 to 4-3) it is applied to the *influent*:
`c_in_reference = c_in_100ml * V_eff/100`, while the effluent stays the raw count —
under Eq. 5-2 a count of zero becomes one counted unit in `V_eff`, not one per
100 mL. Steps 2 and 3 display the factor and the raised influent, and
`guideline_lrv_pairs` exports both as `reference_factor` and
`inflow_concentration_reference`. The per-100-mL effluent concentration is reported
for readability and does not enter the LRV.

Recovery is a fraction in (0,1], not percent. Dilution factor 10 denotes 1:10.
Because `V_eff` is linear in recovery, the reported LRV moves by `log10(R/R0)`: a
tenfold error in the assumed recovery is exactly one log. Step 4 therefore plots the
lower bound over recoveries of 1–100 % and reports the critical recovery at which the
target is no longer met. The curve is an analytic shift of the fitted bounds, not a
refit — exact for the guideline LRV, and within 0.05 log of real refits for the model
candidates (about 0.08 log when every effluent count is zero). It needs one recovery
shared by all effluent analyses.
Technical replicates must use nonoverlapping aliquots and the same sample ID; each
has its own exposure but shares a latent concentration. Environmental samples must
have distinct IDs. A sample location and time should be recorded for traceability.

All PyMC candidates model `count ~ Poisson(effective_mL/100 * concentration)`.
Poisson-lognormal uses lognormal sample concentrations; the negative-binomial
comparator uses gamma sample concentrations. The two **unpaired** candidates are the
default: they treat influent and effluent as independent populations, so every
influent value — including historical samples with no matching effluent — enters the
fit, and the full influent spread propagates into the LRV. The two **conditional**
(covariance) candidates instead model the latent effluent given the paired influent,
`ln C_out = a + b (ln c_in - m) + eps`, and estimate the coupling `b`. Only the
fraction `(1 - b)` of the influent spread reaches the LRV, so `b = 0` reproduces the
unpaired result and `b = 1` removes the influent spread entirely; `b` can also be
pinned to either value. The conditional candidates need exactly one influent
concentration per environmental effluent sample, matched by pair ID
(`model_arrays(..., paired=True)`), and the reported `b` carries a 90 % interval.
Documentation section 8 gives the equations, the simulation pseudocode and a flow
chart of the fit. The output is the population LRV quantile with a one-sided
posterior credible lower bound, distinct from the legacy count-predictive LRV. Exact
zero counts need no pseudocount in the likelihood.
The preview separately shows the guideline's count-scale zero convention.
When matching pair IDs are supplied, the app also reports the paired,
measurement-limited LRV from Eq. 5-2. Technical effluent replicates are pooled
by adding their counts and disjoint effective volumes. Its displayed q10 is
descriptive and has no confidence interval; it is kept separate from the
model-based latent population q10. The paired table can be downloaded as CSV
and is included in the result JSON when a model run is exported.

Unrounded convergence diagnostics, lower-bound Monte Carlo error, marginal
posterior predictive checks at the observed volumes, and candidate spread are
reported. JSON export includes inputs, exposures, settings, posterior quantiles
and package versions. Input or setting changes hide stale results. Fixed priors
are documented in the interface and `count_models.py`. Results are experimental:
paired dependence, uncertain recovery, censored likelihoods and full calibration
are not yet implemented. See `ROBUSTNESS_REVIEW_NOTES.md` for the updated review.

Run focused tests in the Python 3.13 app environment:

```sh
python -m unittest discover -s tests -v
```
