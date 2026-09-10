"""Volume-aware effluent candidates, independent influent/effluent population LRV.

The NB comparator is a Poisson-gamma mixture on concentration. Both candidates
share assay likelihoods and sample grouping; zeros are observed counts.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaincc, roots_hermitenorm
from scipy.stats import norm


def population_quantiles(mu_in, sd_in, mu_out, spread_out, family, q=0.1, nodes=64):
    """Population LRV quantile per parameter draw, without count-simulation noise."""
    mu_in, sd_in, mu_out, spread_out = np.broadcast_arrays(mu_in, sd_in, mu_out, spread_out)
    if family == "poisson_lognormal":
        return (mu_in - mu_out + norm.ppf(q)*np.sqrt(sd_in**2 + spread_out**2))/np.log(10.)
    if family != "negative_binomial":
        raise ValueError("Unbekannter Modellkandidat.")
    # CDF_R(r) = E_in[Pr(C_out >= C_in / 10**r)] for Gamma(shape, mean).
    x, weights = roots_hermitenorm(nodes)
    weights = weights / np.sqrt(2*np.pi)
    center = (mu_in-mu_out)/np.log(10.)
    low, high = center-1000., center+1000.
    for _ in range(56):
        mid = (low+high)/2
        log_argument = (np.log(spread_out)[..., None] + mu_in[..., None]
                        + sd_in[..., None]*x - mid[..., None]*np.log(10.) - mu_out[..., None])
        cdf = (gammaincc(spread_out[..., None], np.exp(np.clip(log_argument, -745, 709))) * weights).sum(axis=-1)
        low = np.where(cdf < q, mid, low)
        high = np.where(cdf >= q, mid, high)
    return (low+high)/2


def validate_payload(payload):
    family = payload.get("family", "poisson_lognormal")
    if family not in ["poisson_lognormal", "negative_binomial"]:
        raise ValueError("Unbekannter Modellkandidat.")
    inflow = np.asarray(payload["vals_zu"], dtype=float)
    counts = np.asarray(payload["vals_ab"], dtype=float)
    exposure = np.asarray(payload["exposures"], dtype=float)
    raw_groups = np.asarray(payload.get("sample_groups", np.arange(len(counts))), dtype=float)
    if any(x.ndim != 1 for x in [inflow, counts, exposure, raw_groups]):
        raise ValueError("Eingaben muessen eindimensional sein.")
    if len(inflow) < 2 or len(counts) < 2 or not(len(counts) == len(exposure) == len(raw_groups)):
        raise ValueError("Unzureichende Daten oder unterschiedliche Laengen von Counts, Volumen und Proben-IDs.")
    if not np.all(np.isfinite(inflow) & (inflow > 0)):
        raise ValueError("Zulaufkonzentrationen muessen endlich und positiv sein.")
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts)) & (counts <= 2**53-1)):
        raise ValueError("Ablauf muss exakte nichtnegative Rohzaehlungen enthalten.")
    if not np.all(np.isfinite(exposure) & (exposure > 0)):
        raise ValueError("Effektive Volumina muessen endlich und positiv sein.")
    if not np.all(np.isfinite(raw_groups) & (raw_groups >= 0) & (raw_groups == np.floor(raw_groups))):
        raise ValueError("Ungueltige Probengruppen.")
    _, groups = np.unique(raw_groups, return_inverse=True)
    if len(np.unique(groups)) < 2:
        raise ValueError("Mindestens zwei unabhaengige Ablaufproben erforderlich.")
    if not 0 < float(payload.get("q", 10)) < 100 or not 0 < float(payload.get("alpha", .05)) < .5:
        raise ValueError("Ungueltiges Perzentil oder Alpha.")
    for setting in ["draws", "warmup", "chains"]:
        number = payload.get(setting, {"draws":1000, "warmup":1000, "chains":2}[setting])
        if int(number) != number or number < 1:
            raise ValueError(f"{setting} muss eine positive ganze Zahl sein.")
    return family, inflow, counts.astype(np.int64), exposure, groups


def build_model(payload):
    import pymc as pm
    family, inflow, counts, exposure, groups = validate_payload(payload)
    with pm.Model() as model:
        # Fixed priors on natural-log concentration, per 100 mL; no data reuse.
        mu_in = pm.Normal("mu_in", mu=0., sigma=10.)
        sd_in = pm.HalfNormal("sd_in", sigma=2.)
        pm.Normal("inflow", mu=mu_in, sigma=sd_in, observed=np.log(inflow))
        mu_out = pm.Normal("mu_out", mu=0., sigma=5.)
        if family == "poisson_lognormal":
            spread = pm.HalfNormal("spread_out", sigma=2.)
            z = pm.Normal("z_out", 0., 1., shape=len(np.unique(groups)))
            concentration = pm.Deterministic("concentration", pm.math.exp(mu_out + spread*z))
        else:
            log_shape = pm.Normal("log_shape", 0., 1.5)
            spread = pm.Deterministic("spread_out", pm.math.exp(log_shape))
            concentration = pm.Gamma("concentration", alpha=spread,
                                     beta=spread/pm.math.exp(mu_out), shape=len(np.unique(groups)))
        pm.Poisson("effluent", mu=exposure*concentration[groups], observed=counts)
    return model


def fit_count_model(payload):
    import arviz as az
    import pymc as pm
    family, inflow, counts, exposure, groups = validate_payload(payload)
    model = build_model(payload)
    seed = int(payload.get("seed", 42))
    with model:
        trace = pm.sample(draws=int(payload.get("draws",1000)), tune=int(payload.get("warmup",1000)),
                          chains=int(payload.get("chains",2)), cores=1, random_seed=seed,
                          progressbar=False, target_accept=.95, compute_convergence_checks=True,
                          init="adapt_diag", return_inferencedata=True)
    arrays = {name: trace.posterior[name].values for name in ["mu_in","sd_in","mu_out","spread_out"]}
    quantiles = population_quantiles(*arrays.values(), family, float(payload.get("q",10))/100)
    quadrature_error = 0.
    if family == "negative_binomial":
        refined = population_quantiles(*arrays.values(), family, float(payload.get("q",10))/100, nodes=128)
        quadrature_error = float(np.max(np.abs(refined-quantiles)))
        quantiles = refined
    trace.posterior["q_lrv"] = (("chain", "draw"), quantiles)
    # Read unrounded diagnostics, including latent sample concentrations.
    rhat = np.concatenate([v.values.ravel() for v in az.rhat(trace).data_vars.values()])
    ess = np.concatenate([v.values.ravel() for v in az.ess(trace, method="tail").data_vars.values()])
    max_rhat = float(np.max(rhat)) if np.all(np.isfinite(rhat)) else None
    min_ess = float(np.min(ess)) if np.all(np.isfinite(ess)) else None
    divergences = int(trace.sample_stats.diverging.sum())
    alpha = float(payload.get("alpha", .05))
    try:
        mcse = float(az.mcse(trace, var_names=["q_lrv"], method="quantile", prob=alpha).q_lrv)
        if not np.isfinite(mcse):
            mcse = None
    except (ValueError, TypeError):
        mcse = None
    warnings = []
    if max_rhat is None or max_rhat > 1.01 or min_ess is None or min_ess < 400 or divergences:
        warnings.append("Konvergenz/effektive Stichprobengroesse unzureichend; mehr Ziehungen oder Modellpruefung erforderlich.")
    if mcse is None or mcse > .05 or quadrature_error > .02:
        warnings.append("Numerische Unsicherheit der Untergrenze ist zu gross oder nicht bestimmbar.")
    if not np.any(counts):
        warnings.append("Alle Ablaufzaehlungen sind null: Streuung und q10 sind stark priorabhaengig.")
    # Marginal predictive check: simulate NEW environmental samples at the observed
    # design, sharing the latent rate across their technical replicates.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
    flat = {name: value.ravel() for name, value in arrays.items()}
    indices = np.linspace(0, quantiles.size-1, min(500, quantiles.size), dtype=int)
    ng = len(np.unique(groups))
    predictive = []
    for j in indices:
        if family == "poisson_lognormal":
            c = rng.lognormal(flat["mu_out"][j], flat["spread_out"][j], size=ng)
        else:
            c = rng.gamma(flat["spread_out"][j], np.exp(flat["mu_out"][j])/flat["spread_out"][j], size=ng)
        rates = exposure*c[groups]
        if not np.all(np.isfinite(rates)) or np.any(rates > 1e15):
            warnings.append("Extrem grosse praediktive Counts; Prior/Modell pruefen.")
            continue
        predictive.append(rng.poisson(rates))
    ppc = []
    if predictive:
        predicted = np.asarray(predictive)
        stats = {"Nullanteil": lambda v: np.mean(v == 0, axis=-1),
                 "Maximum (Count)": lambda v: np.max(v, axis=-1),
                 "q90 (Count)": lambda v: np.quantile(v, .9, axis=-1),
                 "Varianz/Mittel (Count)": lambda v: np.var(v, axis=-1)/np.maximum(np.mean(v, axis=-1), 1e-12)}
        for name, fn in stats.items():
            values = fn(predicted)
            observed = float(fn(counts))
            low, median, high = np.quantile(values, [.05,.5,.95])
            ppc.append(dict(statistic=name, observed=observed, low=float(low), median=float(median), high=float(high)))
        if any(p["observed"] < p["low"] or p["observed"] > p["high"] for p in ppc):
            warnings.append("Mindestens eine beobachtete Randstatistik liegt ausserhalb des 90%-Praediktivbereichs; Modellpassung pruefen.")
    else:
        warnings.append("Keine verwendbaren praediktiven Ziehungen.")
    import importlib.metadata
    versions = {package: importlib.metadata.version(package) for package in ["numpy", "scipy", "pymc", "arviz"]}
    return dict(versions=versions, family=family, estimand="latent_independent_population_lrv", L_alpha=float(np.quantile(quantiles,alpha)),
                median=float(np.median(quantiles)), upper=float(np.quantile(quantiles,1-alpha)),
                q10_samples=quantiles.ravel().tolist(), warnings=list(dict.fromkeys(warnings)),
                diagnostics=dict(max_rhat=max_rhat, min_tail_ess=min_ess, divergences=divergences,
                                 lower_bound_mcse=mcse, quadrature_error=quadrature_error),
                predictive_checks=ppc, n_in=len(inflow), n_out=ng, n_assays=len(counts))
