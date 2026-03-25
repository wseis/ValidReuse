from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

MODE = sys.argv[1] if len(sys.argv) > 1 else "stable"
SERVER_MODE = "--server" in sys.argv[2:]
BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / ".pytensor_cache"
CACHE_ROOT.mkdir(exist_ok=True)

if MODE == "fast":
    compiledir = CACHE_ROOT / "fast"
    compiledir.mkdir(exist_ok=True)
    os.environ.setdefault(
        "PYTENSOR_FLAGS",
        f"base_compiledir={compiledir},compiledir={compiledir / 'compiledir'}",
    )
else:
    compiledir = CACHE_ROOT / "stable"
    compiledir.mkdir(exist_ok=True)
    os.environ.setdefault(
        "PYTENSOR_FLAGS",
        f"base_compiledir={compiledir},compiledir={compiledir / 'compiledir'},mode=FAST_COMPILE,linker=py,cxx=",
    )


def main() -> int:
    try:
        import pymc as pm
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"PyMC konnte nicht importiert werden: {exc}"}))
        return 1

    model_cache: dict[tuple[int, int], object] = {}

    def get_model(n_zu: int, n_ab: int):
        cache_key = (n_zu, n_ab)
        cached_model = model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model

        with pm.Model() as model:
            log_zulauf_data = pm.Data("log_zulauf_data", np.zeros(n_zu, dtype=float))
            ablauf_data = pm.Data("ablauf_data", np.zeros(n_ab, dtype=int))

            meanlog_in = pm.Normal("meanlog_in", mu=0.0, sigma=5.0)
            sdlog_in = pm.HalfNormal("sdlog_in", sigma=2.0)
            pm.Normal("log_zulauf_obs", mu=meanlog_in, sigma=sdlog_in, observed=log_zulauf_data)

            log_mu_out = pm.Normal("log_mu_out", mu=0.0, sigma=5.0)
            mu_out = pm.Deterministic("mu_out", pm.math.exp(log_mu_out))
            alpha_out = pm.HalfNormal("alpha_out", sigma=10.0)
            pm.NegativeBinomial("ablauf_obs", mu=mu_out, alpha=alpha_out, observed=ablauf_data)

        model_cache[cache_key] = model
        return model

    def run_single(task_payload: dict[str, object]) -> dict[str, object]:
        vals_zu_arr = np.asarray(task_payload["vals_zu"], dtype=float)
        vals_ab_arr = np.asarray(task_payload["vals_ab"], dtype=int)

        if np.any(vals_zu_arr <= 0):
            raise ValueError("PyMC benoetigt Zulaufwerte groesser als 0.")

        log_zu = np.log(vals_zu_arr)
        model = get_model(len(log_zu), len(vals_ab_arr))
        initvals = {
            "meanlog_in": float(np.mean(log_zu)),
            "sdlog_in_log__": np.log(max(float(np.std(log_zu, ddof=1)) if len(log_zu) > 1 else 0.1, 0.1)),
            "log_mu_out": float(np.log(float(np.mean(vals_ab_arr) + 0.5))),
            "alpha_out_log__": np.log(max(float(np.std(vals_ab_arr, ddof=1)) if len(vals_ab_arr) > 1 else 1.0, 0.5)),
        }

        with model:
            pm.set_data({"log_zulauf_data": log_zu, "ablauf_data": vals_ab_arr})
            trace = pm.sample(
                draws=int(task_payload["draws"]),
                tune=int(task_payload["warmup"]),
                chains=int(task_payload["chains"]),
                cores=1,
                random_seed=task_payload["seed"],
                initvals=initvals,
                init="jitter+adapt_diag" if MODE == "fast" else "adapt_diag",
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=True,
                target_accept=0.9,
            )

        posterior = trace.posterior
        meanlog_draws = posterior["meanlog_in"].values.reshape(-1)
        sdlog_draws = posterior["sdlog_in"].values.reshape(-1)
        mu_draws = posterior["mu_out"].values.reshape(-1)
        alpha_draws = posterior["alpha_out"].values.reshape(-1)

        rng = np.random.default_rng(task_payload["seed"])
        posterior_draw_count = min(len(meanlog_draws), len(sdlog_draws), len(mu_draws), len(alpha_draws))
        q10_samples = np.empty(posterior_draw_count)

        for i in range(posterior_draw_count):
            in_sim = rng.lognormal(mean=meanlog_draws[i], sigma=sdlog_draws[i], size=int(task_payload["n_sim"]))
            prob = alpha_draws[i] / (alpha_draws[i] + mu_draws[i])
            out_sim = rng.negative_binomial(alpha_draws[i], prob, size=int(task_payload["n_sim"])).astype(float)
            if task_payload["add_one"]:
                out_sim = out_sim + 1.0

            lrv = np.log10(in_sim / out_sim)
            q10_samples[i] = np.percentile(lrv, int(task_payload["q"]))

        return {
            "L_alpha": float(np.percentile(q10_samples, 100 * float(task_payload["alpha"]))),
            "median": float(np.percentile(q10_samples, 50)),
            "upper_(1-alpha)": float(np.percentile(q10_samples, 100 * (1 - float(task_payload["alpha"])))),
            "mean": float(np.mean(q10_samples)),
            "std_dev": float(np.std(q10_samples)),
            "q10_samples": q10_samples.tolist(),
        }

    def process_payload(payload: dict[str, object]) -> dict[str, object]:
        if "tasks" in payload:
            results: dict[str, dict[str, object]] = {}
            errors: dict[str, str] = {}
            for task_id, task_payload in payload["tasks"].items():
                try:
                    results[task_id] = run_single(task_payload)
                except Exception as exc:
                    errors[task_id] = str(exc)
            return {"results": results, "errors": errors}
        return run_single(payload)

    try:
        if SERVER_MODE:
            for line in sys.stdin:
                request = line.strip()
                if not request:
                    continue
                try:
                    payload = json.loads(request)
                    result = process_payload(payload)
                except Exception as exc:
                    result = {"error": str(exc)}
                sys.stdout.write(json.dumps(result) + "\n")
                sys.stdout.flush()
            return 0

        payload = json.loads(sys.stdin.read())
        result = process_payload(payload)
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
