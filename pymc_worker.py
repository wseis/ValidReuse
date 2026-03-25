from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

MODE = sys.argv[1] if len(sys.argv) > 1 else "stable"

if MODE == "fast":
    os.environ.setdefault(
        "PYTENSOR_FLAGS",
        "base_compiledir=/tmp/pytensor_fast,compiledir=/tmp/pytensor_fast/compiledir",
    )
else:
    os.environ.setdefault(
        "PYTENSOR_FLAGS",
        "base_compiledir=/tmp/pytensor,compiledir=/tmp/pytensor/compiledir,mode=FAST_COMPILE,linker=py,cxx=",
    )


def main() -> int:
    try:
        import pymc as pm
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"PyMC konnte nicht importiert werden: {exc}"}))
        return 1

    def run_single(task_payload: dict[str, object]) -> dict[str, object]:
        vals_zu_arr = np.asarray(task_payload["vals_zu"], dtype=float)
        vals_ab_arr = np.asarray(task_payload["vals_ab"], dtype=int)

        if np.any(vals_zu_arr <= 0):
            raise ValueError("PyMC benoetigt Zulaufwerte groesser als 0.")

        log_zu = np.log(vals_zu_arr)

        with pm.Model():
            meanlog_in = pm.Normal("meanlog_in", mu=float(np.mean(log_zu)), sigma=5.0)
            sdlog_in = pm.HalfNormal("sdlog_in", sigma=2.0)
            pm.Normal("log_zulauf_obs", mu=meanlog_in, sigma=sdlog_in, observed=log_zu)

            log_mu_out = pm.Normal("log_mu_out", mu=np.log(float(np.mean(vals_ab_arr) + 0.5)), sigma=5.0)
            mu_out = pm.Deterministic("mu_out", pm.math.exp(log_mu_out))
            alpha_out = pm.HalfNormal("alpha_out", sigma=10.0)
            pm.NegativeBinomial("ablauf_obs", mu=mu_out, alpha=alpha_out, observed=vals_ab_arr)

            trace = pm.sample(
                draws=int(task_payload["draws"]),
                tune=int(task_payload["warmup"]),
                chains=int(task_payload["chains"]),
                cores=1,
                random_seed=task_payload["seed"],
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

    try:
        payload = json.loads(sys.stdin.read())
        if "tasks" in payload:
            results: dict[str, dict[str, object]] = {}
            errors: dict[str, str] = {}
            for task_id, task_payload in payload["tasks"].items():
                try:
                    results[task_id] = run_single(task_payload)
                except Exception as exc:
                    errors[task_id] = str(exc)
            result = {"results": results, "errors": errors}
        else:
            result = run_single(payload)
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
