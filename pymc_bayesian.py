from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


WORKER_PATH = Path(__file__).with_name("pymc_worker.py")
PYMC_AVAILABLE = sys.version_info < (3, 14) and WORKER_PATH.exists()


def pymc_q10_lrv(
    vals_zu: Iterable[int],
    vals_ab: Iterable[int],
    draws: int = 600,
    warmup: int = 300,
    chains: int = 2,
    n_sim: int = 5000,
    q: int = 10,
    alpha: float = 0.05,
    add_one: bool = True,
    seed: int | None = None,
) -> dict[str, float | np.ndarray]:
    if not PYMC_AVAILABLE:
        raise RuntimeError(
            "PyMC ist in dieser Python-Umgebung nicht verfuegbar. "
            "Bitte verwenden Sie Python 3.12 oder 3.13 fuer PyMC."
        )

    payload = {
        "vals_zu": list(vals_zu),
        "vals_ab": list(vals_ab),
        "draws": draws,
        "warmup": warmup,
        "chains": chains,
        "n_sim": n_sim,
        "q": q,
        "alpha": alpha,
        "add_one": add_one,
        "seed": seed,
    }

    errors: list[str] = []
    for mode in ("fast", "stable"):
        completed = subprocess.run(
            [sys.executable, str(WORKER_PATH), mode],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Unbekannter PyMC-Fehler."
            errors.append(f"{mode}: {stderr}")
            continue

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            errors.append(f"{mode}: keine gueltige Antwort")
            continue

        if "error" in result:
            errors.append(f"{mode}: {result['error']}")
            continue

        result["q10_samples"] = np.asarray(result["q10_samples"], dtype=float)
        return result

    raise RuntimeError("PyMC-Subprozess fehlgeschlagen: " + " | ".join(errors))
