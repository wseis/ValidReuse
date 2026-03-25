from __future__ import annotations

import atexit
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable

import numpy as np


WORKER_PATH = Path(__file__).with_name("pymc_worker.py")
PYMC_AVAILABLE = sys.version_info < (3, 14) and WORKER_PATH.exists()


class _PersistentWorker:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        if self.process is not None and self.process.poll() is None:
            return self.process

        self.process = subprocess.Popen(
            [sys.executable, "-u", str(WORKER_PATH), self.mode, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return self.process

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            for attempt in range(2):
                process = self._start()
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("PyMC-Worker konnte nicht gestartet werden.")

                try:
                    process.stdin.write(json.dumps(payload) + "\n")
                    process.stdin.flush()
                    response_line = process.stdout.readline()
                except Exception:
                    response_line = ""

                if response_line:
                    return json.loads(response_line)

                self.close()
                if attempt == 1:
                    raise RuntimeError("PyMC-Worker hat nicht geantwortet.")

            raise RuntimeError("PyMC-Worker konnte nicht verwendet werden.")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=1)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


FAST_WORKER = _PersistentWorker("fast")
STABLE_WORKER = _PersistentWorker("stable")
atexit.register(FAST_WORKER.close)
atexit.register(STABLE_WORKER.close)


def _run_worker(payload: dict[str, object]) -> dict[str, object]:
    errors: list[str] = []
    for mode, worker in (("fast", FAST_WORKER), ("stable", STABLE_WORKER)):
        try:
            result = worker.request(payload)
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
            continue

        if "error" in result:
            errors.append(f"{mode}: {result['error']}")
            continue

        return result

    raise RuntimeError("PyMC-Subprozess fehlgeschlagen: " + " | ".join(errors))


def pymc_q10_lrv_batch(tasks: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    if not PYMC_AVAILABLE:
        raise RuntimeError(
            "PyMC ist in dieser Python-Umgebung nicht verfuegbar. "
            "Bitte verwenden Sie Python 3.12 oder 3.13 fuer PyMC."
        )

    result = _run_worker({"tasks": tasks})
    batch_results = result.get("results", {})
    batch_errors = result.get("errors", {})
    for task_id, task_result in batch_results.items():
        task_result["q10_samples"] = np.asarray(task_result["q10_samples"], dtype=float)
    return {"results": batch_results, "errors": batch_errors}


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

    batch_result = pymc_q10_lrv_batch(
        {
            "single": {
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
        }
    )
    if "single" in batch_result["errors"]:
        raise RuntimeError(batch_result["errors"]["single"])
    return batch_result["results"]["single"]
