"""Effluent denominator handling for the log10 reduction (LRV).

FlexTreat Validierungsleitfaden (Zenodo 10.5281/zenodo.13756832), Kapitel 5.1.1:

    Eq. (5-1)  LRV = log10(c(Zulauf) / c(Ablauf))
    Eq. (5-2)  LRV = log10(c(Zulauf))                  fuer c(Ablauf) < 1

Microbiological indicators are discrete counts, so c(Ablauf) < 1 means an exact
zero rather than a censored value. Eq. (5-2) is therefore a floor at 1 applied
*only* to zeros; non-zero counts stay under Eq. (5-1). The guide explicitly
rejects substituting half the quantification limit, since that would raise the
LRV by 0.3 without justification from the data.
"""

from __future__ import annotations

import numpy as np


FLOOR_AT_ONE = "floor_at_one"
ADD_ONE = "add_one"
NONE = "none"

ZERO_HANDLING_MODES = (FLOOR_AT_ONE, ADD_ONE, NONE)


def resolve_zero_handling(zero_handling: str | None, add_one: bool) -> str:
    """Map the legacy ``add_one`` flag onto a zero-handling mode."""
    if zero_handling is None:
        return ADD_ONE if add_one else NONE
    if zero_handling not in ZERO_HANDLING_MODES:
        raise ValueError(
            f"Unbekannter Modus '{zero_handling}'. Erlaubt: {', '.join(ZERO_HANDLING_MODES)}."
        )
    return zero_handling


def effluent_denominator(values: np.ndarray, zero_handling: str) -> np.ndarray:
    """Return the denominator of Eq. (5-1) for simulated effluent counts."""
    if zero_handling == FLOOR_AT_ONE:
        return np.where(values < 1.0, 1.0, values)
    if zero_handling == ADD_ONE:
        return values + 1.0
    if zero_handling == NONE:
        return values
    raise ValueError(
        f"Unbekannter Modus '{zero_handling}'. Erlaubt: {', '.join(ZERO_HANDLING_MODES)}."
    )
