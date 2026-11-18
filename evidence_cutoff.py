# -*- coding: utf-8 -*-
"""
evidence_cutoff.py  (v1.0.0)

Prediction-level cutoff for the evidence selection cell of the AF3 -> HADDOCK
pipeline ("EVIDENCE DISTRIBUTION + CUTOFF ANALYSIS").

For one evidence (one value per AF3 prediction) the functions return which
predictions lie on the BETTER side of a cutoff.

Methods
-------
kneedle  (default)
    Elbow of the sorted evidence values (Kneedle, Satopaa et al. 2011):
      1. orient so that larger = better (PAE: lower is better -> negated),
         optional transform (e.g. log1p for heavy-tailed counts);
      2. sort best -> worst; x = rank, y = value; normalize both to [0, 1];
      3. distance to the chord d = (1 - y_norm) - x_norm
         d > 0 : curve below the chord (convex: few high values, long tail)
         d < 0 : curve above the chord (concave: many high values, sharp drop)
      4. elbow = rank with the largest |d|; the curve shape is the sign;
      5. keep every prediction whose value is at least as good as the elbow
         value (ties at the elbow value are kept together).
    No elbow (fewer than ``min_points`` values, a flat curve, or
    max |d| < ``min_distance`` i.e. an almost straight line) -> fallback.

gap95  (previous behaviour, also the fallback)
    Sort worst -> best, adjacent gaps, 95th percentile of the gaps, the first
    (worst-side) gap >= that percentile is the cut; everything after it is kept.
    Identical to the original notebook cell.

bootstrap_cutoff() resamples the values to show how stable the elbow is.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

MODULE_VERSION = "1.0.0"

METHODS = ("kneedle", "gap95")
TRANSFORMS = (None, "log1p")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_direction(direction: str) -> None:
    if direction not in ("higher", "lower"):
        raise ValueError(f"direction must be 'higher' or 'lower', got {direction!r}")


def _transform(values: np.ndarray, transform: Optional[str]) -> np.ndarray:
    if transform is None:
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p transform needs non-negative values.")
        return np.log1p(values)
    raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")


# ---------------------------------------------------------------------------
# Kneedle
# ---------------------------------------------------------------------------

def kneedle_elbow(
    scores_best_first: np.ndarray,
    min_points: int = 10,
    min_distance: float = 0.05,
) -> Optional[dict[str, Any]]:
    """
    Elbow of a curve sorted best -> worst (larger = better, non-increasing).

    Returns {knee_rank, shape ('convex'|'concave'), distance, n} or None.
    """
    y = np.asarray(scores_best_first, dtype=float)
    n = len(y)
    if n < min_points:
        return None
    hi, lo = y[0], y[-1]
    if not hi > lo:
        return None

    x_norm = np.arange(n) / (n - 1)
    y_norm = (y - lo) / (hi - lo)
    d = (1.0 - y_norm) - x_norm

    i_convex, i_concave = int(np.argmax(d)), int(np.argmin(d))
    if d[i_convex] >= -d[i_concave]:
        knee, shape, dist = i_convex, "convex", float(d[i_convex])
    else:
        knee, shape, dist = i_concave, "concave", float(-d[i_concave])

    if dist < min_distance:
        return None
    return {"knee_rank": knee, "shape": shape, "distance": dist, "n": n}


def kneedle_cutoff(
    values: np.ndarray,
    direction: str,
    transform: Optional[str] = None,
    min_points: int = 10,
    min_distance: float = 0.05,
) -> Optional[dict[str, Any]]:
    """
    Kneedle cutoff for one evidence. ``values`` must be finite.
    Returns {mask, cutoff_value, knee_rank, shape, distance, n} or None.
    ``mask`` is aligned with ``values``.
    """
    _check_direction(direction)
    v = np.asarray(values, dtype=float)
    t = _transform(v, transform)
    score = t if direction == "higher" else -t

    order = np.argsort(-score, kind="mergesort")       # best -> worst
    elbow = kneedle_elbow(score[order], min_points, min_distance)
    if elbow is None:
        return None

    cut_score = score[order][elbow["knee_rank"]]
    return {
        **elbow,
        "mask": score >= cut_score,
        "cutoff_value": float(v[order][elbow["knee_rank"]]),
    }


# ---------------------------------------------------------------------------
# Original 95th-percentile gap rule (unchanged behaviour)
# ---------------------------------------------------------------------------

def gap95_cutoff(values: np.ndarray, direction: str) -> Optional[dict[str, Any]]:
    """
    Original rule of the notebook cell. Returns {mask, cutoff_value, gap_95,
    primary_gap, primary_gap_position, top_gap_count} or None (< 3 values).
    """
    _check_direction(direction)
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return None

    order = np.argsort(v if direction == "higher" else -v, kind="mergesort")  # worst -> best
    sv = v[order]
    gaps = np.abs(np.diff(sv))
    gap_95 = float(np.percentile(gaps, 95))
    top = np.where(gaps >= gap_95)[0]
    if len(top) == 0:
        return None
    pos = int(top.min())

    mask = np.zeros(len(v), dtype=bool)
    mask[order[pos + 1:]] = True
    return {
        "mask": mask,
        "cutoff_value": float(sv[pos + 1]),
        "gap_95": gap_95,
        "primary_gap": float(gaps[pos]),
        "primary_gap_position": pos,
        "top_gap_count": int(len(top)),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_better_side(
    values: np.ndarray,
    direction: str,
    method: str = "kneedle",
    transform: Optional[str] = None,
    min_points: int = 10,
    min_distance: float = 0.05,
) -> dict[str, Any]:
    """
    Predictions on the better side of the cutoff for one evidence.

    Returns a dict with ``mask`` (aligned with ``values``), ``method_used``
    ('kneedle' | 'gap95' | 'gap95_fallback' | 'none'), ``cutoff_value`` and the
    diagnostics of the method that was used.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    v = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite (drop missing values first).")

    if method == "kneedle":
        res = kneedle_cutoff(v, direction, transform, min_points, min_distance)
        if res is not None:
            return {**res, "method_used": "kneedle", "transform": transform}

    res = gap95_cutoff(v, direction)
    if res is not None:
        used = "gap95" if method == "gap95" else "gap95_fallback"
        return {**res, "method_used": used, "transform": None}

    return {"mask": np.zeros(len(v), dtype=bool), "method_used": "none",
            "cutoff_value": None, "transform": None}


def bootstrap_cutoff(
    values: np.ndarray,
    direction: str,
    transform: Optional[str] = None,
    n_boot: int = 200,
    seed: int = 0,
    min_points: int = 10,
    min_distance: float = 0.05,
) -> dict[str, Any]:
    """
    Stability of the Kneedle cutoff: resample with replacement ``n_boot`` times.
    Returns the 5/50/95 % percentiles of the cutoff value and the fraction of
    resamples without an elbow.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    cuts, none = [], 0
    for _ in range(n_boot):
        r = kneedle_cutoff(rng.choice(v, size=len(v), replace=True),
                           direction, transform, min_points, min_distance)
        if r is None:
            none += 1
        else:
            cuts.append(r["cutoff_value"])
    if not cuts:
        return {"n_boot": n_boot, "no_elbow_fraction": 1.0,
                "cutoff_p05": None, "cutoff_p50": None, "cutoff_p95": None}
    p05, p50, p95 = np.percentile(cuts, [5, 50, 95])
    return {"n_boot": n_boot, "no_elbow_fraction": none / n_boot,
            "cutoff_p05": float(p05), "cutoff_p50": float(p50), "cutoff_p95": float(p95)}


if __name__ == "__main__":
    print(f"evidence_cutoff.py {MODULE_VERSION}")
