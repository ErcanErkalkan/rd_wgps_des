from __future__ import annotations

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

from .models import PatchJob, MaintenanceWindow


# ---------------------------- helpers ---------------------------------


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _infer_horizon_end_h(
    jobs: Dict[int, PatchJob],
    windows: List[MaintenanceWindow],
    snapshots: List[Dict[str, Any]],
    horizon_end_h: Optional[float] = None,
) -> float:
    """Infer simulation end time (hours). Prefer explicit input."""
    if horizon_end_h is not None:
        return float(horizon_end_h)

    # 1) snapshots may carry time stamps (common keys)
    for key in ("t_h", "time_h", "now_h", "sim_time_h"):
        vals = []
        for s in snapshots:
            v = s.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        if vals:
            return float(np.max(vals))

    # 2) windows often have end_time_h / start_time_h / length_hours
    end_vals = []
    for w in windows:
        if hasattr(w, "end_time_h"):
            try:
                end_vals.append(float(getattr(w, "end_time_h")))
                continue
            except Exception:
                pass
        # fall back: start_time_h + length_hours
        if hasattr(w, "start_time_h") and hasattr(w, "length_hours"):
            try:
                end_vals.append(float(getattr(w, "start_time_h")) + float(getattr(w, "length_hours")))
            except Exception:
                pass
    if end_vals:
        return float(np.max(end_vals))

    # 3) last resort (not ideal): use max observed time in jobs
    # (This may under-estimate t_end if many jobs arrive late; pass horizon_end_h if possible.)
    t_candidates = []
    for j in jobs.values():
        if j.start_time_h is not None:
            t_candidates.append(float(j.start_time_h))
        t_candidates.append(float(j.arrival_time_h))
    return float(np.max(t_candidates)) if t_candidates else 0.0


def _km_survival(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier survival curve.

    times: observed times (tilde W), shape (n,)
    events: 1 if event observed (started), 0 if right-censored
    returns:
      t_unique (ascending event/censor times),
      S(t) evaluated right-after each unique time point.
    """
    if len(times) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # Sort by time
    order = np.argsort(times)
    t = times[order]
    e = events[order]

    t_unique = np.unique(t)
    n = len(t)
    at_risk = n
    S = 1.0

    S_vals = []
    for tu in t_unique:
        # d: events at tu, c: censors at tu
        mask = (t == tu)
        d = int(np.sum(e[mask] == 1))
        c = int(np.sum(e[mask] == 0))

        # KM step (apply events first at this time)
        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)

        S_vals.append(S)

        # then remove all (events + censors) at tu from risk set
        at_risk -= (d + c)

    return t_unique.astype(float), np.asarray(S_vals, dtype=float)


def _km_quantile(
    times: List[float],
    events: List[int],
    q: float,
) -> Tuple[float, bool, float]:
    """Return (quantile_value, identifiable, t_last_event).

    identifiable=False if survival never drops below (1-q).
    t_last_event is the last observed event time (max time with event=1), used for '> t_last_event' reporting.
    """
    if not times:
        return float("nan"), False, float("nan")

    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)

    t_event = t[e == 1]
    t_last_event = float(np.max(t_event)) if t_event.size else float("nan")

    t_u, S_u = _km_survival(t, e)
    if t_u.size == 0:
        return float("nan"), False, t_last_event

    target = 1.0 - q  # want S(t) <= target
    idx = np.where(S_u <= target)[0]
    if idx.size == 0:
        return float("nan"), False, t_last_event

    return float(t_u[idx[0]]), True, t_last_event


def _fmt_km(value: float, identifiable: bool, t_last_event: float, decimals: int = 1) -> str:
    """Paper-style display: numeric if identifiable, else >t_last."""
    if identifiable and np.isfinite(value):
        return f"{value:.{decimals}f}"
    if np.isfinite(t_last_event):
        return rf"$> {t_last_event:.{decimals}f}$"
    return "--"


def _max_bound(times: List[float], events: List[int]) -> Tuple[float, str]:
    """Paper-style 'Max bound': max(tilde W), with >= if max comes from censored."""
    if not times:
        return float("nan"), "--"
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    mx = float(np.max(t))
    # if any censored item attains the maximum, it's a lower bound (>=)
    censored_at_max = bool(np.any((t == mx) & (e == 0)))
    disp = (rf"$\ge {mx:.1f}$" if censored_at_max else f"{mx:.1f}")
    return mx, disp


# ---------------------------- main ---------------------------------


def summarize_run(
    scenario: str,
    policy: str,
    jobs: Dict[int, PatchJob],
    windows: List[MaintenanceWindow],
    snapshots: List[Dict[str, Any]],
    *,
    horizon_end_h: Optional[float] = None,
) -> pd.DataFrame:
    """Create a 1-row summary aligned with the paper tables (High bucket focus).

    Paper-aligned High bucket fields:
      - started/arrivals (and ratio)
      - KM P50, KM P90 (hours), censoring-aware (right-censoring)
      - Max bound (hours): max(tilde W), with >= if max from censored
      - overline_B95_days: time-average P95 backlog age (days) from snapshots

    Extra operational fields kept (useful for supplements/plots):
      - util_mean, util_p95
      - eow_start_share_mean, eow_start_share_p95 (delta=30min)
    """
    t_end = _infer_horizon_end_h(jobs, windows, snapshots, horizon_end_h=horizon_end_h)

    # High bucket arrivals
    high_jobs = [j for j in jobs.values() if j.bucket == "High"]
    high_arrivals = len(high_jobs)
    high_started = sum(1 for j in high_jobs if j.start_time_h is not None)

    # Censoring-aware time-to-start:
    # tilde_W = min(t_start, t_end) - t_arr
    # delta = 1[t_start <= t_end]
    tilde_W: List[float] = []
    delta: List[int] = []
    for j in high_jobs:
        arr = float(j.arrival_time_h)
        if j.start_time_h is not None and float(j.start_time_h) <= t_end:
            tilde_W.append(float(j.start_time_h) - arr)
            delta.append(1)
        else:
            tilde_W.append(t_end - arr)
            delta.append(0)

    km_p50, ok50, t_last = _km_quantile(tilde_W, delta, 0.50)
    km_p90, ok90, t_last_90 = _km_quantile(tilde_W, delta, 0.90)
    # (t_last and t_last_90 should be the same; keep separate for clarity)
    max_bound_h, max_bound_disp = _max_bound(tilde_W, delta)

    km_p50_disp = _fmt_km(km_p50, ok50, t_last)
    km_p90_disp = _fmt_km(km_p90, ok90, t_last_90)

    # overline_B95: mean of snapshot backlog_age_p95_High (days), ignoring NaNs
    b95_vals: List[float] = []
    for s in snapshots:
        v = s.get("backlog_age_p95_High")
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isnan(fv):
            continue
        b95_vals.append(fv)
    overline_B95_days = float(np.mean(b95_vals)) if b95_vals else float("nan")

    # Utilization (per-window)
    utils = [float(w.utilization()) for w in windows] if windows else []
    util_mean = float(np.mean(utils)) if utils else float("nan")
    util_p95 = _quantile(utils, 0.95)

    # End-of-window surge indicator (delta=30 min)
    eow = [float(w.end_of_window_start_share(delta_minutes=30.0)) for w in windows] if windows else []
    eow_mean = float(np.mean(eow)) if eow else float("nan")
    eow_p95 = _quantile(eow, 0.95)

    row = {
        "scenario": scenario,
        "policy": policy,

        # Paper-style started/arrivals
        "high_started": high_started,
        "high_arrivals": high_arrivals,
        "high_started_over_arrivals": (high_started / high_arrivals) if high_arrivals else float("nan"),
        "high_started_over_arrivals_disp": f"{high_started}/{high_arrivals}" if high_arrivals else "0/0",

        # KM (censoring-aware) quantiles in hours (paper: KM P50, KM P90)
        "wait_km_p50_h": km_p50,
        "wait_km_p90_h": km_p90,
        "wait_km_p50_disp": km_p50_disp,
        "wait_km_p90_disp": km_p90_disp,

        # Paper 'Max bound' (hours), with >= when driven by censoring
        "wait_max_bound_h": max_bound_h,
        "wait_max_bound_disp": max_bound_disp,

        # Paper overline{B95} (days)
        "overline_B95_days": overline_B95_days,

        # Optional supplement metrics
        "util_mean": util_mean,
        "util_p95": util_p95,
        "eow_start_share_mean": eow_mean,
        "eow_start_share_p95": eow_p95,

        # For debugging/traceability
        "t_end_h": float(t_end),
        "high_censored": int(np.sum(np.asarray(delta) == 0)) if delta else 0,
    }

    return pd.DataFrame([row])
