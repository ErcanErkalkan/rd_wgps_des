from __future__ import annotations

from typing import Dict, List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- KM helpers ----------------------------

def _km_survival(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kaplan–Meier survival curve S(t) for right-censored data.

    times: observed time tilde_W = min(t_start, t_end) - t_arr
    events: 1 if started (event observed), 0 if censored
    returns:
      t_unique (ascending),
      S(t) right-after each unique time
    """
    if times.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order = np.argsort(times)
    t = times[order]
    e = events[order]

    t_unique = np.unique(t)
    at_risk = len(t)
    S = 1.0
    S_vals: List[float] = []

    for tu in t_unique:
        mask = (t == tu)
        d = int(np.sum(e[mask] == 1))  # events
        c = int(np.sum(e[mask] == 0))  # censors

        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)

        S_vals.append(S)
        at_risk -= (d + c)

    return t_unique.astype(float), np.asarray(S_vals, dtype=float)


def _ordered_policies(
    keys: Sequence[str],
    policy_order: Optional[Sequence[str]] = None,
) -> List[str]:
    if not policy_order:
        return list(keys)
    order = [p for p in policy_order if p in keys]
    rest = [p for p in keys if p not in order]
    return order + rest


# ---------------------------- Figures ----------------------------

def plot_ccdf(
    samples_by_policy: Dict[str, Sequence[float]],
    outpath: str,
    title: str = "",
    *,
    # If provided, draws censoring-aware KM CCDF (recommended to match paper when censoring exists)
    events_by_policy: Optional[Dict[str, Sequence[int]]] = None,
    policy_order: Optional[Sequence[str]] = ("FIFO", "CVSS-only", "R (risk-only)", "RD+WGPS"),
) -> None:
    """
    Plot CCDF curves for multiple policies.

    Paper alignment:
      - If events_by_policy is provided: KM survival S(t)=P(W>t) (right-censoring aware).
      - Else: empirical CCDF on finite samples (assumes all are events / started jobs).
    """
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(samples_by_policy.keys(), policy_order):
        vals = samples_by_policy[pol]
        x = np.asarray(list(vals), dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        if events_by_policy is not None and pol in events_by_policy:
            e = np.asarray(list(events_by_policy[pol]), dtype=int)
            if e.size != x.size:
                raise ValueError(f"events_by_policy['{pol}'] length != samples length")

            # KM survival (CCDF)
            t_u, S_u = _km_survival(x, e)

            if t_u.size == 0:
                continue

            # Start at (0,1) like a CCDF; step-post to match paper-style
            xx = np.r_[0.0, t_u]
            yy = np.r_[1.0, S_u]
            plt.step(xx, yy, where="post", label=str(pol))
        else:
            # Empirical CCDF (no censoring)
            x = np.sort(x)
            y = 1.0 - (np.arange(1, x.size + 1) / x.size)
            xx = np.r_[0.0, x]
            yy = np.r_[1.0, y]
            plt.step(xx, yy, where="post", label=str(pol))

    plt.xlabel("Waiting time $W$ (hours)")
    plt.ylabel(r"CCDF: $\Pr(W > x)$")
    if title:
        plt.title(title)

    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_backlog_timeseries(
    snaps_by_policy: Dict[str, List[dict]],
    outpath: str,
    title: str = "",
    *,
    key: str = "backlog_age_p95_High",
    time_key_candidates: Sequence[str] = ("time_h", "t_h", "sim_time_h"),
    policy_order: Optional[Sequence[str]] = ("FIFO", "CVSS-only", "R (risk-only)", "RD+WGPS"),
) -> None:
    """
    Plot backlog-age time series (e.g., High bucket P95) for multiple policies.

    Paper alignment:
      - x-axis: time in days
      - y-axis: backlog age P95 (days)
      - lines may be discontinuous when backlog is empty (NaN breaks the line)
    """
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(snaps_by_policy.keys(), policy_order):
        snaps = snaps_by_policy[pol]
        if not snaps:
            continue

        # pick time key
        tkey = None
        for k in time_key_candidates:
            if k in snaps[0]:
                tkey = k
                break
        if tkey is None:
            raise KeyError(f"No time key found in snapshots. Tried: {time_key_candidates}")

        t_days = np.asarray([float(s[tkey]) for s in snaps], dtype=float) / 24.0

        # NaN -> break line (discontinuous), matching paper caption note
        y = np.asarray([float(s.get(key, np.nan)) for s in snaps], dtype=float)

        plt.plot(t_days, y, label=str(pol))

    plt.xlabel("Time (days)")
    plt.ylabel("Backlog age P95 (days)")
    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_utilization_per_window(
    util_by_policy: Dict[str, Sequence[float]],
    outpath: str,
    title: str = "",
    *,
    policy_order: Optional[Sequence[str]] = ("FIFO", "CVSS-only", "R (risk-only)", "RD+WGPS"),
) -> None:
    """
    Plot per-window utilization across policies.

    Paper alignment:
      - x-axis: window index (1..N)
      - y-axis: utilization in [0,1]
    """
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(util_by_policy.keys(), policy_order):
        utils = np.asarray(list(util_by_policy[pol]), dtype=float)
        if utils.size == 0:
            continue
        x = np.arange(1, utils.size + 1, dtype=int)
        plt.plot(x, utils, label=str(pol))

    plt.xlabel("Window index")
    plt.ylabel("Utilization")
    plt.ylim(0.0, 1.05)
    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
