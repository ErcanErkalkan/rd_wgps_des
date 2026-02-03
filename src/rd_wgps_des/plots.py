from __future__ import annotations

from typing import Dict, List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- KM helpers ----------------------------

def _km_survival(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        d = int(np.sum(e[mask] == 1))
        c = int(np.sum(e[mask] == 0))

        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)

        S_vals.append(S)
        at_risk -= (d + c)

    return t_unique.astype(float), np.asarray(S_vals, dtype=float)


def _ordered_policies(keys: Sequence[str], policy_order: Optional[Sequence[str]] = None) -> List[str]:
    if not policy_order:
        return list(keys)
    order = [p for p in policy_order if p in keys]
    rest = [p for p in keys if p not in order]
    return order + rest


# ---------------------------- B/W style helpers ----------------------------

def _bw_style_for_policy(policy: str) -> dict:
    """
    Black-white print-friendly styles:
    - color fixed to black
    - distinguish by linestyle + marker
    - markers unfilled (white/none) for print clarity
    """
    style_map = {
        "FIFO":      dict(linestyle="-",  marker="o", linewidth=1.6),
        "SPT":       dict(linestyle="--", marker="s", linewidth=1.6),
        "CVSS-only": dict(linestyle="-.", marker="^", linewidth=1.6),
        "R-risk-only": dict(linestyle=":", marker="v", linewidth=1.8),
        "RD-only":   dict(linestyle=(0, (5, 2)), marker="D", linewidth=1.8),     # long dash
        "RD+WGPS":   dict(linestyle="-", marker="X", linewidth=2.3),             # emphasized
    }

    # fallback style set for any unseen policies
    fallback = dict(linestyle=(0, (3, 1, 1, 1)), marker="P", linewidth=1.6)

    s = style_map.get(policy, fallback).copy()
    s.update(
        color="black",
        markersize=5.5,
        markerfacecolor="none",      # hollow markers
        markeredgecolor="black",
        markeredgewidth=0.9,
    )
    return s


def _compute_markevery(n: int, target_markers: int = 12) -> int:
    """
    Choose markevery so we get ~target_markers markers on the curve.
    """
    if n <= 0:
        return 1
    return max(1, int(np.ceil(n / max(1, target_markers))))


def _setup_bw_axes():
    # Slightly thicker axes for print
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["legend.frameon"] = True


# ---------------------------- Figures ----------------------------

def plot_ccdf(
    samples_by_policy: Dict[str, Sequence[float]],
    outpath: str,
    title: str = "",
    *,
    events_by_policy: Optional[Dict[str, Sequence[int]]] = None,
    policy_order: Optional[Sequence[str]] = ("FIFO", "SPT", "CVSS-only", "R-risk-only", "RD-only", "RD+WGPS"),
) -> None:
    """
    CCDF curves for multiple policies (B/W print friendly).
    - If events_by_policy is provided: KM survival S(t)=P(W>t) (right-censoring aware).
    - Else: empirical CCDF on finite samples (assumes all are events / started jobs).
    """
    _setup_bw_axes()
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(samples_by_policy.keys(), policy_order):
        vals = samples_by_policy[pol]
        x = np.asarray(list(vals), dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        style = _bw_style_for_policy(pol)

        if events_by_policy is not None and pol in events_by_policy:
            e = np.asarray(list(events_by_policy[pol]), dtype=int)
            if e.size != x.size:
                raise ValueError(f"events_by_policy['{pol}'] length != samples length")

            t_u, S_u = _km_survival(x, e)
            if t_u.size == 0:
                continue

            xx = np.r_[0.0, t_u]
            yy = np.r_[1.0, S_u]

            me = _compute_markevery(len(xx), target_markers=10)
            plt.step(xx, yy, where="post", label=str(pol), markevery=me, **style)
        else:
            x = np.sort(x)
            y = 1.0 - (np.arange(1, x.size + 1) / x.size)
            xx = np.r_[0.0, x]
            yy = np.r_[1.0, y]

            me = _compute_markevery(len(xx), target_markers=10)
            plt.step(xx, yy, where="post", label=str(pol), markevery=me, **style)

    plt.xlabel("Waiting time $W$ (hours)")
    plt.ylabel(r"CCDF: $\Pr(W > x)$")
    if title:
        plt.title(title)

    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.25, linestyle=":")
    plt.legend(ncol=2, fontsize=8, handlelength=3.0, borderpad=0.6)
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
    policy_order: Optional[Sequence[str]] = ("FIFO", "SPT", "CVSS-only", "R-risk-only", "RD-only", "RD+WGPS"),
) -> None:
    """
    Backlog-age time series (B/W print friendly).
    - x-axis: time in days
    - y-axis: backlog age P95 (days)
    - NaN breaks the line (empty backlog)
    """
    _setup_bw_axes()
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(snaps_by_policy.keys(), policy_order):
        snaps = snaps_by_policy[pol]
        if not snaps:
            continue

        tkey = None
        for k in time_key_candidates:
            if k in snaps[0]:
                tkey = k
                break
        if tkey is None:
            raise KeyError(f"No time key found in snapshots. Tried: {time_key_candidates}")

        t_days = np.asarray([float(s[tkey]) for s in snaps], dtype=float) / 24.0
        y = np.asarray([float(s.get(key, np.nan)) for s in snaps], dtype=float)

        style = _bw_style_for_policy(pol)
        me = _compute_markevery(len(t_days), target_markers=12)
        plt.plot(t_days, y, label=str(pol), markevery=me, **style)

    plt.xlabel("Time (days)")
    plt.ylabel("Backlog age P95 (days)")
    if title:
        plt.title(title)

    plt.grid(True, alpha=0.25, linestyle=":")
    plt.legend(ncol=2, fontsize=8, handlelength=3.0, borderpad=0.6)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_utilization_per_window(
    util_by_policy: Dict[str, Sequence[float]],
    outpath: str,
    title: str = "",
    *,
    policy_order: Optional[Sequence[str]] = ("FIFO", "SPT", "CVSS-only", "R-risk-only", "RD-only", "RD+WGPS"),
) -> None:
    """
    Per-window utilization (B/W print friendly).
    - x-axis: window index (1..N)
    - y-axis: utilization in [0,1]
    """
    _setup_bw_axes()
    plt.figure(figsize=(6.2, 3.9))

    for pol in _ordered_policies(util_by_policy.keys(), policy_order):
        utils = np.asarray(list(util_by_policy[pol]), dtype=float)
        if utils.size == 0:
            continue

        x = np.arange(1, utils.size + 1, dtype=int)
        style = _bw_style_for_policy(pol)
        me = _compute_markevery(len(x), target_markers=14)
        plt.plot(x, utils, label=str(pol), markevery=me, **style)

    plt.xlabel("Window index")
    plt.ylabel("Utilization")
    plt.ylim(0.0, 1.05)
    if title:
        plt.title(title)

    plt.grid(True, alpha=0.25, linestyle=":")
    plt.legend(ncol=2, fontsize=8, handlelength=3.0, borderpad=0.6)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
