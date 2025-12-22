#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import copy
from typing import Dict, List, Tuple

# Allow running without installing the package:
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd

import numpy as np
from rd_wgps_des.sim import (
    ArrivalConfig, DurationConfig, WindowConfig, BurstConfig, SignalGenConfig,
    generate_windows, generate_arrivals, assign_buckets, schedule_over_windows,
)
from rd_wgps_des.policies import RDParams
from rd_wgps_des.metrics import summarize_run
from rd_wgps_des.plots import plot_ccdf, plot_backlog_timeseries, plot_utilization_per_window


# Internal policy IDs (used for config + folder names) mapped to paper-friendly labels (PolicyName).
POLICY_SPECS: List[Tuple[str, str]] = [
    ("FIFO", "FIFO"),
    ("SPT_only", "SPT"),
    ("CVSS_only", "CVSS-only"),
    ("R_risk_only", "R-risk-only"),
    ("RD_only", "RD-only"),
    ("RD_WGPS", "RD+WGPS"),
]



def _scenario_names_in_stable_order(cfg: dict) -> List[str]:
    """Prefer S1/S2/S3 order if present; otherwise preserve YAML dict order."""
    scen = cfg.get("scenarios", {}) or {}
    names = list(scen.keys())
    # Common paper names
    preferred = ["S1_normal", "S2_kev_burst", "S3_lowcap_hetero"]
    out = [n for n in preferred if n in scen]
    out += [n for n in names if n not in out]
    return out


def _policies_in_order(cfg: dict) -> List[Tuple[str, str]]:
    requested = cfg.get("policies_compared")
    if requested is None:
        return POLICY_SPECS
    requested_set = set(requested)
    out = [(pid, label) for pid, label in POLICY_SPECS if pid in requested_set]
    if not out:
        raise ValueError("No valid policies found in `policies_compared` (check config).")
    return out


def _km_wait_times_for_bucket(
    jobs,
    bucket: str,
    horizon_end_h: float,
) -> Tuple[List[float], List[int]]:
    """Return (times, events) for Kaplan–Meier plotting of waiting time."""
    times: List[float] = []
    events: List[int] = []
    for j in jobs.values():
        if getattr(j, "bucket", None) != bucket:
            continue
        if j.start_time_h is not None:
            times.append(float(j.start_time_h - j.arrival_time_h))
            events.append(1)
        else:
            # right-censored at horizon end
            times.append(float(max(horizon_end_h - j.arrival_time_h, 0.0)))
            events.append(0)
    return times, events


def _parse_seeds_arg(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seeds = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                raise ValueError(f"Bad seed range '{p}' (end < start).")
            seeds.extend(list(range(a_i, b_i + 1)))
        else:
            seeds.append(int(p))
    # de-dup while preserving order
    out = []
    seen = set()
    for x in seeds:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _resolve_seeds(cfg: dict, args) -> list[int]:
    # CLI > config 'seeds' > config 'seed'
    if args.seeds:
        seeds = _parse_seeds_arg(args.seeds)
        if not seeds:
            raise ValueError("--seeds provided but parsed empty.")
        return seeds

    if args.n_seeds is not None:
        start = int(args.seed_start)
        n = int(args.n_seeds)
        if n <= 0:
            raise ValueError("--n_seeds must be > 0.")
        return list(range(start, start + n))

    if isinstance(cfg.get("seeds", None), (list, tuple)) and len(cfg["seeds"]) > 0:
        return [int(x) for x in cfg["seeds"]]

    return [int(cfg.get("seed", 42))]


def _seed_interval_str(x: float, lo: float, hi: float, decimals: int = 1) -> str:
    """Format mean with a seed-percentile interval for LaTeX tables."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    if lo is None or hi is None or any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in [lo, hi]):
        return f"{x:.{decimals}f}"
    return f"{x:.{decimals}f} [{lo:.{decimals}f},{hi:.{decimals}f}]"


def _agg_seed_interval(series: pd.Series, qlo: float = 0.025, qhi: float = 0.975):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(vals))
    lo = float(np.quantile(vals, qlo)) if len(vals) >= 2 else np.nan
    hi = float(np.quantile(vals, qhi)) if len(vals) >= 2 else np.nan
    return (mean, lo, hi, int(len(vals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/run_default.yaml)")

    # Multi-seed robustness (addresses MR-1)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Seeds as CSV or ranges, e.g., '42,43,44' or '1-30'. Overrides config.")
    ap.add_argument("--n_seeds", type=int, default=None,
                    help="Number of seeds to run (uses --seed_start). Overrides config.")
    ap.add_argument("--seed_start", type=int, default=1, help="Start seed for --n_seeds. Default=1.")
    ap.add_argument("--plots_seed", type=int, default=None,
                    help="Generate plots/artifacts only for this seed. Default: first seed in the list.")
    ap.add_argument("--no_plots", action="store_true",
                    help="Disable plot/artifact generation (metrics only).")

    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    run_name = str(cfg.get("run_name", cfg_path.stem))
    horizon_days = int(cfg.get("horizon_days", 56))
    horizon_end_h = float(horizon_days) * 24.0

    outdir = ROOT / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve seed set
    seeds = _resolve_seeds(cfg, args)
    plots_seed = int(args.plots_seed) if args.plots_seed is not None else int(seeds[0])
    do_plots = (not args.no_plots)

    print(f"[INFO] Run: {run_name} | horizon_days={horizon_days} | seeds={seeds}")
    if do_plots:
        print(f"[INFO] Plot/artifact seed: {plots_seed}")
    else:
        print("[INFO] Plots disabled (--no_plots)")

    # Shared (seed-invariant) settings
    w = cfg["policy"]["weights"]
    params = RDParams(
        wC=float(w.get("w_C", w.get("wC"))),
        wX=float(w.get("w_X", w.get("wX"))),
        wA=float(w.get("w_A", w.get("wA"))),
        wK=float(w.get("w_K", w.get("wK"))),
        wT=float(w.get("w_T", w.get("wT"))),
        Amax_days=float(cfg["policy"].get("A_max_days", 30.0)),
        eps_hours=float(cfg["policy"].get("eps_hours", 0.25)),
    )
    params.validate()

    # Fairness horizon H (Algorithm 1). (Promotion is applied only for RD+WGPS in sim.py.)
    fair_cfg = (cfg.get("policy", {}) or {}).get("fairness", {}) or {}
    H = int(fair_cfg.get("horizon_H", cfg.get("policy", {}).get("fairness_horizon_H", 3)))

    # Buckets for reporting & WGPS queues
    bcfg = cfg["buckets"]
    high_q = float(bcfg.get("high_quantile", 0.80))
    med_q = float(bcfg.get("med_quantile", 0.50))

    # Base windows
    base_win = cfg["windows"]["base"]
    base_win_cfg = WindowConfig(
        days_of_week=list(map(int, base_win["days_of_week"])),
        start_hour=int(base_win["start_hour"]),
        length_hours=float(base_win["length_hours"]),
        parallelism=int(base_win["parallelism"]),
        max_items=int(base_win.get("max_items", base_win.get("q_examinations_per_window"))),
    )
    base_win_cfg.validate()

    # Optional synthetic signal overrides (Table 3)
    sig_cfg = SignalGenConfig()
    if "signals" in cfg:
        sc = cfg["signals"] or {}
        sig_cfg = SignalGenConfig(
            crit_levels=tuple(sc.get("crit_levels", sig_cfg.crit_levels)),
            crit_probs=tuple(sc.get("crit_probs", sig_cfg.crit_probs)),
            p_ext=float(sc.get("p_ext", sig_cfg.p_ext)),
            ext_beta_a=float(sc.get("ext_beta_a", sig_cfg.ext_beta_a)),
            ext_beta_b=float(sc.get("ext_beta_b", sig_cfg.ext_beta_b)),
            int_beta_a=float(sc.get("int_beta_a", sig_cfg.int_beta_a)),
            int_beta_b=float(sc.get("int_beta_b", sig_cfg.int_beta_b)),
            lambda_A=float(sc.get("lambda_A", sig_cfg.lambda_A)),
            A_cap_days=float(sc.get("A_cap_days", sig_cfg.A_cap_days)),
        )
    sig_cfg.validate()

    policy_specs = _policies_in_order(cfg)

    # ===== multi-seed loop =====
    all_metrics_seed: list[pd.DataFrame] = []

    for seed in seeds:
        # Use a per-seed subfolder only when we generate plots/artifacts to avoid huge output trees.
        write_artifacts = do_plots and (seed == plots_seed)

        if write_artifacts:
            print(f"[INFO] Seed {seed}: generating plots/artifacts into {outdir}")
        else:
            print(f"[INFO] Seed {seed}: metrics only")

        per_seed_metrics = []

        for scen_idx, scen_name in enumerate(_scenario_names_in_stable_order(cfg)):
            scen_cfg = cfg["scenarios"][scen_name]

            # windows override?
            wcfg = base_win_cfg
            if scen_cfg.get("windows_override"):
                ow = scen_cfg["windows_override"]
                wcfg = WindowConfig(
                    days_of_week=list(map(int, ow["days_of_week"])),
                    start_hour=int(ow["start_hour"]),
                    length_hours=float(ow["length_hours"]),
                    parallelism=int(ow["parallelism"]),
                    max_items=int(ow.get("max_items", ow.get("q_examinations_per_window"))),
                )
                wcfg.validate()

            windows = generate_windows(horizon_days, wcfg)

            # arrivals
            ac = scen_cfg["arrival"]
            burst = ac.get("burst", {}) or {}
            kev_rate_burst = float(ac.get("kev_rate_burst", burst.get("kev_rate_burst", ac.get("kev_rate_normal"))))

            arr_cfg = ArrivalConfig(
                weekday_per_day=float(ac.get("weekday_rate_per_day", ac.get("weekday_per_day"))),
                weekend_per_day=float(ac.get("weekend_rate_per_day", ac.get("weekend_per_day"))),
                kev_rate_normal=float(ac["kev_rate_normal"]),
                kev_rate_burst=kev_rate_burst,
                burst=BurstConfig(
                    enabled=bool(burst.get("enabled", False)),
                    start_day=int(burst.get("start_day", 0)),
                    duration_days=int(burst.get("duration_days", 0)),
                ),
            )
            arr_cfg.validate()

            dc = scen_cfg["durations"]
            dur_cfg = DurationConfig(
                dist=str(dc.get("dist", "lognormal")),
                it_mean_h=float(dc["it_mean_h"]),
                it_sigma=float(dc["it_sigma"]),
                iot_mean_h=float(dc["iot_mean_h"]),
                iot_sigma=float(dc["iot_sigma"]),
                ot_mean_h=float(dc["ot_mean_h"]),
                ot_sigma=float(dc["ot_sigma"]),
                # mixture defaults OK (Table 3)
            )
            dur_cfg.validate()

            # deterministic per-scenario seed
            scen_seed = int(seed) + 1000 * scen_idx
            jobs = generate_arrivals(horizon_days, arr_cfg, dur_cfg, seed=scen_seed, sig_cfg=sig_cfg)
            assign_buckets(jobs, params, high_q=high_q, med_q=med_q)

            # plot collectors (per scenario)
            ccdf_km_times_high: Dict[str, List[float]] = {}
            ccdf_km_events_high: Dict[str, List[int]] = {}
            backlog_snaps_by_policy: Dict[str, List[dict]] = {}
            util_by_policy: Dict[str, List[float]] = {}

            for pol_idx, (pol_id, pol_label) in enumerate(policy_specs):
                jobs_pol = copy.deepcopy(jobs)
                windows_pol = copy.deepcopy(windows)

                pol_seed = int(seed) + 1000 * scen_idx + 10 * pol_idx

                jobs_pol, windows_pol, snaps = schedule_over_windows(
                    jobs_pol,
                    windows_pol,
                    policy=pol_label,          # PolicyName in rd_wgps_des.policies
                    params=params,
                    fairness_horizon_H=H,
                    seed=pol_seed,
                    horizon_days=horizon_days,
                )

                # metrics (paper tables use KM-based quantiles with censoring)
                dfm = summarize_run(
                    scenario=scen_name,
                    policy=pol_label,
                    jobs=jobs_pol,
                    windows=windows_pol,
                    snapshots=snaps,
                    horizon_end_h=horizon_end_h,
                )
                dfm["seed"] = int(seed)
                per_seed_metrics.append(dfm)

                if write_artifacts:
                    # save snapshots per policy id (stable folder names)
                    pol_dir = outdir / scen_name / pol_id
                    pol_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(snaps).to_csv(pol_dir / "backlog_snapshots.csv", index=False)

                    # collect for plots
                    t_high, e_high = _km_wait_times_for_bucket(jobs_pol, "High", horizon_end_h=horizon_end_h)
                    ccdf_km_times_high[pol_label] = t_high
                    ccdf_km_events_high[pol_label] = e_high

                    backlog_snaps_by_policy[pol_label] = snaps
                    util_by_policy[pol_label] = [float(w.utilization()) for w in windows_pol]

            if write_artifacts:
                # scenario plots
                scen_dir = outdir / scen_name
                scen_dir.mkdir(parents=True, exist_ok=True)

                plot_ccdf(
                    ccdf_km_times_high,
                    str(scen_dir / f"fig_{scen_name}_wait_ccdf.pdf"),
                    title=f"{scen_name}: CCDF of waiting time (High bucket)",
                    events_by_policy=ccdf_km_events_high,
                )
                plot_backlog_timeseries(
                    backlog_snaps_by_policy,
                    str(scen_dir / f"fig_{scen_name}_backlog_timeseries.pdf"),
                    title=f"{scen_name}: High-bucket backlog age (P95)",
                    key="backlog_age_p95_High",
                )
                plot_utilization_per_window(
                    util_by_policy,
                    str(scen_dir / f"fig_{scen_name}_util_per_window.pdf"),
                    title=f"{scen_name}: Utilization per window",
                )

                # Short names commonly used in the LaTeX template
                if scen_name == "S1_normal":
                    plot_ccdf(
                        ccdf_km_times_high,
                        str(scen_dir / "fig_s1_wait_ccdf.pdf"),
                        title="S1: CCDF of waiting time W (High bucket)",
                        events_by_policy=ccdf_km_events_high,
                    )
                if scen_name == "S2_kev_burst":
                    plot_backlog_timeseries(
                        backlog_snaps_by_policy,
                        str(scen_dir / "fig_s2_kev_burst_backlog_timeseries.pdf"),
                        title="S2: High-bucket backlog age (P95)",
                        key="backlog_age_p95_High",
                    )
                if scen_name == "S3_lowcap_hetero":
                    plot_backlog_timeseries(
                        backlog_snaps_by_policy,
                        str(scen_dir / "fig_s3_lowcap_hetero_backlog_timeseries.pdf"),
                        title="S3: High-bucket backlog age (P95)",
                        key="backlog_age_p95_High",
                    )

        seed_df = pd.concat(per_seed_metrics, ignore_index=True)
        all_metrics_seed.append(seed_df)

    metrics_by_seed = pd.concat(all_metrics_seed, ignore_index=True)
    metrics_by_seed.to_csv(outdir / "metrics_by_seed.csv", index=False)

    # If a single seed was run, keep backward-compatible metrics.csv for the paper template
    if len(seeds) == 1:
        metrics_by_seed.drop(columns=["seed"], errors="ignore").to_csv(outdir / "metrics.csv", index=False)

    # ===== aggregate across seeds =====
    # Percentile intervals across seeds (nonparametric, reviewer-friendly).
    grp = metrics_by_seed.groupby(["scenario", "policy"], dropna=False)

    rows = []
    for (scenario, policy), g in grp:
        row = {"scenario": scenario, "policy": policy, "seed_count": int(g["seed"].nunique())}

        # counts / rates
        row["high_started_median"] = float(np.median(g["high_started"]))
        row["high_arrivals_median"] = float(np.median(g["high_arrivals"]))
        row["high_start_rate_mean"], row["high_start_rate_lo"], row["high_start_rate_hi"], row["high_start_rate_n"] = _agg_seed_interval(g["high_started_over_arrivals"])

        # core paper metrics
        for col in ["wait_km_p50_h", "wait_km_p90_h", "wait_max_bound_h", "overline_B95_days", "util_mean", "eow_start_share_mean"]:
            mean, lo, hi, n = _agg_seed_interval(g[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
            row[f"{col}_n"] = n

        rows.append(row)

    agg = pd.DataFrame(rows).sort_values(["scenario", "policy"]).reset_index(drop=True)
    agg.to_csv(outdir / "metrics_agg.csv", index=False)

    # A LaTeX-ready view that can be fed to scripts/render_latex_tables.py without changes:
    # put interval strings into the *same* columns that the table renderer reads.
    agg_paper_rows = []
    for _, r in agg.iterrows():
        rr = {
            "scenario": r["scenario"],
            "policy": r["policy"],
        }
        rr["high_started"] = int(round(r["high_started_median"]))
        rr["high_arrivals"] = int(round(r["high_arrivals_median"]))
        rr["high_started_over_arrivals"] = float(r["high_start_rate_mean"]) if not pd.isna(r["high_start_rate_mean"]) else np.nan
        rr["high_started_over_arrivals_disp"] = _seed_interval_str(
            float(r["high_start_rate_mean"]),
            float(r["high_start_rate_lo"]),
            float(r["high_start_rate_hi"]),
            decimals=2,
        )

        rr["wait_km_p50_h"] = _seed_interval_str(
            float(r["wait_km_p50_h_mean"]),
            float(r["wait_km_p50_h_lo"]),
            float(r["wait_km_p50_h_hi"]),
            decimals=1,
        )
        rr["wait_km_p90_h"] = _seed_interval_str(
            float(r["wait_km_p90_h_mean"]),
            float(r["wait_km_p90_h_lo"]),
            float(r["wait_km_p90_h_hi"]),
            decimals=1,
        )
        rr["wait_max_bound_h"] = _seed_interval_str(
            float(r["wait_max_bound_h_mean"]),
            float(r["wait_max_bound_h_lo"]),
            float(r["wait_max_bound_h_hi"]),
            decimals=1,
        )
        rr["overline_B95_days"] = _seed_interval_str(
            float(r["overline_B95_days_mean"]),
            float(r["overline_B95_days_lo"]),
            float(r["overline_B95_days_hi"]),
            decimals=2,
        )
        # keep util/eow (supplementary)
        rr["util_mean"] = _seed_interval_str(
            float(r["util_mean_mean"]),
            float(r["util_mean_lo"]),
            float(r["util_mean_hi"]),
            decimals=3,
        )
        rr["eow_start_share_mean"] = _seed_interval_str(
            float(r["eow_start_share_mean_mean"]),
            float(r["eow_start_share_mean_lo"]),
            float(r["eow_start_share_mean_hi"]),
            decimals=3,
        )
        agg_paper_rows.append(rr)

    metrics_agg_paper = pd.DataFrame(agg_paper_rows)
    metrics_agg_paper.to_csv(outdir / "metrics_agg_paper.csv", index=False)

    print(f"[OK] Wrote metrics:\n - {outdir/'metrics_by_seed.csv'}\n - {outdir/'metrics_agg.csv'}\n - {outdir/'metrics_agg_paper.csv'}")
    if len(seeds) == 1:
        print(f" - {outdir/'metrics.csv'} (single-seed compatibility)")
    if do_plots:
        print(f"[OK] Plots/artifacts written for seed={plots_seed} into {outdir}.")

if __name__ == "__main__":
    main()
