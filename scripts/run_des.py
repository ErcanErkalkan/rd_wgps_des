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
    ("CVSS_only", "CVSS-only"),
    ("R_risk_only", "R-risk-only"),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    run_name = str(cfg["run_name"])
    seed = int(cfg.get("seed", 42))
    horizon_days = int(cfg.get("horizon_days", 56))
    horizon_end_h = float(horizon_days) * 24.0

    outdir = ROOT / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # RD params (Eq. 1)
    w = cfg["policy"]["weights"]
    params = RDParams(
        wC=float(w["wC"]),
        wX=float(w["wX"]),
        wA=float(w["wA"]),
        wK=float(w["wK"]),
        wT=float(w["wT"]),
        Amax_days=float(cfg["policy"].get("Amax_days", 30.0)),
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

    all_metrics = []

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
        kev_rate_burst = float(
            ac.get("kev_rate_burst", burst.get("kev_rate_burst", ac.get("kev_rate_normal", 0.0)))
        )
        arr_cfg = ArrivalConfig(
            weekday_per_day=float(ac["weekday_per_day"]),
            weekend_per_day=float(ac["weekend_per_day"]),
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
        scen_seed = seed + 1000 * scen_idx
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

            pol_seed = seed + 1000 * scen_idx + 10 * pol_idx

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
            all_metrics.append(dfm)

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
                str(scen_dir / "fig_s2_backlog_timeseries.pdf"),
                title="S2: Time series of high-bucket backlog age (P95)",
                key="backlog_age_p95_High",
            )
        if scen_name == "S3_lowcap_hetero":
            plot_utilization_per_window(
                util_by_policy,
                str(scen_dir / "fig_s3_util_per_window.pdf"),
                title="S3: Per-window utilization",
            )

    # write combined metrics
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(outdir / "metrics.csv", index=False)

    print(f"[OK] Wrote: {outdir / 'metrics.csv'}")
    print(f"[OK] Figures under: {outdir}/<scenario>/")
    print(f"[Tip] To render LaTeX table rows: python scripts/render_latex_tables.py --results_dir results/{run_name}")


if __name__ == "__main__":
    main()
