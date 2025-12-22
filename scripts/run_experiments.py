#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import copy

import yaml
import pandas as pd

# allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from rd_wgps_des.sim import (
    ArrivalConfig,
    DurationConfig,
    WindowConfig,
    BurstConfig,
    SignalGenConfig,
    generate_windows,
    generate_arrivals,
    assign_buckets,
    schedule_over_windows,
)
from rd_wgps_des.policies import RDParams
from rd_wgps_des.metrics import summarize_run
from rd_wgps_des.plots import (
    plot_ccdf,
    plot_backlog_timeseries,
    plot_utilization_per_window,
)


# Internal policy IDs must match PolicyName literals in policies.py
POLICY_SPECS = [
    ("FIFO", "FIFO"),
    ("SPT", "SPT"),
    ("CVSS-only", "CVSS-only"),
    ("R-risk-only", "R-risk-only"),
    ("RD-only", "RD-only"),
    ("RD+WGPS", "RD+WGPS"),
]


def _stable_policy_seed_offset(pol_id: str) -> int:
    # deterministic offsets (do NOT use hash(); it's randomized across runs)
    table = {
        "FIFO": 10,
        "SPT": 15,
        "CVSS-only": 20,
        "R-risk-only": 30,
        "RD-only": 35,
        "RD+WGPS": 40,
    }
    return table.get(pol_id, 0)


def _scenario_order(cfg: dict) -> list[str]:
    # stable ordering: S1, S2, S3 ... (sorted)
    return sorted(list(cfg["scenarios"].keys()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_name = str(cfg["run_name"])
    seed = int(cfg.get("seed", 42))
    horizon_days = int(cfg.get("horizon_days", 56))

    outdir = ROOT / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- RD params (paper Eq. RD) -----------------------------------
    w = cfg["policy"]["weights"]
    params = RDParams(
        wC=float(w["wC"]),
        wX=float(w["wX"]),
        wA=float(w["wA"]),
        wK=float(w["wK"]),
        wT=float(w["wT"]),
        Amax_days=float(cfg["policy"]["Amax_days"]),
        eps_hours=float(cfg["policy"]["eps_hours"]),
    )
    H = int(cfg["policy"]["fairness_horizon_H"])

    # ---- buckets (RD quantiles at arrival) --------------------------
    bcfg = cfg["buckets"]
    high_q = float(bcfg["high_quantile"])
    med_q = float(bcfg["med_quantile"])

    # ---- base windows ------------------------------------------------
    base_win = cfg["windows"]["base"]
    base_win_cfg = WindowConfig(
        days_of_week=list(map(int, base_win["days_of_week"])),
        start_hour=int(base_win["start_hour"]),
        length_hours=float(base_win["length_hours"]),
        parallelism=int(base_win["parallelism"]),
        max_items=int(base_win["max_items"]),
    )

    # optional signals override (paper generator table)
    sig_cfg = None
    if "signals" in cfg and cfg["signals"]:
        s = cfg["signals"]
        sig_cfg = SignalGenConfig(
            p_ext=float(s.get("p_ext", 0.25)),
            lambda_A=float(s.get("lambda_A", 1.0 / 12.0)),
            A_cap_days=float(s.get("A_cap_days", 180.0)),
        )

    all_metrics: list[pd.DataFrame] = []

    for scen_idx, scen_name in enumerate(_scenario_order(cfg)):
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
                max_items=int(ow["max_items"]),
            )
        windows = generate_windows(horizon_days, wcfg)

        # arrivals
        ac = scen_cfg["arrival"]
        burst = ac.get("burst", {}) or {}
        arr_cfg = ArrivalConfig(
            weekday_per_day=float(ac["weekday_per_day"]),
            weekend_per_day=float(ac["weekend_per_day"]),
            kev_rate_normal=float(ac["kev_rate_normal"]),
            kev_rate_burst=float(ac["kev_rate_burst"]),
            burst=BurstConfig(
                enabled=bool(burst.get("enabled", False)),
                start_day=int(burst.get("start_day", 0)),
                duration_days=int(burst.get("duration_days", 0)),
            ),
        )

        dc = scen_cfg["durations"]
        dur_cfg = DurationConfig(
            dist=str(dc["dist"]),
            it_mean_h=float(dc["it_mean_h"]),
            it_sigma=float(dc["it_sigma"]),
            iot_mean_h=float(dc["iot_mean_h"]),
            iot_sigma=float(dc["iot_sigma"]),
            ot_mean_h=float(dc["ot_mean_h"]),
            ot_sigma=float(dc["ot_sigma"]),
        )

        # deterministic per-scenario seed
        seed_scen = seed + scen_idx * 100
        jobs = generate_arrivals(
            horizon_days,
            arr_cfg,
            dur_cfg,
            seed=seed_scen,
            sig_cfg=sig_cfg,
        )
        assign_buckets(jobs, params, high_q=high_q, med_q=med_q)

        # plot collectors (per scenario)
        ccdf_times_high: dict[str, list[float]] = {}
        ccdf_events_high: dict[str, list[int]] = {}
        backlog_snaps_by_policy: dict[str, list[dict]] = {}
        util_by_policy: dict[str, list[float]] = {}

        for pol_id, pol_label in POLICY_SPECS:
            jobs_pol = copy.deepcopy(jobs)
            windows_pol = copy.deepcopy(windows)

            # WGPS fairness is applied ONLY inside schedule_over_windows for RD+WGPS (paper-consistent)
            jobs_pol, windows_pol, snaps = schedule_over_windows(
                jobs_pol,
                windows_pol,
                pol_id,
                params,
                fairness_horizon_H=H,
                seed=seed_scen + _stable_policy_seed_offset(pol_id),
                horizon_days=horizon_days,
            )

            # metrics row (policy label for paper tables)
            dfm = summarize_run(scen_name, pol_label, jobs_pol, windows_pol, snaps)
            all_metrics.append(dfm)

            # save backlog snapshots per policy (use internal id for folder names)
            pol_dir = outdir / scen_name / pol_id
            pol_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(snaps).to_csv(pol_dir / "backlog_snapshots.csv", index=False)

            # --- CCDF (KM-ready): build (tilde_W, delta) for High bucket ---
            # choose t_end as end of last window or horizon end (whichever is larger)
            t_end_h = float(horizon_days) * 24.0
            if windows_pol:
                t_end_h = max(
                    t_end_h,
                    max(float(w.start_time_h + w.length_h) for w in windows_pol),
                )

            times: list[float] = []
            events: list[int] = []
            for j in jobs_pol.values():
                if j.bucket != "High":
                    continue
                if float(j.arrival_time_h) > t_end_h:
                    continue

                if (
                    j.start_time_h is not None
                    and float(j.start_time_h) <= t_end_h + 1e-12
                ):
                    times.append(float(j.start_time_h - j.arrival_time_h))
                    events.append(1)
                else:
                    # right-censored
                    times.append(float(t_end_h - j.arrival_time_h))
                    events.append(0)

            ccdf_times_high[pol_label] = times
            ccdf_events_high[pol_label] = events

            backlog_snaps_by_policy[pol_label] = snaps
            util_by_policy[pol_label] = [float(w.utilization()) for w in windows_pol]

        # scenario plots (paper-matching filenames)
        scen_dir = outdir / scen_name
        scen_dir.mkdir(parents=True, exist_ok=True)

        plot_ccdf(
            ccdf_times_high,
            str(scen_dir / f"fig_{scen_name}_wait_ccdf.pdf"),
            title=f"{scen_name}: CCDF of waiting time $W$ (High bucket)",
            events_by_policy=ccdf_events_high,  # KM censoring-aware
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

    # write combined metrics
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(outdir / "metrics.csv", index=False)

    print(f"[OK] Wrote: {outdir / 'metrics.csv'}")
    print(f"[OK] Figures under: {outdir}/<scenario>/")


if __name__ == "__main__":
    main()
