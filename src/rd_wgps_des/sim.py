from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import math

import numpy as np

from .models import PatchJob, MaintenanceWindow
from .policies import (
    RDParams,
    PolicyName,
    baseline_order_key,
    wgps_bucket_order_key,
    rd_score,
    risk_only_R,
)


# ------------------------------ Configs ------------------------------

@dataclass(frozen=True)
class WindowConfig:
    days_of_week: List[int]        # 0=Mon, 1=Tue, ..., 6=Sun
    start_hour: int                # local hour-of-day for window start
    length_hours: float            # window length L(w)
    parallelism: int               # number of parallel jobs
    max_items: int                 # candidate examination cap q(w)

    def validate(self) -> None:
        if not self.days_of_week:
            raise ValueError("days_of_week must be non-empty.")
        if any(d < 0 or d > 6 for d in self.days_of_week):
            raise ValueError("days_of_week must be in [0,6].")
        if self.start_hour < 0 or self.start_hour > 23:
            raise ValueError("start_hour must be in [0,23].")
        if self.length_hours <= 0:
            raise ValueError("length_hours must be > 0.")
        if self.parallelism < 1:
            raise ValueError("parallelism must be >= 1.")
        if self.max_items < 1:
            raise ValueError("max_items (q(w)) must be >= 1.")


@dataclass(frozen=True)
class BurstConfig:
    enabled: bool = False
    start_day: int = 0
    duration_days: int = 0

    def is_active(self, day: int) -> bool:
        if not self.enabled:
            return False
        return self.start_day <= day < (self.start_day + self.duration_days)


@dataclass(frozen=True)
class ArrivalConfig:
    weekday_per_day: float
    weekend_per_day: float
    kev_rate_normal: float
    kev_rate_burst: float
    burst: BurstConfig = BurstConfig()

    def validate(self) -> None:
        if self.weekday_per_day < 0 or self.weekend_per_day < 0:
            raise ValueError("arrival rates must be nonnegative.")
        for r in (self.kev_rate_normal, self.kev_rate_burst):
            if r < 0 or r > 1:
                raise ValueError("KEV rates must be in [0,1].")


@dataclass(frozen=True)
class DurationConfig:
    dist: str = "lognormal"

    # Lognormal parameters expressed as (mean in hours, sigma in log-space)
    it_mean_h: float = 0.6
    it_sigma: float = 0.45
    iot_mean_h: float = 0.8
    iot_sigma: float = 0.55
    ot_mean_h: float = 1.6
    ot_sigma: float = 0.65

    # device-class mixture (probabilities must sum to 1)
    p_it: float = 0.5
    p_iot: float = 0.35
    p_ot: float = 0.15

    # optional cap to avoid extreme outliers
    duration_cap_h: Optional[float] = None

    def validate(self) -> None:
        if self.dist != "lognormal":
            raise ValueError("Only dist='lognormal' is currently supported.")
        for m in (self.it_mean_h, self.iot_mean_h, self.ot_mean_h):
            if m <= 0:
                raise ValueError("Means must be > 0.")
        for s in (self.it_sigma, self.iot_sigma, self.ot_sigma):
            if s <= 0:
                raise ValueError("Sigmas must be > 0.")
        ps = self.p_it + self.p_iot + self.p_ot
        if abs(ps - 1.0) > 1e-6:
            raise ValueError("Device-class mixture probabilities must sum to 1.")


@dataclass(frozen=True)
class SignalGenConfig:
    """Synthetic signal generator parameters (Table 'Default synthetic input generator' in the paper)."""

    # criticality distribution over ordinal levels
    crit_levels: Tuple[float, float, float, float] = (0.25, 0.50, 0.75, 1.00)
    crit_probs:  Tuple[float, float, float, float] = (0.30, 0.35, 0.25, 0.10)

    # exposure mixture
    p_ext: float = 0.25
    ext_beta_a: float = 6.0
    ext_beta_b: float = 2.0
    int_beta_a: float = 2.0
    int_beta_b: float = 5.0

    # vulnerability age (days): truncated exponential with rate lambda_A (per day)
    lambda_A: float = 1.0 / 12.0   # example rate (mean 12 days)
    A_cap_days: float = 180.0      # truncate to 180 days

    def validate(self) -> None:
        if any(p < 0 for p in self.crit_probs):
            raise ValueError("crit_probs must be nonnegative.")
        if abs(sum(self.crit_probs) - 1.0) > 1e-6:
            raise ValueError("crit_probs must sum to 1.")
        if not (0 <= self.p_ext <= 1):
            raise ValueError("p_ext must be in [0,1].")
        if self.ext_beta_a <= 0 or self.ext_beta_b <= 0:
            raise ValueError("ext_beta params must be > 0.")
        if self.int_beta_a <= 0 or self.int_beta_b <= 0:
            raise ValueError("int_beta params must be > 0.")
        if self.lambda_A <= 0:
            raise ValueError("lambda_A (rate) must be > 0.")
        if self.A_cap_days <= 0:
            raise ValueError("A_cap_days must be > 0.")


# ------------------------------ Generators ------------------------------

def generate_windows(horizon_days: int, cfg: WindowConfig) -> List[MaintenanceWindow]:
    """Generate maintenance windows over the horizon.

    Day-of-week indexing uses 0=Mon, 1=Tue, ..., 6=Sun (paper convention).
    Simulation start (day 0) is assumed to be Monday.
    """
    cfg.validate()
    windows: List[MaintenanceWindow] = []
    idx = 0
    dow_set = set(cfg.days_of_week)
    for day in range(horizon_days):
        dow = day % 7
        if dow in dow_set:
            start_time_h = day * 24.0 + float(cfg.start_hour)
            windows.append(
                MaintenanceWindow(
                    idx=idx,
                    start_time_h=start_time_h,
                    length_h=float(cfg.length_hours),
                    parallelism=int(cfg.parallelism),
                    max_examinations=int(cfg.max_items),
                )
            )
            idx += 1
    return windows


def _lognormal_mu_from_mean(mean: float, sigma: float) -> float:
    # If D ~ LogNormal(mu, sigma), then E[D] = exp(mu + sigma^2/2) = mean
    return math.log(float(mean)) - (float(sigma) ** 2) / 2.0


def _sample_lognormal(rng: np.random.Generator, mean: float, sigma: float, cap: Optional[float]) -> float:
    mu = _lognormal_mu_from_mean(mean, sigma)
    v = float(rng.lognormal(mean=mu, sigma=float(sigma)))
    if cap is not None:
        v = min(v, float(cap))
    return max(v, 1e-6)


def generate_arrivals(
    horizon_days: int,
    arr_cfg: ArrivalConfig,
    dur_cfg: DurationConfig,
    seed: int = 42,
    sig_cfg: Optional[SignalGenConfig] = None,
) -> Dict[int, PatchJob]:
    """Generate arrivals with synthetic signals and durations (paper Table for generator)."""
    arr_cfg.validate()
    dur_cfg.validate()
    sig_cfg = sig_cfg or SignalGenConfig()
    sig_cfg.validate()

    rng = np.random.default_rng(int(seed))
    jobs: Dict[int, PatchJob] = {}
    jid = 0

    classes = np.array(["IT", "IoT", "OT"], dtype=object)
    probs = np.array([dur_cfg.p_it, dur_cfg.p_iot, dur_cfg.p_ot], dtype=float)

    for day in range(horizon_days):
        dow = day % 7
        weekend = (dow in (5, 6))  # Sat/Sun
        lam = float(arr_cfg.weekend_per_day if weekend else arr_cfg.weekday_per_day)
        n = int(rng.poisson(lam=lam)) if lam > 0 else 0

        kev_rate = float(arr_cfg.kev_rate_burst) if arr_cfg.burst.is_active(day) else float(arr_cfg.kev_rate_normal)

        for _ in range(n):
            arrival_time_h = day * 24.0 + float(rng.uniform(0.0, 24.0))
            device_class = str(rng.choice(classes, p=probs))

            if device_class == "IT":
                duration_h = _sample_lognormal(rng, dur_cfg.it_mean_h, dur_cfg.it_sigma, dur_cfg.duration_cap_h)
            elif device_class == "IoT":
                duration_h = _sample_lognormal(rng, dur_cfg.iot_mean_h, dur_cfg.iot_sigma, dur_cfg.duration_cap_h)
            else:
                duration_h = _sample_lognormal(rng, dur_cfg.ot_mean_h, dur_cfg.ot_sigma, dur_cfg.duration_cap_h)

            # Criticality (ordinal)
            criticality = float(rng.choice(np.array(sig_cfg.crit_levels), p=np.array(sig_cfg.crit_probs)))

            # Exposure mixture
            if rng.random() < sig_cfg.p_ext:
                exposure = float(rng.beta(sig_cfg.ext_beta_a, sig_cfg.ext_beta_b))
            else:
                exposure = float(rng.beta(sig_cfg.int_beta_a, sig_cfg.int_beta_b))

            # Vulnerability age at arrival: A ~ min( Exp(lambda_A), A_cap )
            # numpy.exponential takes "scale" (mean) = 1/lambda (rate)
            vuln_age = float(min(rng.exponential(1.0 / sig_cfg.lambda_A), sig_cfg.A_cap_days))

            # KEV
            kev = int(rng.random() < kev_rate)

            # Threat proxy + synthetic severity for CVSS-only baseline
            if kev == 1:
                threat = float(rng.beta(5, 2))
                cvss = 10.0 * float(rng.beta(5, 2))
            else:
                threat = float(rng.beta(2, 6))
                cvss = 10.0 * float(rng.beta(2, 3))

            jobs[jid] = PatchJob(
                job_id=jid,
                arrival_time_h=arrival_time_h,
                device_class=device_class,
                duration_h=duration_h,
                criticality=criticality,
                exposure=exposure,
                vuln_age_days=vuln_age,
                kev=kev,
                threat=threat,
                cvss=cvss,
            )
            jid += 1

    return jobs


# ------------------------------ Buckets (paper) ------------------------------

def assign_buckets(
    jobs: Dict[int, PatchJob],
    params: RDParams,
    high_q: float = 0.80,
    med_q: float = 0.50,
) -> None:
    """Assign High/Medium/Low buckets by risk-only R_i quantiles at ARRIVAL time (reporting target)."""
    params.validate()
    if not jobs:
        return
    if not (0 < med_q < high_q < 1.0):
        raise ValueError("Require 0 < med_q < high_q < 1.")

    scores = []
    for job in jobs.values():
        s = risk_only_R(job, now_h=job.arrival_time_h, p=params)
        scores.append(float(s))

    scores_arr = np.asarray(scores, dtype=float)
    hi_thr = float(np.quantile(scores_arr, high_q))
    med_thr = float(np.quantile(scores_arr, med_q))

    for job in jobs.values():
        s = float(rd_score(job, now_h=job.arrival_time_h, p=params))
        # store rd_at_arrival if model supports it
        if hasattr(job, "rd_at_arrival"):
            setattr(job, "rd_at_arrival", s)

        if s >= hi_thr:
            job.bucket = "High"
        elif s >= med_thr:
            job.bucket = "Medium"
        else:
            job.bucket = "Low"


# ------------------------------ Scheduler (paper) ------------------------------

def schedule_over_windows(
    jobs: Dict[int, PatchJob],
    windows: List[MaintenanceWindow],
    policy: PolicyName,
    params: RDParams,
    fairness_horizon_H: int = 3,
    seed: int = 42,  # kept for interface stability; not required here
    horizon_days: Optional[int] = None,  # if provided, snapshots align exactly to paper horizon
) -> Tuple[Dict[int, PatchJob], List[MaintenanceWindow], List[Dict[str, Any]]]:
    """Schedule jobs over windows using a WINDOW-GATED discipline (paper).

    Key paper semantics:
      - Gating epoch = window opening.
      - Only jobs arrived up to the window opening are eligible for EXAMINATION in that window.
      - Per-window administrative cap: examine at most q(w) candidates.
      - Start a job only if fit_i(w) and it fits remaining window capacity (via can_fit/assign).
      - If examined but cannot start: defer to NEXT WINDOW (deferrals++).
      - Anti-starvation promotion is enabled ONLY for RD+WGPS and ONLY within bucket.

    Snapshots:
      - Daily snapshots at t=24*d (days), used for backlog-age P95 timeseries.
      - Backlog at time now includes jobs with arrival<=now and (start_time is None OR start_time>now).
    """
    params.validate()
    if fairness_horizon_H < 1:
        raise ValueError("fairness_horizon_H must be >= 1.")

    # Ensure chronological windows
    windows = sorted(windows, key=lambda w: float(w.start_time_h))

    # Arrivals sorted by arrival time (global stream)
    arrivals_sorted = sorted(jobs.values(), key=lambda j: float(j.arrival_time_h))
    arr_ptr = 0

    # Backlog structures
    BUCKETS = ["High", "Medium", "Low"]
    if policy == "RD+WGPS":
        backlog_b: Dict[str, List[PatchJob]] = {b: [] for b in BUCKETS}
        backlog_q: Optional[List[PatchJob]] = None
    else:
        backlog_q = []
        backlog_b = {}

    # Snapshot horizon
    if horizon_days is not None:
        horizon_h = float(horizon_days) * 24.0
    else:
        horizon_h = 0.0
        if windows:
            horizon_h = max(horizon_h, max(float(w.start_time_h + w.length_h) for w in windows))
        if jobs:
            horizon_h = max(horizon_h, max(float(j.arrival_time_h) for j in jobs.values()))
        horizon_h += 24.0  # one extra day to see tail backlog if desired

    snapshot_times_h = [24.0 * d for d in range(int(math.ceil(horizon_h / 24.0)) + 1)]
    snap_idx = 0
    snapshots: List[Dict[str, Any]] = []

    def take_snapshot(now_h: float) -> None:
        # backlog: arrived but not yet started by now_h
        backlog_now = [
            j for j in jobs.values()
            if float(j.arrival_time_h) <= now_h and (j.start_time_h is None or float(j.start_time_h) > now_h)
        ]

        snap: Dict[str, Any] = {"time_h": now_h, "policy": str(policy)}
        for b in BUCKETS:
            ages_days = [(now_h - float(j.arrival_time_h)) / 24.0 for j in backlog_now if j.bucket == b]
            if not ages_days:
                snap[f"backlog_age_p95_{b}"] = float("nan")
                snap[f"backlog_n_{b}"] = 0
            else:
                snap[f"backlog_age_p95_{b}"] = float(np.quantile(np.asarray(ages_days, dtype=float), 0.95))
                snap[f"backlog_n_{b}"] = int(len(ages_days))
        snapshots.append(snap)

    # ---------------- main loop over windows ----------------
    for w in windows:
        now_h = float(w.start_time_h)

        # snapshots strictly up to this window opening
        while snap_idx < len(snapshot_times_h) and snapshot_times_h[snap_idx] <= now_h + 1e-12:
            take_snapshot(snapshot_times_h[snap_idx])
            snap_idx += 1

        # Add arrivals up to this gating epoch (window opening) into backlog
        while arr_ptr < len(arrivals_sorted) and float(arrivals_sorted[arr_ptr].arrival_time_h) <= now_h + 1e-12:
            j = arrivals_sorted[arr_ptr]
            # only enqueue if not already started (should be true)
            if j.start_time_h is None:
                if policy == "RD+WGPS":
                    backlog_b[j.bucket].append(j)
                else:
                    assert backlog_q is not None
                    backlog_q.append(j)
            arr_ptr += 1

        # If nothing eligible at this gating epoch, continue
        if policy == "RD+WGPS":
            if all(len(backlog_b[b]) == 0 for b in BUCKETS):
                continue
        else:
            assert backlog_q is not None
            if not backlog_q:
                continue

        # -------------- RD+WGPS (paper Algorithm 1) --------------
        if policy == "RD+WGPS":
            # Build bucketed snapshot queues Q[b] at window opening
            Q: Dict[str, List[PatchJob]] = {}
            Deferred: Dict[str, List[PatchJob]] = {b: [] for b in BUCKETS}

            for b in BUCKETS:
                promoted = [j for j in backlog_b[b] if int(j.deferrals) >= int(fairness_horizon_H)]
                normal = [j for j in backlog_b[b] if int(j.deferrals) < int(fairness_horizon_H)]

                # Sorting inside bucket follows paper Algorithm 1
                promoted.sort(key=lambda j: wgps_bucket_order_key(j, now_h, params, fairness_horizon_H=fairness_horizon_H))
                # For normal, Algorithm uses decreasing RD only (we reuse same key but it will place non-promoted together;
                # simpler/clearer: explicit normal sort)
                normal.sort(
                    key=lambda j: (
                        -float(rd_score(j, now_h=now_h, p=params)),
                        float(j.arrival_time_h),
                        int(j.job_id),
                    )
                )

                Q[b] = promoted + normal
                backlog_b[b] = []  # Backlog[b] <- empty (paper)

            # Serve within window: examine up to q(w), select from highest nonempty bucket
            while (
                any(Q[b] for b in BUCKETS)
                and w.can_examine_more()
                and w.rem_h() > 1e-12
            ):
                # pick highest-priority nonempty bucket
                b_star = next(b for b in BUCKETS if Q[b])
                job = Q[b_star].pop(0)

                # examination count a <- a+1 (paper)
                w.record_examination(job.job_id)

                # fit_i(w): placeholder True (extend with site/class/calendar constraints)
                fit = True

                if fit and w.can_fit(job.duration_h):
                    start_t = w.assign(job.job_id, job.duration_h)
                    if start_t is not None:
                        job.start_time_h = float(start_t)
                        job.assigned_window_idx = int(w.idx)
                    else:
                        job.deferrals += 1
                        Deferred[b_star].append(job)
                else:
                    job.deferrals += 1
                    Deferred[b_star].append(job)

            # End of window: Backlog[b] <- remaining Q[b] ∪ Deferred[b] (paper)
            for b in BUCKETS:
                backlog_b[b].extend(Q[b])
                backlog_b[b].extend(Deferred[b])

        # -------------- Baselines (paper) --------------
        else:
            assert backlog_q is not None

            # Gate snapshot Q at window opening
            Q = sorted(backlog_q, key=lambda j: baseline_order_key(j, now_h, policy, params))
            backlog_q = []  # Backlog <- empty at window opening (paper)

            Deferred: List[PatchJob] = []

            while Q and w.can_examine_more() and w.rem_h() > 1e-12:
                job = Q.pop(0)
                w.record_examination(job.job_id)

                fit = True
                if fit and w.can_fit(job.duration_h):
                    start_t = w.assign(job.job_id, job.duration_h)
                    if start_t is not None:
                        job.start_time_h = float(start_t)
                        job.assigned_window_idx = int(w.idx)
                    else:
                        # examined but not started => defer
                        job.deferrals += 1
                        Deferred.append(job)
                else:
                    job.deferrals += 1
                    Deferred.append(job)

            # Backlog <- remaining Q + Deferred
            backlog_q.extend(Q)
            backlog_q.extend(Deferred)

    # trailing snapshots up to horizon
    while snap_idx < len(snapshot_times_h):
        take_snapshot(snapshot_times_h[snap_idx])
        snap_idx += 1

    return jobs, windows, snapshots
