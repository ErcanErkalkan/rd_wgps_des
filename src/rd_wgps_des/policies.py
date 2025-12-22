from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from .models import PatchJob


# Paper policy names
PolicyName = Literal["FIFO", "SPT", "CVSS-only", "R-risk-only", "RD-only", "RD+WGPS"]


@dataclass(frozen=True)
class RDParams:
    """Parameters for the RD score (paper Eq. (rd)).

    Weights must sum to 1.0:
      R_i = wC*C + wX*X + wA*A_tilde + wK*K + wT*T
      RD_i = R_i / max(D_i, eps)
    """
    wC: float
    wX: float
    wA: float
    wK: float
    wT: float

    Amax_days: float = 30.0
    eps_hours: float = 0.25

    def validate(self) -> None:
        ws = (self.wC, self.wX, self.wA, self.wK, self.wT)
        if any(w < 0 for w in ws):
            raise ValueError("RD weights must be nonnegative.")
        s = float(sum(ws))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"RD weights must sum to 1.0 (got {s}).")
        if self.Amax_days <= 0:
            raise ValueError("Amax_days must be > 0.")
        if self.eps_hours <= 0:
            raise ValueError("eps_hours must be > 0.")


def _age_tilde(job: PatchJob, now_h: float, Amax_days: float) -> float:
    """A_tilde = min(A/Amax, 1) with A growing over time."""
    return min(job.age_days_at(now_h) / float(Amax_days), 1.0)


def risk_only_R(job: PatchJob, now_h: float, p: RDParams) -> float:
    """R_i (paper Eq. risk_only)."""
    a = _age_tilde(job, now_h, p.Amax_days)
    return (
        p.wC * float(job.criticality)
        + p.wX * float(job.exposure)
        + p.wA * float(a)
        + p.wK * float(job.kev)
        + p.wT * float(job.threat)
    )


def rd_score(job: PatchJob, now_h: float, p: RDParams) -> float:
    """RD_i (paper Eq. rd)."""
    R = risk_only_R(job, now_h, p)
    return float(R / max(float(job.duration_h), float(p.eps_hours)))


# ---------------------------------------------------------------------
# Baseline ordering keys (paper baselines)
# ---------------------------------------------------------------------

def baseline_order_key(job: PatchJob, now_h: float, policy: PolicyName, p: RDParams) -> Tuple:
    """Ascending sort key (Python sorts ascending); higher priority => smaller key.

    For baselines (FIFO, CVSS-only, R-risk-only, RD+WGPS when *not* using WGPS logic).
    In the paper, RD+WGPS uses Algorithm 1 (gated + buckets + promotion),
    so you typically should NOT use this for RD+WGPS service ordering.
    """
    if policy == "FIFO":
        return (float(job.arrival_time_h), int(job.job_id))

    if policy == "SPT":
        # shortest processing time (duration-only)
        return (float(job.duration_h), float(job.arrival_time_h), int(job.job_id))

    if policy == "CVSS-only":
        # higher severity first; ties by arrival
        return (-float(job.cvss), float(job.arrival_time_h), int(job.job_id))

    if policy == "R-risk-only":
        R = risk_only_R(job, now_h, p)
        return (-float(R), float(job.arrival_time_h), int(job.job_id))

    if policy == "RD-only":
        RD = rd_score(job, now_h, p)
        return (-float(RD), float(job.arrival_time_h), int(job.job_id))

    if policy == "RD+WGPS":
        RD = rd_score(job, now_h, p)
        return (-float(RD), float(job.arrival_time_h), int(job.job_id))

    raise ValueError(f"Unknown policy: {policy}")


# ---------------------------------------------------------------------
# WGPS bucket-internal ordering (paper Algorithm 1)
# ---------------------------------------------------------------------

def wgps_bucket_order_key(
    job: PatchJob,
    now_h: float,
    p: RDParams,
    *,
    fairness_horizon_H: int,
) -> Tuple:
    """Ordering inside a bucket at a gating epoch (window opening).

    Paper Algorithm 1:
      Promoted[b] = { i in Backlog[b] : r(i) >= H }
      sort Promoted by: decreasing r(i), then older A_i, then decreasing RD_i
      sort Normal by: decreasing RD_i
      Q[b] = Promoted || Normal  (snapshot at window opening)

    This key returns a tuple that, when used with ascending sort, matches:
      - promoted first (r>=H)
      - within promoted: larger r, then older age, then larger RD
      - within normal: larger RD
    """
    r = int(getattr(job, "deferrals", 0))  # r(i)
    promoted = 1 if r >= int(fairness_horizon_H) else 0

    # Larger age => older => higher priority in promoted tie-break
    age_days = float(job.age_days_at(now_h))
    RD = float(rd_score(job, now_h, p))

    # Ascending key: promoted first => -promoted
    # Within promoted: want r desc, age desc, RD desc
    # Within normal: want RD desc; r/age still included but with promoted=0 it won't dominate
    return (
        -promoted,
        -r,
        -age_days,
        -RD,
        float(job.arrival_time_h),
        int(job.job_id),
    )
