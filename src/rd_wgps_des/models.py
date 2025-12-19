from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ---------------------------- PatchJob ---------------------------------


@dataclass
class PatchJob:
    """A single patch/change item in the DES.

    Times are expressed in hours since simulation start (t=0).

    Paper mapping:
      - D_i  -> duration_h
      - C_i  -> criticality
      - X_i  -> exposure
      - A_i  -> vuln_age_days (at arrival; grows with time)
      - K_i  -> kev
      - T_i  -> threat
      - r(i) -> deferrals (incremented when examined but not started)
    """
    job_id: int
    arrival_time_h: float

    # Fleet / effort
    device_class: str  # "IT", "IoT", "OT"
    duration_h: float  # service time / expected maintenance duration D_i

    # Risk signals (normalized)
    criticality: float    # C_i in [0,1]
    exposure: float       # X_i in [0,1]
    vuln_age_days: float  # A_i at arrival (days since disclosure)
    kev: int              # K_i in {0,1}
    threat: float         # T_i in [0,1]

    # Severity baseline (0..10), used only by CVSS ordering
    cvss: float

    # Reporting bucket (High / Medium / Low)
    bucket: str = "Unassigned"

    # Optional: RD at arrival (useful if buckets are defined by RD quantiles at arrival)
    rd_at_arrival: Optional[float] = None

    # Scheduling outcomes
    start_time_h: Optional[float] = None
    assigned_window_idx: Optional[int] = None
    deferrals: int = 0  # r(i): number of deferrals after examination

    def is_started(self) -> bool:
        return self.start_time_h is not None

    def waiting_time_h(self) -> Optional[float]:
        if self.start_time_h is None:
            return None
        return float(self.start_time_h - self.arrival_time_h)

    def age_days_at(self, now_h: float) -> float:
        """Vulnerability age at time `now_h` (days)."""
        delta_days = max(0.0, (float(now_h) - float(self.arrival_time_h)) / 24.0)
        return float(self.vuln_age_days + delta_days)

    def age_tilde_at(self, now_h: float, Amax_days: float) -> float:
        """Normalized/capped age: \tilde{A} = min(A/Amax, 1)."""
        if Amax_days <= 0:
            return 1.0
        return min(self.age_days_at(now_h) / float(Amax_days), 1.0)

    def risk_score_R(
        self,
        now_h: float,
        weights: Dict[str, float],
        Amax_days: float,
        *,
        use_kev_threat: bool = True,
    ) -> float:
        """Risk-only score R_i (paper Eq. (risk_only))."""
        wC = float(weights.get("wC", 0.0))
        wX = float(weights.get("wX", 0.0))
        wA = float(weights.get("wA", 0.0))
        wK = float(weights.get("wK", 0.0))
        wT = float(weights.get("wT", 0.0))

        A_tilde = self.age_tilde_at(now_h, Amax_days)

        K = float(self.kev) if use_kev_threat else 0.0
        T = float(self.threat) if use_kev_threat else 0.0

        return (
            wC * float(self.criticality)
            + wX * float(self.exposure)
            + wA * float(A_tilde)
            + wK * K
            + wT * T
        )

    def rd_score(
        self,
        now_h: float,
        weights: Dict[str, float],
        Amax_days: float,
        eps_hours: float,
        *,
        use_kev_threat: bool = True,
    ) -> float:
        """RD_i = R_i / max(D_i, eps)  (paper Eq. (rd))."""
        R = self.risk_score_R(
            now_h=now_h,
            weights=weights,
            Amax_days=Amax_days,
            use_kev_threat=use_kev_threat,
        )
        denom = max(float(self.duration_h), float(eps_hours))
        return float(R / denom)


# ------------------------- MaintenanceWindow ---------------------------


@dataclass
class MaintenanceWindow:
    """A maintenance window with limited length and parallelism.

    Paper mapping:
      - L(w) -> length_h
      - q(w) -> max_examinations
      - 'a'  -> examinations_used
    """
    idx: int
    start_time_h: float
    length_h: float
    parallelism: int
    max_examinations: int  # q(w): max candidate examinations in this window

    # Internal per-server cursor (hours since window start); used to compute start times.
    _server_cursors_h: List[float] = field(default_factory=list)

    # Records of assignments (job_id, server_id, start_offset_h, duration_h)
    assignments: List[Tuple[int, int, float, float]] = field(default_factory=list)

    # Exam counter (a in paper): how many candidates were examined in this window
    examinations_used: int = 0
    examined_job_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        if self.length_h <= 0:
            raise ValueError("length_h must be > 0")
        if self.max_examinations < 0:
            raise ValueError("max_examinations must be >= 0")
        self._server_cursors_h = [0.0 for _ in range(self.parallelism)]

    @property
    def end_time_h(self) -> float:
        return float(self.start_time_h + self.length_h)

    def can_examine_more(self) -> bool:
        """Whether the administrative cap q(w) allows examining another candidate."""
        return self.examinations_used < int(self.max_examinations)

    def record_examination(self, job_id: int) -> None:
        """Increment examination count (a <- a+1) and record which job was examined."""
        self.examinations_used += 1
        self.examined_job_ids.append(int(job_id))

    def can_fit(self, duration_h: float) -> bool:
        """Returns True if any server can fit this job within the window."""
        d = float(duration_h)
        for cur in self._server_cursors_h:
            if cur + d <= float(self.length_h) + 1e-12:
                return True
        return False

    def rem_h(self) -> float:
        """A practical 'remaining time' indicator.

        For parallelism=1 this equals the paper's rem.
        For parallelism>1 we return the maximum remaining time across servers.
        """
        return max(float(self.length_h) - float(cur) for cur in self._server_cursors_h)

    def assign(self, job_id: int, duration_h: float) -> Optional[float]:
        """Assign a job to the earliest-available server.

        Returns absolute start time (hours since simulation start) if assigned,
        else None if infeasible under window feasibility.
        """
        d = float(duration_h)

        best_server: Optional[int] = None
        best_cursor: Optional[float] = None

        for sid, cur in enumerate(self._server_cursors_h):
            if cur + d <= float(self.length_h) + 1e-12:
                if best_cursor is None or cur < best_cursor:
                    best_cursor = float(cur)
                    best_server = int(sid)

        if best_server is None or best_cursor is None:
            return None

        start_offset_h = best_cursor
        self._server_cursors_h[best_server] = best_cursor + d
        self.assignments.append((int(job_id), int(best_server), float(start_offset_h), float(d)))
        return float(self.start_time_h + start_offset_h)

    def busy_time_h(self) -> float:
        """Total busy time across all servers (sum of assigned durations)."""
        return float(sum(d for _, _, _, d in self.assignments))

    def capacity_h(self) -> float:
        return float(self.length_h) * float(self.parallelism)

    def utilization(self) -> float:
        cap = self.capacity_h()
        return 0.0 if cap <= 0 else float(min(1.0, self.busy_time_h() / cap))

    def end_of_window_start_share(self, delta_minutes: float = 30.0) -> float:
        """Share of starts that occur within the last `delta_minutes` of the window."""
        if not self.assignments:
            return 0.0
        delta_h = float(delta_minutes) / 60.0
        threshold = max(0.0, float(self.length_h) - delta_h)
        late = sum(1 for _, _, start_off, _ in self.assignments if float(start_off) >= threshold - 1e-12)
        return float(late / len(self.assignments))
