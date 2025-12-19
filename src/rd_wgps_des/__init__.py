"""rs_wgps_des

A small, reproducible discrete-event simulation (DES) for evaluating
Risk–Duration (RD) prioritization and a Window-Gated Patch Scheduler (WGPS)
under gated maintenance windows.

This codebase is intended to accompany the papers:
- "Risk–Duration Score for Patch Prioritization in SME IoT Fleets"
- "Queueing Analysis by Simulation of Risk–Duration–Prioritized Patch Scheduling Under Gated Windows"

Design goals:
- Minimal dependencies (PyYAML, numpy, pandas, matplotlib)
- Transparent, configurable synthetic signal generation (paper generator table)
- Window-gated scheduling with an examination cap q(w) and fairness horizon H (Algorithm 1)
"""

# Policies / scoring
from .policies import RDParams, PolicyName

# Backward-compatibility alias (optional)
RSParams = RDParams  # legacy name; prefer RDParams

# Simulation + configs
from .sim import (
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

# Metrics + plots
from .metrics import summarize_run
from .plots import plot_ccdf, plot_backlog_timeseries, plot_utilization_per_window


__all__ = [
    # scoring / policy
    "RDParams",
    "RSParams",      # legacy alias
    "PolicyName",
    # configs
    "ArrivalConfig",
    "DurationConfig",
    "WindowConfig",
    "BurstConfig",
    "SignalGenConfig",
    # generators / scheduler
    "generate_windows",
    "generate_arrivals",
    "assign_buckets",
    "schedule_over_windows",
    # outputs
    "summarize_run",
    "plot_ccdf",
    "plot_backlog_timeseries",
    "plot_utilization_per_window",
]
