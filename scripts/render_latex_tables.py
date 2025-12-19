#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


POLICY_ORDER = [
    "FIFO",
    "CVSS-only",
    "R (risk-only)",
    "RD+WGPS",
]


def _cell(v, decimals: int = 1) -> str:
    """Format a LaTeX table cell.
    - If v is a string (e.g., '>\\!209.5'), return as-is.
    - If NaN/None, return '--'.
    - Else format numeric with given decimals.
    """
    if v is None:
        return "--"
    if isinstance(v, str):
        s = v.strip()
        return s if s else "--"
    try:
        if pd.isna(v):
            return "--"
    except Exception:
        pass
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return "--"


def _get(row: pd.Series, keys: list[str], default=None):
    for k in keys:
        if k in row.index:
            return row[k]
    return default


def _max_bound_cell(v) -> str:
    """Max bound shown as >= value (LaTeX)."""
    if v is None:
        return "--"
    if isinstance(v, str):
        s = v.strip()
        return s if s else "--"
    try:
        if pd.isna(v):
            return "--"
    except Exception:
        pass
    try:
        return f"$\\ge {_cell(float(v), decimals=1)}$"
    except Exception:
        return "--"


def _started_over_arrivals(row: pd.Series) -> str:
    hs = _get(row, ["high_scheduled", "started_high", "started"], default=None)
    ha = _get(row, ["high_arrivals", "arrivals_high", "arrivals"], default=None)

    # already provided as string?
    soa = _get(row, ["high_started_over_arrivals", "started_over_arrivals"], default=None)
    if isinstance(soa, str) and soa.strip():
        return soa.strip()

    try:
        hs_i = int(hs)
        ha_i = int(ha)
        return f"{hs_i}/{ha_i}" if ha_i > 0 else "--"
    except Exception:
        return "--"


def _sort_policy(df: pd.DataFrame) -> pd.DataFrame:
    if "policy" not in df.columns:
        return df
    order_map = {p: i for i, p in enumerate(POLICY_ORDER)}
    df = df.copy()
    df["_pord"] = df["policy"].map(lambda x: order_map.get(str(x), 999))
    df = df.sort_values(["_pord", "policy"]).drop(columns=["_pord"])
    return df


def _make_table_rows(df: pd.DataFrame, scenario: str, outfile: Path) -> None:
    d = df[df["scenario"] == scenario].copy()

    # Some older metric files might include per-bucket rows; if so, keep High.
    if "bucket" in d.columns:
        d = d[d["bucket"] == "High"].copy()

    if d.empty:
        outfile.write_text("", encoding="utf-8")
        return

    d = _sort_policy(d)

    lines = []
    for _, row in d.iterrows():
        policy = str(_get(row, ["policy"], default="--"))

        km_p50 = _get(row, ["wait_km_p50_h", "wait_p50_h", "km_p50_h"], default=None)
        km_p90 = _get(row, ["wait_km_p90_h", "wait_p90_h", "km_p90_h"], default=None)

        # If your metrics store display-ready strings, keep them (e.g., '>\!209.5')
        km_p50_cell = _cell(km_p50, decimals=1)
        km_p90_cell = _cell(km_p90, decimals=1)

        maxb = _get(row, ["max_bound_h", "wait_max_bound_h", "max_bound"], default=None)
        maxb_cell = _max_bound_cell(maxb)

        b95 = _get(row, ["backlog_p95_mean_days", "backlog_p95_mean_d", "backlog_p95_mean"], default=None)
        b95_cell = _cell(b95, decimals=2)

        started = _started_over_arrivals(row)

        lines.append(
            f"{policy} & {started} & {km_p50_cell} & {km_p90_cell} & {maxb_cell} & {b95_cell} \\\\"
        )

    outfile.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="e.g., results/run_default (relative to repo root)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.is_absolute():
        rdir = (ROOT / rdir).resolve()

    metrics_path = rdir / "metrics.csv"
    df = pd.read_csv(metrics_path)

    # Write LaTeX row snippets (paper tables: S1/S2/S3, High bucket, censoring-aware)
    _make_table_rows(df, "S1_normal", rdir / "latex_rows_S1_high.txt")
    _make_table_rows(df, "S2_kev_burst", rdir / "latex_rows_S2_high.txt")
    _make_table_rows(df, "S3_lowcap_hetero", rdir / "latex_rows_S3_high.txt")

    print("[OK] Wrote LaTeX row snippets into results dir:")
    print(" - latex_rows_S1_high.txt")
    print(" - latex_rows_S2_high.txt")
    print(" - latex_rows_S3_high.txt")


if __name__ == "__main__":
    main()
