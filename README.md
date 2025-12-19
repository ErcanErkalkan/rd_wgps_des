# RD–WGPS DES (reproducible discrete‑event simulation)

This repository contains a **minimal, publishable-quality** Python implementation of the simulation used in the paper:

**“Queueing Analysis by Simulation of Risk–Duration–Prioritized Patch Scheduling Under Gated Windows”**

It implements:

- **RD score** (*Risk–Duration*): a “risk-per-unit-time” ranking
  \[
  RD_i=\frac{R_i}{\max(D_i,\varepsilon)},\quad
  R_i=w_C C_i+w_X X_i+w_A\tilde{A}_i+w_K K_i+w_T T_i,\quad
  \tilde{A}_i=\min(A_i/A_{\max},1)
  \]
- **WGPS** (*Window-Gated Patch Scheduler*, Algorithm 1): at each window opening (gating epoch),
  build **bucketed snapshot queues** (High/Med/Low), examine at most **q(w)** candidates, and start a job only if
  it is admissible **and** fits the residual window capacity; otherwise it is **deferred to the next window**.
  Starvation is mitigated via the **fairness horizon** `H` (promotion within the same bucket once deferrals ≥ H).

## Structure

- `src/rd_wgps_des/` — package code (RD scoring, synthetic signals, DES, plots, metrics)
- `scripts/run_des.py` — experiment runner (writes `results/<run_name>/...`)
- `scripts/render_latex_tables.py` — renders paper-ready LaTeX rows from `metrics.csv`
- `configs/run_default.yaml` — configuration aligned with paper scenarios **S1–S3**

## Minimal requirements

This codebase is intentionally lightweight:

- Python 3.10+ (tested with 3.11)
- `PyYAML`, `numpy`, `pandas`, `matplotlib`

## How to run

From the repository root:

```bash
python scripts/run_des.py --config configs/run_default.yaml
```

### Outputs

The runner produces:

- `results/<run_name>/metrics.csv` — combined metrics for all scenarios & policies (KM-based wait-time quantiles)
- Per-scenario figures under `results/<run_name>/<scenario>/`:
  - `fig_<scenario>_wait_ccdf.pdf`
  - `fig_<scenario>_backlog_timeseries.pdf`
  - `fig_<scenario>_util_per_window.pdf`
- Per-policy snapshots under `results/<run_name>/<scenario>/<policy_id>/backlog_snapshots.csv`

**Tip (paper tables):**

```bash
python scripts/render_latex_tables.py --results_dir results/<run_name>
```

This writes small `latex_rows_*.txt` snippets into the same results folder.

## Configuration notes

Key configuration fields in `configs/run_default.yaml`:

- `policy.weights` — the RD numerator weights `(wC,wX,wA,wK,wT)` (must sum to 1)
- `policy.Amax_days`, `policy.eps_hours` — age cap and epsilon in RD denominator
- `policy.fairness.horizon_H` — the fairness horizon `H` used by **RD+WGPS**
- `windows.base.q_examinations_per_window` — the per-window examination cap `q(w)`
  - (alias: `max_items` is also accepted by the runner)
- `signals:` — optional overrides for the synthetic signal generator (criticality levels, exposure distributions, age hazard, etc.)
- `scenarios:` — defines S1–S3 arrival/duration patterns and (optional) `windows_override`

## Reproducibility

- The runner uses deterministic seeding derived from `seed`, scenario index, and policy index.
- Results should be reproducible across runs given the same config.

## Citation

If you reuse this code for academic work, please cite the accompanying paper.

## License
MIT License — see `LICENSE`.

