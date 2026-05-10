# §IV.C Execution — Progress Log

## Phase P — Prerequisite: F-H-R fold solver

- **Status:** PASS
- **Sims run:** 0 (purely thermodynamic; no PDE)
- **Wall-clock:** ~3 s
- **Script:** `scripts/iv_c_fold_solver.py`
- **Outputs:**
  - `data/iv_c/folds/default.npz` — folds at default `Params()`
  - `data/iv_c/folds/S_chi_sweep.npz` — `S_chi ∈ [0.3, 2.0]`, 35 points
  - `data/iv_c/folds/default_theta_J_curve.npz` — dense θ(J) curve (diagnostic)

### Method

At fixed `S_chi`, μ(J,θ)=μ_b is linear in θ → explicit
`θ(J) = (μ_b - μ_base(J)) / (S_chi · φ(J)²)` (single-valued in J).
Folds = local extrema of θ(J), refined by `fsolve` on simultaneous
`μ=μ_b ∧ ∂_J μ=0`. Bath value `μ_b = m_bath(p) = μ(J_init, θ=0) ≈ 0.0662`
(matches `scan_optimized.py`/`linear_stability_1d.py`; **not** the 0 in the
plan's draft pseudocode).

### Key numbers at default `Params()`

| quantity                    | measured   | target   | error |
|-----------------------------|-----------:|---------:|------:|
| μ_b                         |   0.0662   |   —      |   —   |
| θ_lo                        |   0.8442   |   ~0.85  | 0.7%  |
| θ_up                        |   3.1843   |   ~3.18  | 0.1%  |
| Δθ                          |   2.3401   |   ~2.3   | 1.7%  |
| ln(θ_up/θ_lo)               |   1.3276   |   —      |   —   |
| J_lo* (collapsed fold J)    |   0.1947   |   ~0.2   |       |
| J_up* (swollen fold J)      |   0.7975   |   —      |       |
| J_collapsed_at_up           |   0.1511   |   —      |       |
| J_swollen_at_lo             |   1.2341   |   —      |       |
| ΔJ (full cycle)             |   1.0830   |   ~1.2   | ~10%  |
| ΔJ jump @ θ_up              |   0.6464   |   —      |       |
| ΔJ jump @ θ_lo              |   1.0393   |   —      |       |

### S_chi sweep summary

- All 35 points succeeded; **no NaN anywhere** (well beyond plan's required `S_χ ∈ [0.5, 1.6]`).
- `Δθ · S_chi = 2.340` exactly **constant** across the sweep (std = 0.000).
  This is the analytic side of scaling-law (iii) — the value `h ≈ 2.34`
  becomes the prediction PDE measurements must match in Phase A.
- ΔJ also constant (1.0830) across `S_chi`, as expected: J-folds are set
  by `μ_base(J)` which has no `S_chi` dependence; `S_chi` only rescales θ.

### Issues encountered

- None. Algorithm converged at every grid point.
- Pre-existing utilities `chem_pot`, `df_dJ`, `m_bath` from
  `scripts/linear_stability_1d.py` reused without modification.

### Decision

PASS — proceed to **Phase A (sanity check: amplitude scaling)** when user confirms.

---

## Phase A — Sanity check: amplitude scaling

- **Status:** PASS
- **Sims run:** 12 PDE
- **Wall-clock:** ~3.5 min (12 workers, N=101)
- **Scripts:** `scripts/iv_c_phaseA_amplitude.py`, `scripts/iv_c_phaseA_check.py`
- **Outputs:**
  - `data/iv_c/phaseA/amplitude_results.json`  — per-point measurements
  - `data/iv_c/phaseA/amplitude_h_analytic.npz` — Phase P prediction
  - `data/iv_c/phaseA/phaseA_check.npz`        — decision summary

### Settings (overrode plan defaults)

| field    | plan default | used | reason |
|----------|-------------:|-----:|--------|
| `N`      | 51           | 101  | At N=51 the LF (oscillation) region is much narrower than at the fig4 reference of N=301; only 4/12 points oscillated. N=101 recovered 6/12 oscillating cleanly. Plan §12 explicitly allows escalating N when needed. |
| `t_end`  | 300          | 400  | Borderline points (low Bi_T) need a longer settling window; transient ignition can take ~100 t-units. |
| `t_window` | (150,300)  | (200,400) | Same reason — analyze the latter 200 t-units. |

### Result table — oscillating points

| Bi_T | S_chi | Δθ_surf | meas h = Δθ·S_χ | rel err vs h_pred=2.340 |
|-----:|------:|--------:|----------------:|------------------------:|
| 0.05 | 0.70  | 3.2535  | 2.2774          | 2.68%   |
| 0.10 | 0.70  | 3.7395  | 2.6176          | 11.86%  |
| 0.10 | 1.00  | 2.6536  | 2.6536          | 13.39%  |
| 0.10 | 1.30  | 1.9101  | 2.4832          | 6.11%   |
| 0.15 | 0.80  | 2.7552  | 2.2042          | 5.81%   |
| 0.15 | 1.10  | 2.4061  | 2.6467          | 13.10%  |

- max rel err: **13.39%** (PASS threshold = 20%)
- mean rel err: 8.83%

### Non-oscillating points (not failures of scaling — outside LF region at N=101)

| Bi_T | S_chi | end-state | comment |
|-----:|------:|-----------|---------|
| 0.05 | 1.0   | hot-runaway SS  (J_surf=0.15, θ=2.88) | LF in fig4 (N=301); resolution-sensitive boundary |
| 0.05 | 1.3   | hot-runaway SS  (J_surf=0.15, θ=2.26) | "" |
| 0.10 | 1.6   | hot-runaway SS  (J_surf=0.15, θ=1.81) | "" |
| 0.20 | 0.9   | warm SS         (J_surf=1.01, θ=2.95) | steady-front per fig4 |
| 0.20 | 1.2   | warm SS         (J_surf=3.91, θ=3.39) | "" |
| 0.25 | 1.0   | warm SS         (J_surf=1.09, θ=2.16) | borderline LF/cold per fig4 |

These do **not** invalidate the scaling — the 6 oscillating points already span
S_χ ∈ [0.7, 1.3] and Bi_T ∈ [0.05, 0.15], which is enough to verify that
Δθ·S_χ is approximately constant (the prediction is parameter-independent within
the LF region).

### Issues encountered

- Initial Windows-console UnicodeEncodeError when printing ↔ (GBK codec). Fixed by stripping decorative non-ASCII chars from print statements; data save now happens before final summary print.

### Decision

PASS — proceed to **Phase B (period scaling)** when user confirms.

---
