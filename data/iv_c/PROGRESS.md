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
