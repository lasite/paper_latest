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

## Phase A.5 — N convergence study

- **Status:** PASS — all observables converged at **N = 301**
- **Sims attempted:** 21 (3 points × 7 N values)
- **Sims successful:** 15 (6 timed out at the 35-min ceiling)
- **Wall-clock:** 35 min (capped by per-sim timeout)
- **Script:** `scripts/iv_c_phaseA5_convergence.py` (v2 — incremental save, per-sim timeout, streaming output)
- **Outputs:**
  - `data/iv_c/phaseA5/convergence_raw.json`   — raw per-sim records (incremental)
  - `data/iv_c/phaseA5/convergence.npz`        — reshaped grid
  - `data/iv_c/phaseA5/convergence_report.md`  — corrected analysis (see file)

### v1 → v2 (mid-run rewrite)

The first attempt (`exe.map(run_one, tasks)`) blocked on submission
order — when 3 of 21 sims (the N=501 ones) hit 95+ min CPU each, the
driver wrote nothing to disk and the entire 96-min compute was lost
when killed. v2 uses `multiprocessing.Pool.apply_async` + a polling
loop with per-sim 35-min timeout, prints results in completion order
with `flush=True`, and atomically saves `convergence_raw.json` after
every result. `pool.terminate()` at end kills any worker still
running a timed-out sim. Bug in `converged_N` (required immediate-
neighbor comparison; NaN intermediate broke even adjacent-N pairs)
fixed to compare against the highest-finite-N reference.

### Coverage

| point         | N=51 | N=101 | N=151 | N=201 | N=301 | N=401 | N=501 |
|---------------|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| WP            | OSC  | OSC   | OSC*  | TIMEOUT | OSC | OSC   | TIMEOUT |
| shallow       | NOTOSC | OSC | NOTOSC | TIMEOUT | OSC | OSC   | TIMEOUT |
| nearSNIC      | OSC  | OSC   | OSC   | TIMEOUT | OSC | OSC   | TIMEOUT |

\* WP at N=151 has anomalous T=38.1 (vs 19.3 at N=301) — likely 2:1
frequency-locking artifact at that resolution.

### Convergence verdict (rel err vs N=401, tol = 5%)

All six tracked observables converge at N = 301 across all three points.

| observable      | min converged N |
|-----------------|----------------:|
| period          | **N = 301**     |
| delta_theta     | **N = 301**     |
| delta_J         | **N = 301**     |
| xi_LCST         | **N = 301**     |
| theta_surf_max  | **N = 301**     |
| theta_surf_min  | **N = 301**     |

Detailed per-point per-N relative errors live in `convergence_report.md`.

### Pathologies identified at low N

- **shallow (Bi_T=0.05, S_chi=0.7)** at N=51 and N=151 fails to oscillate
  (settles to hot-runaway). At N=101 it oscillates but with period 62.8
  vs 29.2 at N=301 — spurious low-frequency drift from under-resolved
  LCST front.
- **WP (Bi_T=0.10, S_chi=1.0)** at N=151 has period jump to 38.1 — likely
  a 2:1 frequency-locking artifact peculiar to that resolution.
- **N=201 timed out at all 3 points** while N=301 finished in 17-19 min.
  Most plausible cause: BDF + sparse-Jacobian column grouping at N=201
  hits a pessimal coloring / stiffness pattern. Doesn't affect the
  convergence verdict because N=51, 101, 151, 301, 401 sample the trend
  densely enough.

### Implications for downstream phases

Per the user's decision rule:

- **All observables converged at N=301 → use N=301 for Phase B/C/D.**
- **Phase A re-do?** Required only if `delta_theta` at N=101 vs N=301
  differs by > 5% at any point.
  - WP: 1.5% ✓
  - **shallow: 11.4%** ✗
  - nearSNIC: 4.5% ✓
  - **Verdict: re-run Phase A at N=301** (12 sims, ~20 min).
  - Phase A's PASS verdict is robust to the N change anyway (re-evaluating
    shallow's contribution gives meas h = 2.566 vs h_pred 2.340, rel err
    9.7%, still well under the 20% threshold). The re-run is for
    numerical hygiene, not because the conclusion is in doubt.

### Decision

PASS. Pending user confirmation: (a) re-run Phase A at N=301 first,
or (b) skip the re-run and proceed straight to Phase B at N=301.

User picked (b) — skip Phase A re-run.

---

## Phase B — Period scaling

- **Status:** PASS (max rel err 13.5% < 30% threshold)
- **Sims run:** 25 PDE on a 5×5 (Bi_T, S_chi) grid at N=301
- **Wall-clock:** 22.5 min (24 workers, t_end=400)
- **Scripts:** `scripts/iv_c_phaseB_period.py`, `scripts/iv_c_phaseB_check.py`
- **Outputs:**
  - `data/iv_c/phaseB/period_raw.json`     — per-sim records (incremental)
  - `data/iv_c/phaseB/period_scan.npz`     — reshaped 5×5 grids
  - `data/iv_c/phaseB/phaseB_check.npz`    — decision summary
  - `Figure/pub/iv_c_period_collapse.{pdf,png}` — collapse figure

### Grid choice

Plan-suggested grid `Bi_T ∈ {0.04..0.26}, S_chi ∈ {0.5..1.6}` overlaps
with steady-front and steady-cold regions per the fig4 phase diagram
(esp. Bi_T ≥ 0.16 and S_chi ≤ 0.7). Retuned grid:

- `Bi_T_vals = [0.04, 0.05, 0.07, 0.09, 0.12]`  — small Bi_T to access asymptote
- `S_chi_vals = [0.9, 1.0, 1.1, 1.3, 1.5]`  — robust LF strip

### Surprise finding: Bi_T = 0.04 fails to oscillate at long t

All 5 `(Bi_T=0.04, S_chi)` points settle to **hot-runaway SS** (J ≈ 0.157,
the collapsed-gel asymptote φ → 1) over `t ∈ (200, 400)`, even at
N=301. The fig4 phase diagram (also N=301 but t_end=200) showed these
as LF.

The discrepancy is almost certainly **t_end**: fig4 used t_end=200 and
caught the initial ignition transient as oscillation, but extending to
t_end=400 reveals these points eventually drift into hot-runaway. This
is consistent with the relaxation-oscillator picture — at very small
Bi_T the cooling phase becomes too slow to recover from runaway
ignition, so the limit cycle ceases to exist as an attractor.

7 NOT_OSC points total: all 5 Bi_T=0.04 + (Bi_T=0.05, S_chi=1.3) +
(Bi_T=0.05, S_chi=1.5).

This narrows the usable Bi_T range to [0.05, 0.12] for low S_chi and
[0.07, 0.12] for high S_chi — still 4 / 3 points per S_chi line, enough
to verify the scaling.

### Result table — oscillating points

| S_chi | ln(θ_up/θ_lo) | smallest Bi_T | T·Bi_T at smallest | rel err |
|------:|--------------:|--------------:|-------------------:|--------:|
| 0.90  | 1.3276        | 0.05          | 1.3937             | 4.98%   |
| 1.00  | 1.3276        | 0.05          | 1.3832             | 4.19%   |
| 1.10  | 1.3276        | 0.05          | 1.3846             | 4.29%   |
| 1.30  | 1.3276        | 0.07          | 1.4931             | 12.47%  |
| 1.50  | 1.3276        | 0.07          | 1.5063             | 13.46%  |

- max rel err: **13.5%** (PASS threshold 30%)
- mean rel err: 7.9%
- ln(θ_up/θ_lo) ≈ 1.33 is essentially **constant across S_chi** (because
  θ_up and θ_lo both scale as 1/S_chi at fixed material parameters);
  the period collapse is to a single number.

### Trend check

For each S_chi, T·Bi_T monotonically decreases toward 1.33 as Bi_T → 0,
**always approaching from above**, consistent with the analytic correction
`T·Bi_T = ln(θ_up/θ_lo) + Bi_T·e^(-Γ_A·θ_lo)/(Da·J*·(1−φ*)^m_act·Γ_A)`
which is positive for Bi_T > 0.

The departure is steeper at low S_chi (S_chi=0.9: Bi_T=0.12 → T·Bi_T=2.56,
86% above asymptote) than at high S_chi (S_chi=1.5: Bi_T=0.12 → 1.93, 45%
above), consistent with the Frank-Kamenetskii correction prefactor.

### Decision

PASS — proceed to **Phase C (onset threshold)** when user confirms.

---
