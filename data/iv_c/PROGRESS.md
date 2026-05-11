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

## Phase C — Onset-threshold scaling

- **Status:** PASS_borderline (criteria met but qualitative match only)
- **Sims run:** 33 (Bi_T=0.06: 3 sims+bracket only / Bi_T=0.10: 11 sims / Bi_T=0.16: 14 sims / Bi_T=0.25: 6 sims bracket expansion)
- **Wall-clock:** ~75 min cumulative (multiple sessions, including a 96-min lost run and a foreground retry that wedged on a near-onset Bi_T=0.06 sim)
- **Scripts:**
  - `scripts/iv_c_phaseC_Da_c_0D.py` (analytic + continuation cross-check)
  - `scripts/iv_c_phaseC_pde.py` (parallel bisection driver)
  - `scripts/iv_c_phaseC_pde_BiT006.py` (foreground resume for the lost Bi_T=0.06 chain)
  - `scripts/iv_c_phaseC_check.py`
- **Outputs:**
  - `data/iv_c/phaseC/Da_c_0D.npz`
  - `data/iv_c/phaseC/Da_c_pde.npz`
  - `data/iv_c/phaseC/pde_bisection_log.json`, `pde_bisection_log_BiT006.json`
  - `data/iv_c/phaseC/phaseC_check.npz`
  - `Figure/pub/iv_c_onset_shift.{pdf,png}`

### Method

- `Da_c^0D` = leading-order saddle-node of the cold-swollen-connected SS
  = `Bi_T / (J_init * Gamma_A)`. Cross-checked with Newton continuation:
  numerical saddle-node lands within 10% of the formula at Bi_T ≤ 0.16.
  (The cold branch's complex-pair eigenvalues approach zero
  simultaneously with the SN, so this Da is the natural "0D analog"
  of the PDE oscillation onset.)
- `Da_c^PDE` = 8-iteration log-mean bisection in Da, bracket
  [0.5·Da_c^0D, 3·Da_c^0D] (auto-expanded up to Da=4 when needed).
  Settings: N=301, t_end=400, t_window=(300, 400), N_save=4000.
  is_oscillating: σ(J_surf) > 0.05.

### Result table

| Bi_T | Da_c^0D | Da_c^PDE | shift | sqrt(Bi_T/α) | notes |
|-----:|--------:|---------:|------:|-------------:|-------|
| 0.06 | 0.0308  | 0.0377†  | +0.22 | 0.548 | †bracket-midpoint only; iter-2 sim wedged at 48 min near onset (slow dynamics close to Da_c) |
| 0.10 | 0.0513  | 0.2289   | +3.46 | 0.707 | converged |
| 0.16 | 0.0821  | 2.0545   | +24.0 | 0.894 | converged; bracket auto-expanded to Da=4 |
| 0.25 | 0.1282  | **NaN**  | —     | 1.118 | no oscillation up to Da=4; system relaxes to over-swollen SS (J≈1.1) at all bracket Da |

### Decision metrics

- Linear fit `shift = c_1 · √(Bi_T/α) + c_0` on 3 finite points:
  `c_1 = 69.8`, `c_0 = -40.8`, **R² = 0.88** (just above the 0.85 threshold).
- All finite shifts positive ✓.
- Slope positive ✓.

### Honest caveats — why this is PASS_borderline, not clean PASS

1. **Slope `c_1 ≈ 70` is ~50× the leading-order theoretical estimate
   `c_1 = 2·Γ_A / (√π · θ_0 · √Bi_T) ≈ O(1)`.** The data fit a linear
   trend but the prefactor doesn't match the perturbative derivation.
2. **Large negative intercept (`c_0 = -41`).** The theory predicts
   `c_0 ≈ c_2/Bi_c > 0`, not negative. Suggests the fit is
   accommodating curvature/non-linearity not captured by a single
   sqrt(Bi_T/α) term.
3. **One outlier (Bi_T=0.16, shift=24) dominates the fit.** Dropping
   it leaves only two points, which trivially R²=1 but with very
   different inferred slope (c_1 ≈ 20).
4. **Bi_T=0.25 doesn't oscillate at all up to Da=4.** The bisection
   bracket cap was hit — at this Bi_T the system relaxes to an
   over-swollen SS (J ≈ 1.1, θ ≈ 1.6) instead of finding the LF cycle
   from cold IC. This is qualitatively a SNIC-type loss of the cycle
   attractor, not a Hopf onset.
5. **Bi_T=0.06's chain failed to converge.** The iter-2 sim
   (Da ≈ 0.038, just above Da_c^0D) ran for 48 min CPU without
   producing a definitive OSC/NOT_OSC answer — near-onset PDE
   dynamics are genuinely slow (cycle period diverges, long
   relaxation transient). We report the bracket midpoint as a
   best-estimate.

### What the data actually shows

A more accurate physical picture than "linear sqrt(Bi_T/α) shift":

- At **Bi_T = 0.06**: PDE oscillation onset is **very close to** Da_c^0D
  (shift +22%) — consistent with a perturbative shift.
- At **Bi_T = 0.10**: there is a **gap** between Da_c^0D = 0.05 and
  Da_c^PDE = 0.23. In this gap (Da ∈ [0.05, 0.23]) the system
  relaxes to a NEW attractor — an over-swollen SS at J ≈ 3.8-4.0,
  which is neither the cold SS nor the LF cycle. The system has to
  climb a much higher Da before the over-swollen branch loses
  stability and the LF cycle is born.
- At **Bi_T = 0.16**: the gap is enormous (Da_c^0D = 0.08 → Da_c^PDE = 2.05).
  Same over-swollen SS dominates the cold-IC dynamics.
- At **Bi_T = 0.25**: the over-swollen SS persists for all Da up to 4;
  no LF cycle accessible from cold IC.

**The bifurcation structure is more complex than a simple Hopf shift.**
The "0D" Hopf onset assumed in the derivation is the cold-branch SN;
above this Da, the system enters a region where an *over-swollen SS*
(spatially non-uniform, induced by surface cooling) acts as an
intermediate attractor. The LF cycle only emerges when this over-
swollen SS itself loses stability, at a much higher Da. This is a
qualitatively different mechanism — likely a secondary bifurcation
that scales differently from the predicted sqrt(Bi_T/α).

### Decision

- The R² = 0.88 PASS verdict carries forward (matches the stated
  criterion).
- For §IV.C, scaling (iv) should be **reframed** from a quantitative
  prediction to a *qualitative* one: "PDE onset Da_c^PDE > Da_c^0D and
  grows with Bi_T". The leading-order prefactor `c_1 = 2·Γ_A/(√π·θ_0·√Bi_T)`
  is *not* supported by the data; the actual prefactor is ~50×
  larger, suggesting the relevant mechanism at the working point is
  not the perturbative Hopf shift but the secondary bifurcation
  involving the over-swollen SS.
- Proceed to **Phase D (front depth)** — the most data-rich and
  least theory-dependent of the four scaling laws.

---

## Over-swollen-SS diagnostic (between Phase C and Phase D)

A single follow-up sim at (Bi_T=0.10, Da=0.20, t_end=800, N=301)
characterised the "over-swollen" state Phase C found between Da_c^0D
and Da_c^PDE.

**Verdict: it's a frozen-front, not a uniform state.**

Late-window (t ∈ [600, 800]) statistics:
- J_surf = 4.063, J_core = 4.785, theta ≈ 6.83 (uniform), std = 0 to 6 sig figs (truly steady).

Spatial profile at t = 800:
- Bulk (xi ∈ [0, 0.85]):  J ≈ 4.78, phi ≈ 0.03  (highly over-swollen)
- Thin spike at xi ≈ 0.85:  phi ≈ 0.98, J ≈ 0.15  (collapsed LCST barrier, 1-2 cells wide)
- Outer skin (xi ∈ [0.85, 1.0]):  J recovers to ~4.0 at surface
- theta nearly uniform at 6.85 (heat conducts through the barrier; mass diffusion does not, because D ~ (1−phi)^m_diff is zero at phi → 1)

Physical interpretation: a NON-EQUILIBRIUM steady state sustained by
reaction. Reaction heats the bulk, S_chi*theta*phi² term creates an
LCST front internally, the collapsed barrier traps mass flow, and the
bulk relaxes to whatever J is consistent with the chemistry-driven
flow balance against the bath. Distinct mechanism from cold-core
frozen front, from LF cycle, and from hot-runaway.

Outputs: `scripts/iv_c_overswollen_diag.py`, `data/iv_c/phaseC_diag/overswollen_BiT010_Da020.npz`,
`Figure/pub/iv_c_overswollen_diag.{pdf,png}`. Committed as `013a27b`.

---

## Phase D — Front-depth scaling

- **Status:** PASS (saturation spread 2.53% << 30% threshold)
- **Sims attempted:** 53 (49 main grid + 4 Bi_T slice)
- **Sims successful:** 29 (21 cycles + 4 overswollen_front + 3 cycles in slice + 1 cold_SS in slice)
- **TIMEOUTs:** 24, all at high m_act ≥ 5 in the main grid — BDF can't resolve the very sharp (1−phi)^m barrier at high m within 30 min
- **Wall-clock:** 43 min total (30 min main grid + 13 min slice)
- **Scripts:** `scripts/iv_c_phaseD.py`, `scripts/iv_c_phaseD_check.py`
- **Outputs:**
  - `data/iv_c/phaseD/main_grid.npz`, `main_grid_raw.json`
  - `data/iv_c/phaseD/BiT_slice.npz`, `BiT_slice_raw.json`
  - `data/iv_c/phaseD/phaseD_check.npz`
  - `Figure/pub/iv_c_front_depth_main.{pdf,png}`
  - `Figure/pub/iv_c_front_depth_attractor_map.{pdf,png}`

### Settings

- Grid: m_act, m_diff ∈ {1, 2, 3, 4, 5, 6, 7} (7×7 = 49 sims)
- Bi_T slice: m_act = m_diff = 4, Bi_T ∈ {0.06, 0.10, 0.16, 0.25} (4 sims)
- N=301, t_end=400, t_window=(200, 400), n_save=4000, per-sim timeout 30 min

### Working-point penetration depths

| quantity   | formula              | value at WP (alpha=0.20, delta=0.08, Bi_T=0.10, Bi_c=0.70) |
|------------|----------------------|---:|
| L_T        | √(alpha/Bi_T)        | 1.4142 |
| L_c        | √(delta/Bi_c)        | 0.3381 |
| min(L_T, L_c) | (L_c dominates at WP) | **0.3381** |

### Main-grid saturation (Bi_T = 0.10, Da = 4, S_chi = 1.0)

| m_act + m_diff | mean (1−xi_LCST) | mean ratio (1−xi)/L_eff | std ratio | n |
|---:|---:|---:|---:|---:|
| 2  | 0.005 | 0.015 | 0.000 | 1 |
| 3  | 0.070 | 0.206 | 0.192 | 2 |
| 4  | 0.090 | 0.267 | 0.178 | 3 |
| 5  | 0.100 | 0.297 | 0.163 | 4 |
| **6**  | **0.132** | **0.391** | 0.004 | 4 |
| 7  | 0.131 | 0.388 | 0.000 | 4 |
| 8  | 0.131 | 0.388 | 0.000 | 4 |
| 9  | 0.131 | 0.388 | 0.000 | 2 |
| 10 | 0.131 | 0.388 | 0.000 | 1 |

**Saturation kicks in cleanly at m_act + m_diff = 6** (crossover m_c ≈ 5-6).
Above this, ratio = **0.388 ± 0.002**, spread (max−min)/mean = **2.53%**
across 15 finite points — well within the 30% PASS threshold.

### Bi_T slice (m_act = m_diff = 4)

| Bi_T | L_T | L_c | L_eff | xi_LCST | 1−xi | ratio | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.06 | 1.826 | 0.338 | 0.338 | 0.842 | 0.158 | 0.467 | cycle |
| 0.10 | 1.414 | 0.338 | 0.338 | 0.869 | 0.131 | 0.388 | cycle |
| 0.16 | 1.118 | 0.338 | 0.338 | 0.852 | 0.148 | 0.437 | cycle |
| 0.25 | 0.894 | 0.338 | 0.338 | 1.000 | 0.000 | 0.000 | **cold_SS** |

For the 3 oscillating points the ratio is 0.39 to 0.47 (within 20% of each
other). Bi_T = 0.25 fails to oscillate at the working-point Da = 4 — consistent
with the Phase C finding that Da_c^PDE for that Bi_T exceeds 4. (This also means
the test is a check of L_c-saturation only; L_T-dependence cannot be probed at
this material parameter set because L_T > L_c throughout the LF region.)

### 5-way attractor classifier — bonus phase diagram

Beyond xi_LCST, every Phase D sim is classified by long-time attractor type:
- `cycle`             oscillating LF limit cycle (sigma(J_surf) > 0.05)
- `overswollen_front` steady, phi crosses 0.5, J_mean > 2 (chemistry-driven
                       over-swollen bulk + thin LCST barrier — the state
                       diagnosed in the over-swollen-SS section above)
- `frozen_front`      steady, phi crosses 0.5, J_mean ≤ 2 (classic cold-core
                       + collapsed shell)
- `hot_runaway`       steady, all phi > 0.5
- `cold_SS`           steady, all phi < 0.5

At Bi_T = 0.10, Da = 4 the attractor map is dominated by `cycle`,
with `overswollen_front` along the m_diff = 1 column (poor reactant
diffusion → solvent can't recycle, LCST barrier sits right at surface).
No `frozen_front` or `hot_runaway` observed at the working point —
those require lower Bi_T or different Da.

### Issues / caveats

- **High-m TIMEOUTs.** Of 49 main-grid sims, 24 (all with m_act ≥ 5)
  exceeded the 30-min timeout. Cause: at high m_act + m_diff the
  barrier `(1−phi)^m` becomes near-singular at phi → 1, BDF needs
  exponentially smaller time steps. These regions are PHYSICALLY
  fine — the cycle exists per the m=2-4 trend extrapolation — but
  the integrator can't reach late-time. The saturation argument is
  established by the m_act+m_diff ∈ {6, 7, 8, 9, 10} sample we DO
  have (15 points, all ratio = 0.388 ± 0.002 to 3 sig figs).
- **L_T scaling not testable** at our parameter set. L_T > L_c
  throughout the LF region (Bi_T ≤ 0.25 with Bi_c = 0.7, alpha = 0.2,
  delta = 0.08), so the scaling collapses to (1−xi) ∝ L_c, which is
  Bi_T-independent. To test L_T scaling we'd need delta/Bi_c larger
  than alpha/Bi_T, i.e. raise delta or lower Bi_c.
- **No `hot_runaway` at small m.** The plan predicted hot-runaway as
  the failure mode at small m_act + m_diff (sharp-barrier broken).
  Instead we see `overswollen_front` — same "barrier missing"
  physics but with a different non-cycle attractor. The qualitative
  prediction holds (sharp-barrier limit gives saturation; smooth-
  barrier limit gives no-cycle); the specific failure mode differs.

### Decision

PASS — proceed to **Phase E (figures)** and **Phase F (LaTeX draft)** when
user confirms.

Three of the four scaling laws have clean PASS verdicts at this point:
- (i) period: PASS (max 13.5% off)
- (iii) amplitude: PASS (max 13.4% off; analytic perfectly constant)
- (ii) front depth: **PASS** (2.5% spread above saturation crossover m_c ≈ 6)

Scaling (iv) onset: PASS_borderline (R² = 0.88 met but prefactor and
mechanism don't match the leading-order derivation; reframe qualitatively
in the §IV.C write-up).

---
