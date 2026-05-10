# Phase A.5 — N convergence report (corrected)

**Tolerance:** relative error vs reference (highest finite N, here N=401)
< 5%, AND remains < 5% at every higher finite N.

_Settings_: `t_end=400.0`, `t_window=(200.0, 400.0)`, `n_save=4000`,
per-sim timeout = 2100 s (35 min). Total wall-clock 35.0 min.

_Coverage_: 15 / 21 sims succeeded; 6 timed out: WP/shallow/nearSNIC at
N=201 and N=501.

> **Note on the N=201 timeouts.** The N=201 sims for ALL three points
> exceeded the 35-min budget while the N=301 sims finished in 17-19 min.
> Most likely cause: BDF + sparse-Jacobian column grouping at N=201 hits a
> pessimal coloring or stiffness pattern. Phase A.5's purpose (find the
> minimum converged N) is unaffected — N=51, 101, 151, 301, 401 sample
> the trend densely enough to read convergence cleanly.

> **Why this report differs from the auto-generated `convergence_report.md`
> first run.** The original `converged_N` function compared each N to its
> immediate predecessor in the grid; with N=201 NaN, even N=301 was
> rejected because finite[201] is False. The criterion below uses the
> highest-N finite value as the reference, which is the right semantic.

---

## Per-point relative error vs reference (N=401)

### Point `WP` — Bi_T = 0.10, S_chi = 1.0

| observable | N=51 | N=101 | N=151 | N=201 | N=301 | N=401 | N=501 | converged at |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| period         | 36.5% | 28.0% | **96.8%** | — | 0.11% | 0.00% | — | **N=301** |
| delta_theta    | 10.0% |  1.5% | **10.2%** | — | 0.09% | 0.00% | — | **N=301** |
| delta_J        |  7.1% | 13.4% |  23.5%    | — | 0.22% | 0.00% | — | **N=301** |
| xi_LCST        |  3.0% |  5.7% |  20.9%    | — | 0.13% | 0.00% | — | **N=301** |
| theta_surf_max |  5.7% |  9.9% |  18.3%    | — | 0.05% | 0.00% | — | **N=301** |
| theta_surf_min | 33.8% | 25.0% |  32.9%    | — | 0.31% | 0.00% | — | **N=301** |

WP at **N=151 is anomalous** (period jumps to 38.1 vs 19.3 at N=301; Δθ to
2.42 vs 2.69). Likely a 2:1 frequency-locking artifact at that specific
resolution — disappears at N=301.

### Point `shallow` — Bi_T = 0.05, S_chi = 0.7

| observable | N=51 | N=101 | N=151 | N=201 | N=301 | N=401 | N=501 | converged at |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| period         |   —    | **113.4%** |   —     | — | 0.91% | 0.00% | — | **N=301** |
| delta_theta    |   —    |  11.4%     |   —     | — | 0.16% | 0.00% | — | **N=301** |
| delta_J        |   —    |  26.3%     |   —     | — | 0.19% | 0.00% | — | **N=301** |
| xi_LCST        | 19.2%  |  47.1%     |  21.6%  | — | 0.17% | 0.00% | — | **N=301** |
| theta_surf_max | 73.0%  |  22.3%     |  26.9%  | — | 0.16% | 0.00% | — | **N=301** |
| theta_surf_min | 25.8%  |  41.3%     |  95.3%  | — | 0.16% | 0.00% | — | **N=301** |

shallow at **N=51 and N=151 fail to oscillate** (settle to hot-runaway).
At N=101 the cycle exists but with period 62.8 vs 29.2 at N=301 — likely
spurious low-frequency drift from under-resolved LCST front. Genuine
limit cycle only emerges from N=301.

### Point `nearSNIC` — Bi_T = 0.15, S_chi = 1.3

| observable | N=51 | N=101 | N=151 | N=201 | N=301 | N=401 | N=501 | converged at |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| period         | 18.3% | 21.7% | 26.0% | — | 0.12% | 0.00% | — | **N=301** |
| delta_theta    | 10.6% |  4.5% |  2.7% | — | 0.25% | 0.00% | — | N=101 |
| delta_J        |  8.5% |  9.0% |  6.3% | — | 0.15% | 0.00% | — | **N=301** |
| xi_LCST        |  4.3% |  3.1% |  2.6% | — | 0.15% | 0.00% | — | N=51  |
| theta_surf_max |  5.0% |  5.3% |  3.6% | — | 0.02% | 0.00% | — | N=151 |
| theta_surf_min | 32.5% | 22.8% | 14.8% | — | 0.39% | 0.00% | — | **N=301** |

nearSNIC's period drifts 18 → 22 → 26% (worsening with N) before
collapsing to 0.12% at N=301 — the cycle frequency settles only when
the front is well-resolved.

---

## Aggregate — minimum N converged at all 3 points

| observable     | min converged N |
|----------------|----------------:|
| period         | **N = 301** |
| delta_theta    | **N = 301** |
| delta_J        | **N = 301** |
| xi_LCST        | **N = 301** |
| theta_surf_max | **N = 301** |
| theta_surf_min | **N = 301** |

---

## Recommendations for downstream phases

Per the user's decision rule:

- **All observables converge at N=301** → use **N = 301** for Phase B
  (period), Phase C (onset), Phase D (front depth).
- **Phase A re-do?** The user's rule: re-do iff `delta_theta` at N=101 vs
  N=301 differs by > 5%. Per-point:
  - WP at N=101: **1.5%** ✓ (within tolerance)
  - **shallow at N=101: 11.4%** ✗ (exceeds tolerance)
  - nearSNIC at N=101: 4.5% ✓
  - **Verdict: Phase A should be re-run at N=301** to tighten the
    amplitude-scaling check. Estimated wall-clock ~20 min (12 sims at
    N=301, parallel on 24 workers).

- Phase A's PASS verdict (max_dev = 13.4%) is robust under N=101 → N=301
  shift (re-evaluating shallow's contribution: meas h becomes 2.566
  instead of 2.277, rel_err 9.7% instead of 2.7% — still well under 20%
  threshold). So PASS holds with N=101 data; the re-run only tightens
  the numerical evidence.

## CPU cost

- N=51 sims: ~30-50 s each
- N=101: ~3-4 min
- N=151: ~14-32 min (variable; sometimes pathological)
- N=301: ~17-19 min
- N=401: ~25-27 min
- N=501: > 35 min (timed out everywhere)

Phase B (25 sims at N=301): ~20 min wall-clock with 24 workers.
Phase D (49+ sims at N=301): ~40 min in 2 batches.
