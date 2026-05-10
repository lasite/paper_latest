#!/usr/bin/env python3
"""
scan_hysteresis_C.py — D2: IC-induced bistability sweep at the C-cell.

At Bi_T=0.059, S_chi=1.80 (representative C-cell), trace late-time
attractor selection as a function of one experimentally-controllable
initial condition (default: theta_0, an initial thermal pulse) for two
seed types:
  * "cold seed":  J_0 = 1.30 (near homogeneous swollen state)
  * "hot seed":   J_0 = 0.20 (pre-collapsed, hot-runaway-like)
with u_0 = 0.5 fixed (matching the existing 5x5 IC ensemble).

Purpose: quantify the IC-induced hysteresis prediction stated in
§IV.D.  The cold-seed branch tracks the LCST-front cycle for theta_0
below the basin threshold and jumps to hot-runaway above; the hot
seed reveals where the runaway basin extends.  The gap between the
two curves is the bistable region of the IC parameter.

Output cache: data/fig6/hysteresis_C_<axis>.npz containing:
  axis_name, axis_values  : control-parameter axis (theta_0 by default)
  J0_cold, J0_hot         : seed J_0 values
  u0, J_perturb_amp       : auxiliary IC scalars
  cell_Bi_T, cell_S_chi   : cell parameters
  N, t_end, n_save        : grid/time settings
  t                       : downsampled time grid (n_save,)
  Jm_cold, thm_cold       : (n_axis, n_save) bulk-mean trajectories from cold seed
  Jm_hot,  thm_hot        : (n_axis, n_save) from hot seed
  Jm_term_cold, thm_term_cold : terminal means (last 5%)
  Jm_term_hot,  thm_term_hot
  success_cold, success_hot   : boolean per axis index per seed

Re-render: scripts/make_hysteresis.py reads this cache and is fast
(< 5 s); regenerate the cache only when parameters change.

Usage:
  python scan_hysteresis_C.py                 # run default theta_0 sweep
  python scan_hysteresis_C.py --axis u0       # sweep u_0 instead
  python scan_hysteresis_C.py --n_axis 41     # finer axis sampling
  FIG4_WORKERS=12 python scan_hysteresis_C.py # override worker count
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin BLAS thread counts BEFORE importing numpy (worker threads are spawned
# per-process; nested threading is wasted work).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# --- C-cell parameters (matching §IV.D representative point) ---
CELL_BI_T  = 0.059
CELL_S_CHI = 1.80

# --- Seed configuration ---
J0_COLD = 1.30   # cold-swollen seed
J0_HOT  = 0.20   # pre-collapsed (hot-runaway-like) seed
U0_FIXED = 0.5   # matches existing 5x5 IC ensemble

# --- Numerical grid ---
# t_end shorter than the 5x5 ensemble (200) because we only need the late-time
# attractor classification (cycle vs runaway), not multi-cycle accuracy.
# N=121 to match the reference 5x5 ensemble - lower N appears to leak the
# cold-seed trajectory to runaway numerically at the C-cell.
N_GRID = 121
T_END  = 150.0
N_SAVE = 600

DATA_DIR = _HERE.parent / "data" / "fig6"


# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────

def _worker(task):
    """Run one PDE simulation; return projected (Jm(t), thm(t)) and term."""
    (axis_name, axis_value, seed_label, J0, theta0, u0, idx) = task

    from scan_optimized import (
        Params, finalize_params, rhs_mol_logJ, make_jac_sparsity,
        make_sparse_fd_jac, cell_centers, _LOG_J_MAX,
    )
    from fig2_data import WORKING_POINT
    from scipy.integrate import solve_ivp

    p_dict = dict(WORKING_POINT)
    # Use the same tolerances as the reference 5x5 IC ensemble (rtol=1e-6,
    # atol=1e-8); slightly relaxed max_step retains performance on the
    # stiff high-theta_0 cases without compromising the basin classification.
    p_dict.update(Bi_T=CELL_BI_T, S_chi=CELL_S_CHI,
                  N=N_GRID, t_end=T_END, n_save=N_SAVE,
                  max_step=1.0)
    p = Params(**p_dict)
    p = finalize_params(p)
    n3 = 3 * p.N
    n  = p.N

    x = cell_centers(n)
    log_J_min = np.log(p.phi_p0 * 1.02)
    Jvec  = np.maximum(np.full(n, J0)     + 1e-3 * np.cos(np.pi * x),
                       np.exp(log_J_min) + 1e-6)
    uvec  = np.maximum(np.full(n, u0)     + 1e-3 * np.cos(np.pi * x),
                       p.u_floor)
    thvec = np.full(n, theta0)            + 1e-3 * np.cos(np.pi * x)
    Wvec  = Jvec * uvec
    y0    = np.concatenate([np.log(Jvec), Wvec, thvec])

    rhs_fn = lambda t, y: rhs_mol_logJ(t, y, p)
    S = make_jac_sparsity(p.N)
    jac_sparse, _ = make_sparse_fd_jac(rhs_fn, S, n3)

    t0_wall = time.perf_counter()
    sol = solve_ivp(
        fun=rhs_fn, jac=jac_sparse,
        t_span=(0.0, p.t_end), y0=y0,
        t_eval=np.linspace(0, p.t_end, p.n_save),
        method=p.method, rtol=p.rtol, atol=p.atol, max_step=p.max_step,
    )
    dt_wall = time.perf_counter() - t0_wall

    if sol.success:
        J     = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
        theta = sol.y[2*n:]
        Jm    = J.mean(axis=0)
        thm   = theta.mean(axis=0)
        n5    = max(int(0.05 * len(sol.t)), 5)
        Jm_term  = float(Jm[-n5:].mean())
        thm_term = float(thm[-n5:].mean())
    else:
        Jm = thm = np.full(p.n_save, np.nan)
        Jm_term = thm_term = np.nan

    return dict(
        axis_name=axis_name, axis_value=axis_value,
        seed=seed_label, idx=idx, J0=J0, theta0=theta0, u0=u0,
        t=sol.t, Jm=Jm, thm=thm,
        Jm_term=Jm_term, thm_term=thm_term,
        success=bool(sol.success), dt=dt_wall,
    )


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def build_tasks(axis_name: str, axis_values: np.ndarray):
    tasks = []
    for k, v in enumerate(axis_values):
        v = float(v)
        for seed_label, J0_seed in [("cold", J0_COLD), ("hot", J0_HOT)]:
            if axis_name == "theta0":
                tasks.append((axis_name, v, seed_label,
                              J0_seed, v, U0_FIXED, k))
            elif axis_name == "u0":
                tasks.append((axis_name, v, seed_label,
                              J0_seed, 0.0, v, k))
            else:
                raise ValueError(f"unknown axis: {axis_name}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["theta0", "u0"], default="theta0",
                    help="which IC variable to sweep")
    ap.add_argument("--axis_min", type=float, default=None)
    ap.add_argument("--axis_max", type=float, default=None)
    ap.add_argument("--n_axis", type=int, default=25)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("FIG4_WORKERS", 24)))
    ap.add_argument("--force", action="store_true",
                    help="recompute even if cache exists")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"hysteresis_C_{args.axis}.npz"

    if cache_path.exists() and not args.force:
        print(f"  Cache already present: {cache_path}")
        print(f"  Re-run with --force to overwrite.")
        return

    if args.axis == "theta0":
        amin = 0.0   if args.axis_min is None else args.axis_min
        amax = 6.0   if args.axis_max is None else args.axis_max
    else:  # u0
        amin = 0.05  if args.axis_min is None else args.axis_min
        amax = 1.5   if args.axis_max is None else args.axis_max

    # The cold-seed basin transition happens for theta0 in (0, 4) at the C-cell.
    # Sample more densely in [0, 3] to resolve the jump, with sparser
    # coverage of the high-theta plateau.
    if args.axis == "theta0" and args.axis_min is None:
        axis_values = np.unique(np.concatenate([
            np.linspace(0.0, 3.0, max(args.n_axis - 5, 1)),
            np.linspace(3.5, amax, 5),
        ]))
    else:
        axis_values = np.linspace(amin, amax, args.n_axis)
    tasks = build_tasks(args.axis, axis_values)

    print(f"=== D2 hysteresis sweep at C-cell ===", flush=True)
    print(f"  Bi_T={CELL_BI_T}, S_chi={CELL_S_CHI}, u0={U0_FIXED}", flush=True)
    print(f"  axis = {args.axis} in [{amin:.3f}, {amax:.3f}], n={args.n_axis}",
          flush=True)
    print(f"  J0_cold={J0_COLD}, J0_hot={J0_HOT}", flush=True)
    print(f"  Total simulations: {len(tasks)}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(flush=True)

    n_axis = len(axis_values)
    Jm_cold     = np.full((n_axis, N_SAVE), np.nan)
    thm_cold    = np.full((n_axis, N_SAVE), np.nan)
    Jm_hot      = np.full((n_axis, N_SAVE), np.nan)
    thm_hot     = np.full((n_axis, N_SAVE), np.nan)
    Jm_term_c   = np.full(n_axis, np.nan)
    thm_term_c  = np.full(n_axis, np.nan)
    Jm_term_h   = np.full(n_axis, np.nan)
    thm_term_h  = np.full(n_axis, np.nan)
    succ_c      = np.zeros(n_axis, dtype=bool)
    succ_h      = np.zeros(n_axis, dtype=bool)
    t_axis      = None

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, task): task for task in tasks}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            k = r["idx"]
            if r["seed"] == "cold":
                Jm_cold[k, :]   = r["Jm"]
                thm_cold[k, :]  = r["thm"]
                Jm_term_c[k]    = r["Jm_term"]
                thm_term_c[k]   = r["thm_term"]
                succ_c[k]       = r["success"]
            else:
                Jm_hot[k, :]    = r["Jm"]
                thm_hot[k, :]   = r["thm"]
                Jm_term_h[k]    = r["Jm_term"]
                thm_term_h[k]   = r["thm_term"]
                succ_h[k]       = r["success"]
            if t_axis is None and r["success"]:
                t_axis = r["t"]
            done += 1
            print(f"  [{done:>3}/{len(tasks)}] axis={r['axis_value']:.3f} "
                  f"seed={r['seed']:<4s}  term=(J={r['Jm_term']:.2f}, "
                  f"th={r['thm_term']:.2f})  {r['dt']:.0f}s  "
                  f"success={r['success']}", flush=True)

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.0f}s = {total/60:.1f} min")

    np.savez_compressed(
        cache_path,
        axis_name=args.axis,
        axis_values=axis_values,
        J0_cold=J0_COLD, J0_hot=J0_HOT, u0=U0_FIXED,
        cell_Bi_T=CELL_BI_T, cell_S_chi=CELL_S_CHI,
        N=N_GRID, t_end=T_END, n_save=N_SAVE,
        t=t_axis if t_axis is not None else np.linspace(0, T_END, N_SAVE),
        Jm_cold=Jm_cold,  thm_cold=thm_cold,
        Jm_hot=Jm_hot,    thm_hot=thm_hot,
        Jm_term_cold=Jm_term_c,  thm_term_cold=thm_term_c,
        Jm_term_hot=Jm_term_h,   thm_term_hot=thm_term_h,
        success_cold=succ_c, success_hot=succ_h,
    )
    print(f"  Saved: {cache_path}")


if __name__ == "__main__":
    main()
