#!/usr/bin/env python3
"""
scan_basin_C_dense.py — D1: refine the basin boundary at the C-cell.

Re-run the (J_0, theta_0) IC sweep at Bi_T=0.059, S_chi=1.80 with a
denser grid (default 15x15 = 225 sims) than the existing 5x5 ensemble,
producing a clean basin-of-attraction map for Fig.6(c).

Output: data/fig6/basin_C_dense.npz
  J0_vals, theta0_vals : axis grids
  Jm_term, thm_term    : terminal mean of (J, theta) per cell
  attractor            : 0=cycle, 1=runaway, -1=failed/ambiguous
  success              : boolean per cell

Cost: ~225 sims at ~150s each / 24 workers ≈ 25 min (PDE-bound).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

CELL_BI_T  = 0.059
CELL_S_CHI = 1.80

N_GRID = 121
T_END  = 200.0
N_SAVE = 1000
U0_FIXED = 0.5

DATA_DIR = _HERE.parent / "data" / "fig6"


def _worker(task):
    j_idx, i_idx, J0, theta0, u0 = task
    from scan_optimized import (
        Params, finalize_params, rhs_mol_logJ, make_jac_sparsity,
        make_sparse_fd_jac, cell_centers, _LOG_J_MAX,
    )
    from fig2_data import WORKING_POINT
    from scipy.integrate import solve_ivp

    p_dict = dict(WORKING_POINT)
    p_dict.update(Bi_T=CELL_BI_T, S_chi=CELL_S_CHI,
                  N=N_GRID, t_end=T_END, n_save=N_SAVE)
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

    t0_w = time.perf_counter()
    sol = solve_ivp(
        fun=rhs_fn, jac=jac_sparse,
        t_span=(0.0, p.t_end), y0=y0,
        t_eval=np.linspace(0, p.t_end, p.n_save),
        method=p.method, rtol=p.rtol, atol=p.atol, max_step=p.max_step,
    )
    dt_w = time.perf_counter() - t0_w

    if sol.success:
        J     = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
        theta = sol.y[2*n:]
        Jm    = J.mean(axis=0)
        thm   = theta.mean(axis=0)
        n5    = max(int(0.05 * len(sol.t)), 5)
        Jm_term  = float(Jm[-n5:].mean())
        thm_term = float(thm[-n5:].mean())
        attractor = 0 if Jm_term < 3.0 else 1   # cycle vs runaway
    else:
        Jm_term = thm_term = np.nan
        attractor = -1

    return dict(j=j_idx, i=i_idx, J0=J0, theta0=theta0,
                Jm_term=Jm_term, thm_term=thm_term,
                attractor=attractor, success=bool(sol.success), dt=dt_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_J", type=int, default=15)
    ap.add_argument("--n_theta", type=int, default=15)
    ap.add_argument("--J_min", type=float, default=0.20)
    ap.add_argument("--J_max", type=float, default=1.30)
    ap.add_argument("--theta_min", type=float, default=0.0)
    ap.add_argument("--theta_max", type=float, default=10.0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("FIG4_WORKERS", 24)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "basin_C_dense.npz"
    if cache.exists() and not args.force:
        print(f"  Cache exists: {cache} (use --force to overwrite)")
        return

    J0_vals     = np.linspace(args.J_min, args.J_max, args.n_J)
    theta0_vals = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    NJ = len(J0_vals); NT = len(theta0_vals)
    print(f"=== D1 dense basin scan at C-cell ===")
    print(f"  J0 ∈ [{args.J_min}, {args.J_max}], n={NJ}")
    print(f"  θ0 ∈ [{args.theta_min}, {args.theta_max}], n={NT}")
    print(f"  Total: {NJ*NT} sims, workers={args.workers}")

    tasks = []
    for j, t0v in enumerate(theta0_vals):
        for i, J0v in enumerate(J0_vals):
            tasks.append((j, i, float(J0v), float(t0v), float(U0_FIXED)))

    Jm_term  = np.full((NT, NJ), np.nan)
    thm_term = np.full((NT, NJ), np.nan)
    attr     = np.full((NT, NJ), -1, dtype=np.int8)
    succ     = np.zeros((NT, NJ), dtype=bool)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, task): task for task in tasks}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            j, i = r["j"], r["i"]
            Jm_term[j, i]  = r["Jm_term"]
            thm_term[j, i] = r["thm_term"]
            attr[j, i]     = r["attractor"]
            succ[j, i]     = r["success"]
            done += 1
            print(f"  [{done:>3}/{len(tasks)}] J0={r['J0']:.2f} θ0={r['theta0']:.2f} "
                  f"-> attractor={r['attractor']} (J={r['Jm_term']:.2f}) "
                  f"{r['dt']:.0f}s")

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.0f}s = {total/60:.1f} min")

    np.savez_compressed(
        cache,
        J0_vals=J0_vals, theta0_vals=theta0_vals,
        Jm_term=Jm_term, thm_term=thm_term,
        attractor=attr, success=succ,
        cell_Bi_T=CELL_BI_T, cell_S_chi=CELL_S_CHI,
        u0=U0_FIXED, N=N_GRID, t_end=T_END,
    )
    print(f"  Saved: {cache}")


if __name__ == "__main__":
    main()
