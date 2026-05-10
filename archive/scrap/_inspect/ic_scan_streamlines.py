#!/usr/bin/env python3
"""
ic_scan_streamlines.py — IC sweep at the C-cell to reveal the basin
structure as a streamline-like ensemble in (⟨J⟩, ⟨θ⟩) projection.

At Bi_T=0.059, S_χ=1.80 (representative C-cell with confirmed
bistability), run a 5×5 = 25 grid of uniform initial conditions in
(J_0, θ_0) and trace each trajectory's projected (⟨J⟩(t), ⟨θ⟩(t)).
Save each trajectory to disk so make_fig6_manifold.py can load them
and overlay all 25 paths in panel (c), showing how trajectories from
different IC regions converge to either the LCST-front cycle or the
hot-runaway state.

Cost: 25 sims × ~1.5 min (N=121) ≈ 6-8 min on 4 workers.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# CELL: representative C-cell where bistability is established
CELL_BI_T  = 0.059
CELL_S_CHI = 1.80

N_GRID  = 121
T_END   = 200.0
N_SAVE  = 1000  # downsample to keep cache small
N_WORK  = int(os.environ.get("FIG4_WORKERS", 24))

# IC grid (uniform-IC parametrization)
J0_VALS    = np.array([0.20, 0.40, 0.70, 1.00, 1.30])
THETA0_VALS = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
U0_FIXED    = 0.5  # mid-range; both attractors deplete u at surface anyway

# Cache target
DATA_DIR = _HERE.parent.parent / "data" / "fig4"
CACHE = DATA_DIR / "ic_streamlines_C.npz"


def _worker(task):
    """Run one PDE simulation with uniform (J0, θ0, u0) IC, return
    projected ⟨J⟩(t), ⟨θ⟩(t) and terminal mean values."""
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

    x = cell_centers(p.N)
    n = p.N

    log_J_min = np.log(p.phi_p0 * 1.02)
    Jvec = np.maximum(np.full(n, J0) + 1e-3 * np.cos(np.pi * x),
                      np.exp(log_J_min) + 1e-6)
    uvec = np.maximum(np.full(n, u0) + 1e-3 * np.cos(np.pi * x),
                      p.u_floor)
    thvec = np.full(n, theta0) + 1e-3 * np.cos(np.pi * x)
    Wvec = Jvec * uvec
    y0 = np.concatenate([np.log(Jvec), Wvec, thvec])

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
    dt = time.perf_counter() - t0_wall

    success = sol.success
    if success:
        J = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
        theta = sol.y[2*n:]
        Jm  = J.mean(axis=0)
        thm = theta.mean(axis=0)
        # Terminal: last 5%
        n5 = max(int(0.05 * len(sol.t)), 5)
        Jm_term  = float(Jm[-n5:].mean())
        thm_term = float(thm[-n5:].mean())
    else:
        Jm = thm = np.full(p.n_save, np.nan)
        Jm_term = thm_term = np.nan

    return dict(
        j_idx=j_idx, i_idx=i_idx, J0=J0, theta0=theta0,
        Jm=Jm, thm=thm, t=sol.t,
        Jm_term=Jm_term, thm_term=thm_term,
        success=bool(success), dt=dt,
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        print(f"  Cache already present: {CACHE}")
        print(f"  Delete to re-run.")
        return

    tasks = []
    for j, theta0 in enumerate(THETA0_VALS):
        for i, J0 in enumerate(J0_VALS):
            tasks.append((j, i, float(J0), float(theta0), float(U0_FIXED)))

    print(f"=== IC streamline scan at C-cell ===")
    print(f"  Bi_T={CELL_BI_T}, S_chi={CELL_S_CHI}")
    print(f"  J0 ∈ {list(J0_VALS)}")
    print(f"  θ0 ∈ {list(THETA0_VALS)}")
    print(f"  u0 = {U0_FIXED}")
    print(f"  Grid: {len(J0_VALS)} × {len(THETA0_VALS)} = {len(tasks)} ICs")
    print(f"  Workers: {N_WORK}")
    print()

    NJ = len(J0_VALS)
    NT = len(THETA0_VALS)
    Jm_grid    = np.full((NT, NJ, N_SAVE), np.nan)
    thm_grid   = np.full((NT, NJ, N_SAVE), np.nan)
    Jm_term    = np.full((NT, NJ), np.nan)
    thm_term   = np.full((NT, NJ), np.nan)
    success    = np.zeros((NT, NJ), dtype=bool)
    t_axis     = None

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORK) as ex:
        futs = {ex.submit(_worker, task): task for task in tasks}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            j, i = r["j_idx"], r["i_idx"]
            Jm_grid[j, i, :]  = r["Jm"]
            thm_grid[j, i, :] = r["thm"]
            Jm_term[j, i]     = r["Jm_term"]
            thm_term[j, i]    = r["thm_term"]
            success[j, i]     = r["success"]
            if t_axis is None and r["success"]:
                t_axis = r["t"]
            done += 1
            print(f"  [{done:>2}/{len(tasks)}] J0={r['J0']:.2f}, "
                  f"θ0={r['theta0']:.1f}: term=(J={r['Jm_term']:.2f}, "
                  f"θ={r['thm_term']:.2f}), {r['dt']:.0f}s, "
                  f"success={r['success']}")

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.0f}s = {total/60:.1f} min")

    np.savez_compressed(
        CACHE,
        Bi_T=CELL_BI_T, S_chi=CELL_S_CHI,
        N=N_GRID, t_end=T_END, u0=U0_FIXED,
        J0_vals=J0_VALS, theta0_vals=THETA0_VALS,
        Jm_grid=Jm_grid, thm_grid=thm_grid,
        Jm_term=Jm_term, thm_term=thm_term,
        success=success, t=t_axis,
    )
    print(f"  Saved: {CACHE}")


if __name__ == "__main__":
    main()
