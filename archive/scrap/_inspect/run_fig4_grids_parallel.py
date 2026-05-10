#!/usr/bin/env python3
"""
run_fig4_grids_parallel.py — Build both fig4 grids (Bi_T×S_chi and
Bi_T×Da) in a single process pool. Serial main_full + main_da would do
2×225 = 450 simulations sequentially within each grid; here we submit
all 450 to one pool so the worker farm stays full across grid
boundaries.

Each simulate(N=301, t_end=200) costs ~8 min single-core; with
n_workers≈16–20 the whole sweep finishes in ~3 hours wall time.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pin BLAS to 1 thread per worker BEFORE numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _worker(task):
    grid_id, j, i, p_dict = task
    from fig4_data import classify_point
    r = classify_point(p_dict)
    return grid_id, j, i, r


def main():
    from fig4_data import (
        BI_T_VALS, S_CHI_VALS, DA_VALS, _grid_path, REGIME_NAMES,
        REG_FAILED, DATA_DIR,
    )
    from fig2_data import WORKING_POINT

    os.makedirs(DATA_DIR, exist_ok=True)

    n_workers = int(os.environ.get("FIG4_WORKERS", "16"))
    print(f"  Workers: {n_workers}")

    GRIDS = [
        ("main", "Bi_T", BI_T_VALS, "S_chi", S_CHI_VALS),
        ("da",   "Bi_T", BI_T_VALS, "Da",    DA_VALS),
    ]

    # Allocate result arrays; build task list (skip if cache present)
    grids_state = {}
    tasks = []
    for grid_id, px, xv, py, yv in GRIDS:
        path = Path(_grid_path(px, py))
        if path.exists():
            print(f"  cache exists for grid {grid_id}: {path.name}; skipping")
            grids_state[grid_id] = None
            continue
        NX, NY = len(xv), len(yv)
        grids_state[grid_id] = dict(
            px=px, py=py, x=xv, y=yv,
            regime      = np.full((NY, NX), REG_FAILED, dtype=int),
            J_amp_max   = np.full((NY, NX), np.nan),
            surf_amp    = np.full((NY, NX), np.nan),
            phi_max     = np.full((NY, NX), np.nan),
            phi_max_min = np.full((NY, NX), np.nan),
            J_mean      = np.full((NY, NX), np.nan),
            J_eq        = np.full((NY, NX), np.nan),
            period      = np.full((NY, NX), np.nan),
        )
        for j, ya in enumerate(yv):
            for i, xa in enumerate(xv):
                p = dict(WORKING_POINT)
                p[px] = float(xa); p[py] = float(ya)
                tasks.append((grid_id, j, i, p))

    if not tasks:
        print("  Nothing to do.")
        return

    print(f"  Submitting {len(tasks)} simulations across {len([g for g in grids_state.values() if g is not None])} grids")

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for f in as_completed(futs):
            grid_id, j, i, r = f.result()
            g = grids_state[grid_id]
            g["regime"][j, i]      = r["regime"]
            g["J_amp_max"][j, i]   = r["J_amp_max"]
            g["surf_amp"][j, i]    = r["surf_amp"]
            g["phi_max"][j, i]     = r["phi_max"]
            g["phi_max_min"][j, i] = r["phi_max_min"]
            g["J_mean"][j, i]      = r["J_mean"]
            g["J_eq"][j, i]        = r["J_eq"]
            g["period"][j, i]      = r["period"]
            done += 1
            xv = g["x"][i]; yv = g["y"][j]
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (len(tasks) - done)
            print(f"  [{done:>3}/{len(tasks)}] [{grid_id}] {g['px']}={xv:7.4f} "
                  f"{g['py']}={yv:7.4f}  "
                  f"{REGIME_NAMES.get(r['regime'],'?'):>16}  "
                  f"J_amp={r['J_amp_max']:.3f}  phi_max={r['phi_max']:.3f}  "
                  f"({elapsed/60:.0f} min elapsed, ETA {eta/60:.0f} min)",
                  flush=True)

    # Save caches
    for grid_id, g in grids_state.items():
        if g is None:
            continue
        path = _grid_path(g["px"], g["py"])
        np.savez_compressed(
            path, x=g["x"], y=g["y"], regime=g["regime"],
            J_amp_max=g["J_amp_max"], surf_amp=g["surf_amp"],
            phi_max=g["phi_max"], phi_max_min=g["phi_max_min"],
            J_mean=g["J_mean"], J_eq=g["J_eq"], period=g["period"],
        )
        print(f"  Saved: {path}")

    dt = time.perf_counter() - t0
    print(f"\n  Total: {dt:.0f}s ({dt/60:.1f} min)")


if __name__ == "__main__":
    main()
