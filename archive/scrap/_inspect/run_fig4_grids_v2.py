#!/usr/bin/env python3
"""
run_fig4_grids_v2.py — second-pass fig4 sweep at 25×25 (main) +
25×15 (secondary), with three improvements over the v1 runner:

  1. Wall-time event in solve_ivp aborts any single cell after
     `MAX_WALL_SEC` (default 1500s = 25 min). SNIC stalls now return
     REG_FAILED instead of hanging the master.
  2. 24 workers (vs 16) — server has 32 cores, plenty of headroom
     for BLAS=1 single-thread sims.
  3. Logs every cell completion to stdout in the same regex-friendly
     format as v1, so Plan-B log-mining still works as fallback.

Per-sim time: ~9 min @ N=301 t_end=200 single-core.
At 25×25 + 25×15 = 1000 sims / 24 workers = ~6-8 hours wall.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pin BLAS BEFORE numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


MAX_WALL_SEC = 1500.0  # 25 min per cell hard cap


def _classify_with_timeout(p_dict, max_wall_sec=MAX_WALL_SEC):
    """Drop-in replacement for fig4_data.classify_point that adds a
    solve_ivp wall-time event so SNIC slow-flow cells can't hang.
    Mirrors classify_point's diagnostic dict exactly so the master
    can stay agnostic.
    """
    from scan_optimized import (
        Params, finalize_params, rhs_mol_logJ, make_jac_sparsity,
        make_sparse_fd_jac, initial_state, _LOG_J_MAX,
    )
    from fig4_data import (
        SIM_OVERRIDES, T_ANA, AMP_THRESH, PHI_LCST,
        REG_FAILED, REG_STEADY_COLD, REG_BULK_HOPF, REG_LCST_FRONT,
        REG_GLOBAL_COLLAPSE, REG_STEADY_COLLAPSED, REG_STEADY_FRONT,
    )
    from fig3_data import cold_J_eq
    from scipy.integrate import solve_ivp
    from scipy.signal import find_peaks

    out = dict(regime=REG_FAILED, J_amp_max=np.nan, phi_max=np.nan,
               phi_max_min=np.nan, J_mean=np.nan, J_eq=np.nan,
               surf_amp=np.nan, period=np.nan)
    try:
        p = Params(**{**p_dict, **SIM_OVERRIDES})
        p = finalize_params(p)
        x, y0 = initial_state(p)
        n3 = 3 * p.N
        rhs_fn = lambda t, y: rhs_mol_logJ(t, y, p)
        S = make_jac_sparsity(p.N)
        jac_sparse, _ = make_sparse_fd_jac(rhs_fn, S, n3)

        # Wall-time event: returns positive while OK, negative when
        # elapsed wall time exceeds the budget. terminal=True halts
        # integration as soon as the event crosses zero.
        t0_wall = time.perf_counter()

        def wall_event(t, y):
            return max_wall_sec - (time.perf_counter() - t0_wall)
        wall_event.terminal = True
        wall_event.direction = -1

        sol = solve_ivp(
            fun=rhs_fn, jac=jac_sparse,
            t_span=(0.0, p.t_end), y0=y0,
            t_eval=np.linspace(0, p.t_end, p.n_save),
            method=p.method, rtol=p.rtol, atol=p.atol,
            max_step=p.max_step,
            events=wall_event,
        )
    except Exception as e:
        out["error"] = f"setup-or-solve-failed: {e}"
        return out

    if not sol.success or sol.status == 1:
        # status==1 means an event terminated integration → wall-time hit
        out["error"] = ("wall-timeout" if sol.status == 1
                        else f"solver-fail: {sol.message}")
        return out

    n = p.N
    log_J_min = np.log(p.phi_p0 * 1.02)
    J = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
    theta = sol.y[2*n:]
    t = sol.t
    idx = (t >= T_ANA[0]) & (t <= T_ANA[1])
    if idx.sum() < 50:
        out["error"] = "short-window"
        return out

    Jw = J[:, idx]
    phi = p.phi_p0 / np.maximum(Jw, 1e-12)
    J_amp_per_xi = Jw.max(axis=1) - Jw.min(axis=1)
    phi_max_per_xi = phi.max(axis=1)

    out["J_amp_max"]   = float(J_amp_per_xi.max())
    out["phi_max"]     = float(phi_max_per_xi.max())
    out["phi_max_min"] = float(phi_max_per_xi.min())
    out["J_mean"]      = float(Jw.mean())
    out["surf_amp"]    = float(Jw[-1].max() - Jw[-1].min())

    try:
        Js = Jw[-1] - Jw[-1].mean()
        prom = max(0.05 * out["surf_amp"], 0.01)
        pk, _ = find_peaks(Js, prominence=prom, distance=3)
        if len(pk) >= 2:
            tt = t[idx]
            out["period"] = float(np.median(np.diff(tt[pk])))
    except Exception:
        pass

    J_eq = cold_J_eq(p_dict)
    out["J_eq"] = float(J_eq) if np.isfinite(J_eq) else 1.0

    is_steady = (out["surf_amp"] < AMP_THRESH and
                 out["J_amp_max"] < AMP_THRESH)
    if is_steady:
        if out["phi_max"] < PHI_LCST:
            out["regime"] = REG_STEADY_COLD
        elif out["phi_max_min"] > PHI_LCST:
            out["regime"] = REG_STEADY_COLLAPSED
        else:
            out["regime"] = REG_STEADY_FRONT
    else:
        if out["phi_max"] < PHI_LCST:
            out["regime"] = REG_BULK_HOPF
        elif out["phi_max_min"] > PHI_LCST:
            out["regime"] = REG_GLOBAL_COLLAPSE
        else:
            out["regime"] = REG_LCST_FRONT

    return out


def _worker(task):
    grid_id, j, i, p_dict = task
    r = _classify_with_timeout(p_dict)
    return grid_id, j, i, r


def main():
    from fig4_data import (
        BI_T_VALS, S_CHI_VALS, DA_VALS, _grid_path, REGIME_NAMES,
        REG_FAILED, DATA_DIR,
    )
    from fig2_data import WORKING_POINT

    os.makedirs(DATA_DIR, exist_ok=True)
    n_workers = int(os.environ.get("FIG4_WORKERS", "24"))
    print(f"  Workers: {n_workers}")
    print(f"  BI_T_VALS: {len(BI_T_VALS)} pts ({BI_T_VALS[0]:.4f}..{BI_T_VALS[-1]:.4f})")
    print(f"  S_CHI_VALS: {len(S_CHI_VALS)} pts ({S_CHI_VALS[0]:.2f}..{S_CHI_VALS[-1]:.2f})")
    print(f"  DA_VALS: {len(DA_VALS)} pts ({DA_VALS[0]:.4f}..{DA_VALS[-1]:.4f})")
    print(f"  Per-cell wall-time cap: {MAX_WALL_SEC:.0f}s")

    GRIDS = [
        ("main", "Bi_T", BI_T_VALS, "S_chi", S_CHI_VALS),
        ("da",   "Bi_T", BI_T_VALS, "Da",    DA_VALS),
    ]

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

    print(f"  Submitting {len(tasks)} simulations across "
          f"{len([g for g in grids_state.values() if g is not None])} grids")

    t0 = time.perf_counter()
    done = 0
    n_failed_wall = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for f in as_completed(futs):
            grid_id, j, i, r = f.result()
            g = grids_state[grid_id]
            for k in ("regime", "J_amp_max", "surf_amp", "phi_max",
                      "phi_max_min", "J_mean", "J_eq", "period"):
                g[k][j, i] = r[k]
            done += 1
            xv = g["x"][i]; yv = g["y"][j]
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (len(tasks) - done)
            err_tag = ""
            if r["regime"] == REG_FAILED:
                n_failed_wall += 1
                err_tag = f"  ERR={r.get('error','?')[:40]}"
            print(f"  [{done:>4}/{len(tasks)}] [{grid_id}] {g['px']}={xv:7.4f} "
                  f"{g['py']}={yv:7.4f}  "
                  f"{REGIME_NAMES.get(r['regime'],'?'):>16}  "
                  f"J_amp={r['J_amp_max']:.3f}  phi_max={r['phi_max']:.3f}  "
                  f"({elapsed/60:.0f} min, ETA {eta/60:.0f} min){err_tag}",
                  flush=True)

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
    print(f"  Failed/timeout cells: {n_failed_wall}/{len(tasks)}")


if __name__ == "__main__":
    main()
