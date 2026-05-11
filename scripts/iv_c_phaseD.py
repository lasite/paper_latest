#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseD.py — Phase D: front-depth scaling.

Two scans:
  (1) Main grid (m_act, m_diff) in {1..7}^2 = 49 sims at
      Bi_T=0.10, Da=4.0, S_chi=1.0  (the working point)
  (2) Bi_T slice: m_act = m_diff = 4, Bi_T in {0.06, 0.10, 0.16, 0.25}
      = 4 sims; tests L_T scaling at fixed transport exponents.

Per sim we measure
  xi_LCST            innermost xi where max_t(phi) > 0.5 during cycle
  classification     5-way long-time attractor type:
                        cycle              (oscillating)
                        hot_runaway        (all phi > 0.5, steady)
                        overswollen_front  (steady, phi crosses 0.5,
                                            J_mean > 2; chemistry-driven
                                            over-swollen bulk + thin
                                            LCST barrier)
                        frozen_front       (steady, phi crosses 0.5,
                                            J_mean <= 2; classic
                                            cold-core + collapsed shell)
                        cold_SS            (steady, all phi < 0.5)

Both scans run with N=301, t_end=400, t_window=(200, 400), per-sim
30-min timeout, streaming output and incremental save.

Outputs
  data/iv_c/phaseD/main_grid_raw.json
  data/iv_c/phaseD/main_grid.npz              (xi_LCST, classification)
  data/iv_c/phaseD/BiT_slice_raw.json
  data/iv_c/phaseD/BiT_slice.npz
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseD"
OUT.mkdir(parents=True, exist_ok=True)

# Main grid — m_act, m_diff in [1, 7]
M_ACT_VALS = np.arange(1, 8)        # 1..7
M_DIFF_VALS = np.arange(1, 8)

# Bi_T slice grid (fixed m_act=m_diff=4)
BI_T_VALS = np.array([0.06, 0.10, 0.16, 0.25])
M_FIXED = 4

# Working-point constants
DA_WP = 4.0
S_CHI_WP = 1.0
BI_T_WP = 0.10

# Sim settings
N_GRID = 301
T_END = 400.0
T_WINDOW = (200.0, T_END)
N_SAVE = 4000
PER_SIM_TIMEOUT_S = 30 * 60          # 30 min
OSC_J_STD_FLOOR = 0.05
PHI_LCST = 0.5
J_OVERSWOLLEN_THRESHOLD = 2.0        # J_mean above this → "over-swollen"
J_COLLAPSED_THRESHOLD = 0.3          # J_max below this → "hot-runaway"


# ---------------------------------------------------------------------------
# Per-sim measurement
# ---------------------------------------------------------------------------

def classify_long_time(result):
    """Return dict with xi_LCST, classification, and diagnostic scalars."""
    t = result["t"]
    x = result["x"]
    J = result["J"]
    theta = result["theta"]
    phi = result["phi"]

    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    if mask.sum() < 50:
        return dict(classification="window_too_small",
                    xi_LCST=float("nan"),
                    J_mean=float("nan"), J_max=float("nan"),
                    J_min=float("nan"), J_surf_std=float("nan"),
                    phi_max=float("nan"), phi_max_min=float("nan"),
                    theta_mean=float("nan"))

    Jw = J[:, mask]
    th_w = theta[:, mask]
    phi_w = phi[:, mask]

    J_surf = Jw[-1]
    J_surf_std = float(np.std(J_surf))
    is_oscillating = J_surf_std > OSC_J_STD_FLOOR

    phi_max_over_t = phi_w.max(axis=1)         # per-xi max phi
    phi_max = float(phi_max_over_t.max())      # global max phi
    phi_max_min = float(phi_max_over_t.min())  # min over xi of (max_t phi)
    J_mean = float(Jw.mean())
    J_max = float(Jw.max())
    J_min = float(Jw.min())
    theta_mean = float(th_w.mean())

    # xi_LCST: innermost xi where phi exceeds LCST during cycle
    above = phi_max_over_t > PHI_LCST
    if not above.any():
        xi_LCST = 1.0                  # no front (cold-swollen everywhere)
    elif above.all():
        xi_LCST = 0.0                  # full collapse (hot-runaway)
    else:
        xi_LCST = float(x[int(np.argmax(above))])  # smallest xi above LCST

    # Classification cascade
    if is_oscillating:
        classification = "cycle"
    elif phi_max_min > PHI_LCST:
        classification = "hot_runaway"             # all xi collapsed
    elif phi_max <= PHI_LCST and J_max > J_OVERSWOLLEN_THRESHOLD:
        # Over-swollen with NO collapsed cell — pure over-swollen bulk
        # (rare, but possible at low Da before barrier nucleates)
        classification = "overswollen_uniform"
    elif phi_max > PHI_LCST and J_mean > J_OVERSWOLLEN_THRESHOLD:
        # Chemistry-driven over-swollen bulk + thin LCST barrier
        classification = "overswollen_front"
    elif phi_max > PHI_LCST:
        # Classic frozen front: cold-core + collapsed-shell
        classification = "frozen_front"
    else:
        classification = "cold_SS"

    return dict(
        classification=classification,
        is_oscillating=is_oscillating,
        xi_LCST=xi_LCST,
        J_surf_std=J_surf_std,
        J_mean=J_mean, J_max=J_max, J_min=J_min,
        phi_max=phi_max, phi_max_min=phi_max_min,
        theta_mean=theta_mean,
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(task):
    Bi_T = task["Bi_T"]; Da = task["Da"]
    S_chi = task["S_chi"]; m_act = task["m_act"]; m_diff = task["m_diff"]
    p = Params(Bi_T=Bi_T, Da=Da, S_chi=S_chi,
               m_act=m_act, m_diff=m_diff,
               N=N_GRID, t_end=T_END, n_save=N_SAVE)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        info = classify_long_time(r)
        return {**task, **info,
                "wall_s": time.perf_counter() - t0,
                "nfev": int(r.get("nfev", -1)),
                "error": None}
    except Exception as e:
        return {**task,
                "classification": "sim_error",
                "is_oscillating": False, "xi_LCST": float("nan"),
                "J_surf_std": float("nan"),
                "J_mean": float("nan"), "J_max": float("nan"),
                "J_min": float("nan"), "phi_max": float("nan"),
                "phi_max_min": float("nan"),
                "theta_mean": float("nan"),
                "wall_s": time.perf_counter() - t0,
                "nfev": -1,
                "error": f"{type(e).__name__}: {e}"}


def make_timeout_record(task):
    return {**task,
            "classification": "TIMEOUT",
            "is_oscillating": False, "xi_LCST": float("nan"),
            "J_surf_std": float("nan"),
            "J_mean": float("nan"), "J_max": float("nan"),
            "J_min": float("nan"), "phi_max": float("nan"),
            "phi_max_min": float("nan"),
            "theta_mean": float("nan"),
            "wall_s": PER_SIM_TIMEOUT_S, "nfev": -1,
            "error": f"TIMEOUT (> {PER_SIM_TIMEOUT_S}s)"}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(records, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump([r for r in records if r is not None],
                  f, indent=2, default=str)
    tmp.replace(path)


def save_grid(results, BI_T, Da, S_chi):
    """Reshape main-grid results into (n_mact, n_mdiff) arrays."""
    nx, ny = len(M_ACT_VALS), len(M_DIFF_VALS)
    xi_grid = np.full((nx, ny), np.nan)
    class_grid = np.full((nx, ny), "unfilled", dtype="<U24")
    Jmean_grid = np.full((nx, ny), np.nan)
    phimax_grid = np.full((nx, ny), np.nan)

    for r in results:
        if r is None: continue
        i = int(np.argmin(np.abs(M_ACT_VALS - r["m_act"])))
        j = int(np.argmin(np.abs(M_DIFF_VALS - r["m_diff"])))
        xi_grid[i, j] = r["xi_LCST"]
        class_grid[i, j] = r["classification"]
        Jmean_grid[i, j] = r["J_mean"]
        phimax_grid[i, j] = r["phi_max"]

    np.savez(OUT / "main_grid.npz",
             m_act=M_ACT_VALS, m_diff=M_DIFF_VALS,
             Bi_T=BI_T, Da=Da, S_chi=S_chi,
             xi_LCST=xi_grid, classification=class_grid,
             J_mean=Jmean_grid, phi_max=phimax_grid)


def save_slice(results):
    n = len(BI_T_VALS)
    xi = np.full(n, np.nan)
    cls = np.full(n, "unfilled", dtype="<U24")
    Jmean = np.full(n, np.nan)
    phimax = np.full(n, np.nan)
    for r in results:
        if r is None: continue
        i = int(np.argmin(np.abs(BI_T_VALS - r["Bi_T"])))
        xi[i] = r["xi_LCST"]
        cls[i] = r["classification"]
        Jmean[i] = r["J_mean"]
        phimax[i] = r["phi_max"]
    np.savez(OUT / "BiT_slice.npz",
             Bi_T=BI_T_VALS, m_act=M_FIXED, m_diff=M_FIXED,
             xi_LCST=xi, classification=cls,
             J_mean=Jmean, phi_max=phimax)


# ---------------------------------------------------------------------------
# Driver — runs a list of tasks in parallel with per-sim timeout
# ---------------------------------------------------------------------------

def run_tasks(tasks, raw_path, banner):
    print("=" * 70, flush=True)
    print(f" {banner}", flush=True)
    print("=" * 70, flush=True)
    n_workers = min(len(tasks), 24)
    print(f"  total sims: {len(tasks)}, workers: {n_workers}", flush=True)
    print(f"  N={N_GRID}, t_end={T_END}, n_save={N_SAVE}", flush=True)
    print(f"  per-sim timeout: {PER_SIM_TIMEOUT_S}s", flush=True)
    print(flush=True)

    t_total = time.perf_counter()
    pool = mp.Pool(processes=n_workers)
    pending = {}
    for i, t in enumerate(tasks):
        ar = pool.apply_async(run_one, (t,))
        pending[ar] = (i, t, time.perf_counter() + PER_SIM_TIMEOUT_S)

    results = [None] * len(tasks)
    printed = set()
    completed = 0

    def _print(r, t_elapsed):
        tag = (f"[{t_elapsed/60:5.1f}m]  "
               f"BiT={r['Bi_T']:.2f} Da={r['Da']:.2f} "
               f"m_act={r['m_act']:.0f} m_diff={r['m_diff']:.0f}")
        cls = r.get("classification", "?")
        if r["error"]:
            print(f"  {tag}: {cls:18s}  {r['error'][:40]}  "
                  f"[{r['wall_s']/60:.1f}m]", flush=True)
            return
        xi = r.get("xi_LCST", float('nan'))
        Jmean = r.get("J_mean", float('nan'))
        phimax = r.get("phi_max", float('nan'))
        print(f"  {tag}: {cls:18s}  xi_L={xi:.3f}  "
              f"Jmean={Jmean:.3f}  phimax={phimax:.3f}  "
              f"[{r['wall_s']/60:.1f}m]", flush=True)

    while completed < len(tasks):
        finished_now = []
        for ar, (i, t, deadline) in list(pending.items()):
            if ar.ready():
                try:
                    r = ar.get(timeout=0.0)
                except Exception as e:
                    r = {**make_timeout_record(t),
                         "error": f"exception: {type(e).__name__}: {e}"}
                results[i] = r
                finished_now.append(ar)
                completed += 1
            elif time.perf_counter() > deadline:
                results[i] = make_timeout_record(t)
                finished_now.append(ar)
                completed += 1

        for ar in finished_now:
            del pending[ar]

        for idx, r in enumerate(results):
            if r is None or idx in printed:
                continue
            _print(r, time.perf_counter() - t_total)
            printed.add(idx)

        if finished_now:
            save_json(results, raw_path)

        if completed < len(tasks):
            time.sleep(1.0)

    pool.terminate(); pool.join()
    save_json(results, raw_path)
    elapsed = time.perf_counter() - t_total
    print(f"\n  wall-clock: {elapsed/60:.1f} min", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Main grid: (m_act, m_diff) at fixed WP ──────────────────────
    main_tasks = [
        {"Bi_T": BI_T_WP, "Da": DA_WP, "S_chi": S_CHI_WP,
         "m_act": int(ma), "m_diff": int(md)}
        for ma in M_ACT_VALS for md in M_DIFF_VALS
    ]
    main_results = run_tasks(main_tasks,
                              OUT / "main_grid_raw.json",
                              "Phase D - main grid (49 sims)")
    save_grid(main_results, BI_T_WP, DA_WP, S_CHI_WP)
    print(f"  saved {OUT / 'main_grid.npz'}", flush=True)

    # Class tally
    classes = [r["classification"] for r in main_results
               if r is not None]
    from collections import Counter
    print(f"\n  Class tally (main grid): {dict(Counter(classes))}",
          flush=True)
    print(flush=True)

    # ── Bi_T slice (m_act = m_diff = 4) ─────────────────────────────
    slice_tasks = [
        {"Bi_T": float(b), "Da": DA_WP, "S_chi": S_CHI_WP,
         "m_act": M_FIXED, "m_diff": M_FIXED}
        for b in BI_T_VALS
    ]
    slice_results = run_tasks(slice_tasks,
                               OUT / "BiT_slice_raw.json",
                               "Phase D - Bi_T slice (4 sims)")
    save_slice(slice_results)
    print(f"  saved {OUT / 'BiT_slice.npz'}", flush=True)


if __name__ == "__main__":
    main()
