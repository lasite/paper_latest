#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseA_amplitude.py — Phase A: amplitude sanity check.

Runs 12 PDE simulations spanning the LCST-front region of (Bi_T, S_chi)
and measures the limit-cycle amplitudes (Δθ_surf, ΔJ_surf) at the
surface (xi = 1) over t ∈ [150, 300].

The analytic prediction (scaling law iii) is
    Δθ_surf · S_chi = h(chi_inf, chi_1, Omega_e, phi_p0)  ≈ 2.34
constant across the LCST-front region, where h is read off from
data/iv_c/folds/S_chi_sweep.npz produced by Phase P.

Outputs
-------
data/iv_c/phaseA/amplitude_results.json — per-point measured amplitudes
data/iv_c/phaseA/amplitude_h_analytic.npz — h(S_chi) from Phase P sweep
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseA"
OUT.mkdir(parents=True, exist_ok=True)

TEST_POINTS = [
    (0.05, 0.7), (0.05, 1.0), (0.05, 1.3),
    (0.10, 0.7), (0.10, 1.0), (0.10, 1.3), (0.10, 1.6),
    (0.15, 0.8), (0.15, 1.1),
    (0.20, 0.9), (0.20, 1.2),
    (0.25, 1.0),
]

T_WINDOW = (200.0, 400.0)   # latter half: avoids ignition transient
T_END = 400.0
N_GRID = 101                # N=51 sees a much narrower LF region than N=301
OSC_STD_FLOOR_J = 0.05      # J_surf std below this → not oscillating


# ---------------------------------------------------------------------------
# Amplitude measurement
# ---------------------------------------------------------------------------

def measure_cycle_amplitude(result, t_window=T_WINDOW):
    """Extract Δθ_surf, ΔJ_surf, mean θ_surf, mean J_surf from PDE result."""
    t = result["t"]
    J = result["J"]
    theta = result["theta"]

    mask = (t >= t_window[0]) & (t <= t_window[1])
    if mask.sum() < 100:
        return {"is_oscillating": False, "reason": "t_window has too few samples"}

    J_surf = J[-1, mask]
    th_surf = theta[-1, mask]

    # Coarse oscillation gate: std of J_surf must exceed floor
    J_std = float(np.std(J_surf))
    if J_std < OSC_STD_FLOOR_J:
        return {
            "is_oscillating": False,
            "J_surf_std": J_std,
            "J_surf_mean": float(np.mean(J_surf)),
            "theta_surf_mean": float(np.mean(th_surf)),
            "reason": f"J_surf std {J_std:.4f} < floor {OSC_STD_FLOOR_J}",
        }

    return {
        "is_oscillating": True,
        "J_surf_max": float(np.max(J_surf)),
        "J_surf_min": float(np.min(J_surf)),
        "delta_J_surf": float(np.max(J_surf) - np.min(J_surf)),
        "theta_surf_max": float(np.max(th_surf)),
        "theta_surf_min": float(np.min(th_surf)),
        "delta_theta_surf": float(np.max(th_surf) - np.min(th_surf)),
        "J_surf_mean": float(np.mean(J_surf)),
        "theta_surf_mean": float(np.mean(th_surf)),
        "J_surf_std": J_std,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(params_kwargs):
    """Run one simulation and return amplitude metrics."""
    Bi_T = params_kwargs["Bi_T"]
    S_chi = params_kwargs["S_chi"]
    p = Params(**params_kwargs)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        amp = measure_cycle_amplitude(r)
        return {
            "Bi_T": Bi_T,
            "S_chi": S_chi,
            "amp": amp,
            "wall_s": time.perf_counter() - t0,
            "nfev": int(r.get("nfev", -1)),
            "error": None,
        }
    except Exception as e:
        return {
            "Bi_T": Bi_T,
            "S_chi": S_chi,
            "amp": None,
            "wall_s": time.perf_counter() - t0,
            "nfev": -1,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print(" Phase A — amplitude sanity check (12 PDE sims)")
    print("=" * 64)

    tasks = [
        {"Bi_T": float(bt), "S_chi": float(sc),
         "N": N_GRID, "t_end": T_END}
        for bt, sc in TEST_POINTS
    ]
    n_workers = min(len(tasks), 24)
    print(f"  test points: {len(tasks)}, workers: {n_workers}")
    print(f"  N={N_GRID}, t_end={T_END}, t_window={T_WINDOW}")

    t_total = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for r in exe.map(run_one, tasks):
            results.append(r)
            tag = f"Bi_T={r['Bi_T']:.2f}  S_chi={r['S_chi']:.1f}"
            if r["error"]:
                print(f"  {tag}: ERROR  ({r['error']})  [{r['wall_s']:.1f}s]")
            elif r["amp"] is None or not r["amp"].get("is_oscillating", False):
                reason = (r["amp"].get("reason", "no amp")
                          if r["amp"] else "no amp")
                print(f"  {tag}: NOT OSC  ({reason})  [{r['wall_s']:.1f}s]")
            else:
                a = r["amp"]
                print(f"  {tag}:  Δθ={a['delta_theta_surf']:.3f}  "
                      f"ΔJ={a['delta_J_surf']:.3f}  "
                      f"θ_mean={a['theta_surf_mean']:.3f}  "
                      f"J_mean={a['J_surf_mean']:.3f}  "
                      f"[{r['wall_s']:.1f}s]")

    elapsed = time.perf_counter() - t_total
    n_osc = sum(1 for r in results
                if r["amp"] and r["amp"].get("is_oscillating", False))
    print(f"\n  oscillating: {n_osc}/{len(results)}")
    print(f"  wall-clock: {elapsed:.1f}s")

    # Diagnostic: classify the non-oscillating points by steady-state level
    print("\n  Non-oscillating point diagnostics:")
    for r in results:
        amp = r["amp"]
        if amp is None or amp.get("is_oscillating", False):
            continue
        J_m = amp.get("J_surf_mean", float("nan"))
        th_m = amp.get("theta_surf_mean", float("nan"))
        if J_m > 1.0 and th_m < 0.5:
            cls = "cold-swollen SS"
        elif J_m < 0.3 and th_m > 1.0:
            cls = "hot-runaway SS"
        else:
            cls = "warm SS"
        print(f"    Bi_T={r['Bi_T']:.2f} S_chi={r['S_chi']:.1f}: "
              f"{cls:18s}  J_mean={J_m:.3f}  theta_mean={th_m:.3f}")

    # Save JSON FIRST so any later print error doesn't lose data
    out_json = OUT / "amplitude_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  saved {out_json}")

    # ── Analytic prediction h(S_chi) from Phase P ───────────────────
    fold_npz = OUT.parent / "folds" / "S_chi_sweep.npz"
    folds = np.load(fold_npz)
    S_arr = folds["S_chi"]
    dth_arr = folds["delta_theta"]
    h_arr = dth_arr * S_arr
    h_mean = float(np.nanmean(h_arr))
    h_std = float(np.nanstd(h_arr))
    print("\n  Analytic h = delta_theta * S_chi (from Phase P sweep)")
    print(f"    mean = {h_mean:.4f}   std = {h_std:.4f}   "
          "(constant => scaling law iii holds analytically)")

    np.savez(OUT / "amplitude_h_analytic.npz",
             S_chi=S_arr, delta_theta=dth_arr, h=h_arr,
             h_mean=h_mean, h_std=h_std)
    print(f"  saved {OUT / 'amplitude_h_analytic.npz'}")


if __name__ == "__main__":
    main()
