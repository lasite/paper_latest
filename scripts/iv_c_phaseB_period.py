#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseB_period.py — Phase B: period scaling.

Scaling law (i):
    T_PDE * Bi_T  ->  ln(theta_up / theta_lo)   as Bi_T -> 0
where theta_up, theta_lo are the F-H-R folds (functions of S_chi only).

We run a 5x5 grid in (Bi_T, S_chi), measure the limit-cycle period at
the surface, and verify the collapse T*Bi_T vs the S_chi-dependent
asymptote ln(theta_up/theta_lo).

Grid choice: the plan's nominal grid (Bi_T in {0.04..0.26}, S_chi in
{0.5..1.6}) overlaps significantly with steady-front and steady-cold
regions per the fig4 phase diagram (especially Bi_T >= 0.16). The grid
below has been retuned so all 25 points sit robustly inside the LCST-
front region at the converged resolution (N=301).

Resolution: N=301, t_end=400, t_window=(200, 400) — chosen by Phase A.5
convergence verdict (period converged at N=301 across 3 representative
points to within 1% of N=401).

Robustness: per-sim 30-min hard timeout, streaming output via
multiprocessing.Pool.apply_async + polling, incremental JSON save.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseB"
OUT.mkdir(parents=True, exist_ok=True)

# ── Grid (5 x 5 = 25 points; all robustly in LF region per fig4 N=301) ──
BI_T_VALS  = np.array([0.04, 0.05, 0.07, 0.09, 0.12])
S_CHI_VALS = np.array([0.9,  1.0,  1.1,  1.3,  1.5])

# ── Resolution and integration window ──
N_GRID = 301
T_END = 400.0
T_WINDOW = (200.0, T_END)
N_SAVE = 4000

PER_SIM_TIMEOUT_S = 30 * 60          # 30 min hard timeout
OSC_J_STD_FLOOR = 0.05
RAW = OUT / "period_raw.json"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _surface(result):
    t = result["t"]
    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    return mask, t[mask], result["J"][-1, mask], result["theta"][-1, mask]


def measure_period(result):
    _, t_w, J_surf, _ = _surface(result)
    if len(t_w) < 50 or np.std(J_surf) < OSC_J_STD_FLOOR:
        return float("nan"), float("nan"), 0
    yd = J_surf - J_surf.mean()
    amp = float(yd.max() - yd.min())
    prom = max(0.15 * amp, 1e-3)
    mdist = max(3, len(yd) // 20)
    peaks, _ = find_peaks(yd, prominence=prom, distance=mdist)
    if len(peaks) < 3:
        return float("nan"), float("nan"), int(len(peaks))
    periods = np.diff(t_w[peaks])
    return float(np.mean(periods)), float(np.std(periods)), int(len(peaks))


def measure_amplitudes(result):
    _, _, J_surf, th_surf = _surface(result)
    osc = bool(np.std(J_surf) >= OSC_J_STD_FLOOR)
    return {
        "is_oscillating": osc,
        "delta_theta": float(th_surf.max() - th_surf.min()) if osc else float("nan"),
        "delta_J": float(J_surf.max() - J_surf.min()) if osc else float("nan"),
        "theta_surf_max": float(th_surf.max()),
        "theta_surf_min": float(th_surf.min()),
        "J_surf_max": float(J_surf.max()),
        "J_surf_min": float(J_surf.min()),
        "theta_surf_mean": float(th_surf.mean()),
        "J_surf_mean": float(J_surf.mean()),
        "J_surf_std": float(np.std(J_surf)),
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(task):
    Bi_T = task["Bi_T"]; S_chi = task["S_chi"]
    p = Params(Bi_T=Bi_T, S_chi=S_chi, N=N_GRID, t_end=T_END, n_save=N_SAVE)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        T_mean, T_std, npk = measure_period(r)
        amp = measure_amplitudes(r)
        return {
            "Bi_T": Bi_T, "S_chi": S_chi,
            "wall_s": time.perf_counter() - t0,
            "nfev": int(r.get("nfev", -1)),
            "T_mean": T_mean, "T_std": T_std, "n_peaks": npk,
            "amp": amp, "error": None,
        }
    except Exception as e:
        return {
            "Bi_T": Bi_T, "S_chi": S_chi,
            "wall_s": time.perf_counter() - t0, "nfev": -1,
            "T_mean": float("nan"), "T_std": float("nan"), "n_peaks": 0,
            "amp": None, "error": f"{type(e).__name__}: {e}",
        }


def make_timeout_record(task):
    return {
        "Bi_T": task["Bi_T"], "S_chi": task["S_chi"],
        "wall_s": PER_SIM_TIMEOUT_S, "nfev": -1,
        "T_mean": float("nan"), "T_std": float("nan"), "n_peaks": 0,
        "amp": None,
        "error": f"TIMEOUT (> {PER_SIM_TIMEOUT_S}s)",
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_incremental(results):
    tmp = RAW.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2, default=str)
    tmp.replace(RAW)


def save_grid(results):
    nx, ny = len(BI_T_VALS), len(S_CHI_VALS)
    T_grid = np.full((nx, ny), np.nan)
    T_std = np.full((nx, ny), np.nan)
    dTheta = np.full((nx, ny), np.nan)
    dJ = np.full((nx, ny), np.nan)
    is_osc = np.zeros((nx, ny), dtype=bool)

    for r in results:
        if r is None:
            continue
        i = int(np.argmin(np.abs(BI_T_VALS - r["Bi_T"])))
        j = int(np.argmin(np.abs(S_CHI_VALS - r["S_chi"])))
        T_grid[i, j] = r["T_mean"]
        T_std[i, j]  = r["T_std"]
        amp = r.get("amp")
        if amp:
            is_osc[i, j] = bool(amp.get("is_oscillating", False))
            dTheta[i, j] = amp.get("delta_theta", np.nan)
            dJ[i, j]     = amp.get("delta_J", np.nan)

    npz = OUT / "period_scan.npz"
    np.savez(npz,
             Bi_T=BI_T_VALS, S_chi=S_CHI_VALS,
             T_grid=T_grid, T_std=T_std,
             delta_theta_grid=dTheta, delta_J_grid=dJ,
             is_oscillating=is_osc)
    print(f"  saved {npz}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print(" Phase B - period scaling (5x5 grid at N=301)", flush=True)
    print("=" * 70, flush=True)
    tasks = [{"Bi_T": float(b), "S_chi": float(s)}
             for b in BI_T_VALS for s in S_CHI_VALS]
    n_workers = min(len(tasks), 24)
    print(f"  Bi_T  values: {list(BI_T_VALS)}", flush=True)
    print(f"  S_chi values: {list(S_CHI_VALS)}", flush=True)
    print(f"  total sims: {len(tasks)}, workers: {n_workers}", flush=True)
    print(f"  N={N_GRID}, t_end={T_END}, t_window={T_WINDOW}", flush=True)
    print(f"  per-sim timeout: {PER_SIM_TIMEOUT_S}s "
          f"({PER_SIM_TIMEOUT_S/60:.0f} min)", flush=True)
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
        tag = (f"[{t_elapsed/60:5.1f}m] BiT={r['Bi_T']:.2f} S={r['S_chi']:.2f}")
        if r["error"]:
            print(f"  {tag}: {r['error'][:50]}  [{r['wall_s']:.1f}s]", flush=True)
            return
        amp = r["amp"]
        osc = bool(amp and amp.get("is_oscillating", False))
        if osc:
            print(f"  {tag}: T={r['T_mean']:7.3f} +/- {r['T_std']:.3f}   "
                  f"dTh={amp['delta_theta']:.3f}  dJ={amp['delta_J']:.3f}  "
                  f"npk={r['n_peaks']:2d}  [{r['wall_s']/60:.1f}m]", flush=True)
        else:
            J_m = amp.get("J_surf_mean", float("nan")) if amp else float("nan")
            th_m = amp.get("theta_surf_mean", float("nan")) if amp else float("nan")
            print(f"  {tag}: NOT_OSC  J={J_m:.3f}  th={th_m:.3f}  "
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
            save_incremental(results)

        if completed < len(tasks):
            time.sleep(1.0)

    elapsed = time.perf_counter() - t_total
    print(f"\n  total wall-clock: {elapsed/60:.1f} min", flush=True)
    print("  terminating pool...", flush=True)
    pool.terminate()
    pool.join()

    save_incremental(results)
    save_grid(results)

    n_osc = sum(1 for r in results
                if r and r.get("amp") and r["amp"].get("is_oscillating"))
    n_to = sum(1 for r in results
               if r and r.get("error") and "TIMEOUT" in r["error"])
    print(f"  oscillating: {n_osc}/{len(results)}", flush=True)
    print(f"  timeouts:    {n_to}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
