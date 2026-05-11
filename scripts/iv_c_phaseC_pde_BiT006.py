#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseC_pde_BiT006.py — Phase C resume: bisection for Bi_T=0.06 only.

The original parallel driver (iv_c_phaseC_pde.py) lost its Bi_T=0.06
chain when the background python process tree was killed at ~25 min.
This script runs ONLY that chain, sequentially in foreground, with
stdout flushing after every sim so progress is visible in real time.

Identical bisection algorithm and brackets to the original.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseC"

BI_T = 0.06
DA_C_0D = BI_T / (1.30 * 1.5)     # = 0.030769  (= Bi_T / (J_init * Gamma_A))
DA_LO_INIT = 0.5 * DA_C_0D
DA_HI_INIT = 3.0 * DA_C_0D
MAX_BRACKET_HI = 4.0

N_GRID = 301
T_END = 400.0
T_WINDOW = (300.0, T_END)
N_SAVE = 4000
OSC_J_STD_FLOOR = 0.05

LOG_PATH = OUT / "pde_bisection_log_BiT006.json"


def is_oscillating(result):
    t = result["t"]
    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    if mask.sum() < 50:
        return False, dict(reason="window too small")
    Js = result["J"][-1, mask]
    std = float(np.std(Js))
    return (std > OSC_J_STD_FLOOR), dict(
        std=std, J_mean=float(np.mean(Js)),
        J_max=float(np.max(Js)), J_min=float(np.min(Js)),
    )


def run_sim(Da, iteration):
    p = Params(Bi_T=BI_T, Da=Da, N=N_GRID, t_end=T_END, n_save=N_SAVE)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        osc, diag = is_oscillating(r)
        wall = time.perf_counter() - t0
        print(f"  [iter {iteration:2d}] Da={Da:.5f}  "
              f"osc={'TRUE ' if osc else 'False'}  "
              f"std={diag['std']:.4f}  Jmean={diag['J_mean']:.3f}  "
              f"Jmax={diag.get('J_max', float('nan')):.3f}  "
              f"Jmin={diag.get('J_min', float('nan')):.3f}  "
              f"[{wall:.1f}s]", flush=True)
        return {
            "Bi_T": BI_T, "Da": Da, "iter": iteration,
            "wall_s": wall, "is_oscillating": osc, "diag": diag,
            "error": None,
        }
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"  [iter {iteration:2d}] Da={Da:.5f}  ERROR  "
              f"{type(e).__name__}: {str(e)[:60]}  [{wall:.1f}s]",
              flush=True)
        return {
            "Bi_T": BI_T, "Da": Da, "iter": iteration,
            "wall_s": wall, "is_oscillating": False, "diag": None,
            "error": f"{type(e).__name__}: {e}",
        }


def save_log(records):
    tmp = LOG_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, default=str)
    tmp.replace(LOG_PATH)


def main():
    print("=" * 70, flush=True)
    print(f" Phase C resume - Bi_T = {BI_T} bisection ONLY (sequential)",
          flush=True)
    print("=" * 70, flush=True)
    print(f"  Da_c^0D = {DA_C_0D:.5f}", flush=True)
    print(f"  Initial bracket: [{DA_LO_INIT:.5f}, {DA_HI_INIT:.5f}]",
          flush=True)
    print(f"  Settings: N={N_GRID}, t_end={T_END}, "
          f"t_window={T_WINDOW}, n_save={N_SAVE}", flush=True)
    print(f"  is_oscillating threshold: sigma(J_surf) > {OSC_J_STD_FLOOR}",
          flush=True)
    print(flush=True)

    t_total = time.perf_counter()
    records = []
    Da_lo = DA_LO_INIT
    Da_hi = DA_HI_INIT
    it = 0

    r_lo = run_sim(Da_lo, it); records.append(r_lo); save_log(records); it += 1
    r_hi = run_sim(Da_hi, it); records.append(r_hi); save_log(records); it += 1

    expansions = 0
    while r_lo["is_oscillating"] == r_hi["is_oscillating"]:
        if r_hi["is_oscillating"]:
            Da_lo /= 2.0
            print(f"  bracket too high → lower Da_lo to {Da_lo:.5f}",
                  flush=True)
            r_lo = run_sim(Da_lo, it); records.append(r_lo)
            save_log(records); it += 1
        else:
            if Da_hi >= MAX_BRACKET_HI:
                print(f"  bracket cap reached (Da_hi={Da_hi:.3f}), "
                      f"no oscillation found. Marking NaN.", flush=True)
                np.savez(OUT / "Da_c_pde_BiT006.npz",
                         Bi_T=BI_T, Da_c_pde=float("nan"),
                         Da_c_0D=DA_C_0D,
                         reason="no oscillation up to Da=4")
                save_log(records)
                return
            Da_hi = min(Da_hi * 2.0, MAX_BRACKET_HI)
            print(f"  bracket too low → raise Da_hi to {Da_hi:.5f}",
                  flush=True)
            r_hi = run_sim(Da_hi, it); records.append(r_hi)
            save_log(records); it += 1
        expansions += 1
        if expansions > 6:
            print(f"  cannot bracket after {expansions} expansions",
                  flush=True)
            np.savez(OUT / "Da_c_pde_BiT006.npz",
                     Bi_T=BI_T, Da_c_pde=float("nan"),
                     Da_c_0D=DA_C_0D,
                     reason=f"could not bracket after {expansions} expansions")
            save_log(records)
            return

    print(f"  bracketed: lo={Da_lo:.5f} (NOT_OSC), hi={Da_hi:.5f} (OSC)",
          flush=True)

    for k in range(8):
        Da_mid = float(np.sqrt(Da_lo * Da_hi))
        r_mid = run_sim(Da_mid, it); records.append(r_mid)
        save_log(records); it += 1
        if r_mid["is_oscillating"]:
            Da_hi = Da_mid
        else:
            Da_lo = Da_mid
        print(f"     → new bracket [{Da_lo:.5f}, {Da_hi:.5f}]", flush=True)

    Da_c = float(np.sqrt(Da_lo * Da_hi))
    elapsed = time.perf_counter() - t_total

    print(flush=True)
    print(f"  Da_c^PDE = {Da_c:.5f}", flush=True)
    print(f"  shift = (Da_c^PDE - Da_c^0D) / Da_c^0D = "
          f"{(Da_c - DA_C_0D)/DA_C_0D:.2%}", flush=True)
    print(f"  total wall-clock: {elapsed/60:.1f} min", flush=True)

    np.savez(OUT / "Da_c_pde_BiT006.npz",
             Bi_T=BI_T, Da_c_pde=Da_c, Da_c_0D=DA_C_0D,
             Da_lo_final=Da_lo, Da_hi_final=Da_hi,
             reason="converged",
             wall_s=elapsed)
    save_log(records)
    print(f"  saved {OUT / 'Da_c_pde_BiT006.npz'}", flush=True)


if __name__ == "__main__":
    main()
