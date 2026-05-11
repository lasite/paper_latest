#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_lt_patch_sweep.py — L_T patch L_T.2: 4-point Bi_T sweep at alpha=0.04,
delta=0.50. Each sim is N=301, t_end=400.

Runs all 4 in parallel via multiprocessing.Pool.apply_async with a per-sim
timeout to avoid losing progress if one sim hangs. Incremental save after
each completion.
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

OUT = _HERE.parent / "data" / "iv_c" / "lt_patch"
OUT.mkdir(parents=True, exist_ok=True)

BI_T_VALS = [0.06, 0.10, 0.16, 0.25]
ALPHA = 0.04
DELTA = 0.50
BI_C = 0.70
N_GRID = 301
T_END = 400.0
N_SAVE = 4000
PER_SIM_TIMEOUT = 35 * 60   # 35 min
PHASE_D_PLATEAU = 0.388


def run_one(Bi_T):
    """Worker function — runs one sim, returns plain-dict summary (no arrays)."""
    try:
        p = Params(
            Bi_T=float(Bi_T), S_chi=1.0,
            alpha=ALPHA, delta=DELTA, Bi_c=BI_C,
            N=N_GRID, t_end=T_END, n_save=N_SAVE,
        )
        p = finalize_params(p)
        t0 = time.perf_counter()
        r = simulate(p)
        wall = time.perf_counter() - t0

        t = r["t"]
        x = r["x"]
        J = r["J"]
        theta = r["theta"]
        phi = r["phi"]

        mask = t >= 200
        J_surf = J[-1, mask]
        theta_surf = theta[-1, mask]
        phi_max = phi[:, mask].max(axis=1)

        is_osc = bool(np.std(J_surf) > 0.05)

        crossing = np.where(phi_max > 0.5)[0]
        if len(crossing) > 0:
            xi_LCST = float(x[crossing[0]])
        else:
            xi_LCST = 1.0
        if (phi_max > 0.5).all():
            xi_LCST = 0.0

        L_T = float(np.sqrt(ALPHA / Bi_T))
        L_c = float(np.sqrt(DELTA / BI_C))

        # Save the per-sim raw arrays for later inspection
        np.savez(OUT / f"sim_BiT{Bi_T:.2f}.npz",
                 t=t, x=x, J=J, theta=theta, phi=phi,
                 Bi_T=Bi_T, alpha=ALPHA, delta=DELTA, Bi_c=BI_C,
                 N=N_GRID, t_end=T_END)

        return {
            "Bi_T": float(Bi_T),
            "is_oscillating": is_osc,
            "xi_LCST": xi_LCST,
            "J_surf_peak": float(np.max(J_surf)),
            "J_surf_trough": float(np.min(J_surf)),
            "J_surf_std": float(np.std(J_surf)),
            "theta_surf_peak": float(np.max(theta_surf)),
            "theta_surf_trough": float(np.min(theta_surf)),
            "L_T": L_T, "L_c": L_c,
            "wall_s": wall,
            "error": None,
        }
    except Exception as e:
        return {
            "Bi_T": float(Bi_T),
            "is_oscillating": False,
            "xi_LCST": float("nan"),
            "L_T": float(np.sqrt(ALPHA / Bi_T)),
            "L_c": float(np.sqrt(DELTA / BI_C)),
            "wall_s": float("nan"),
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    print("=" * 70, flush=True)
    print(" L_T patch L_T.2 sweep", flush=True)
    print(f"   Bi_T values: {BI_T_VALS}", flush=True)
    print(f"   alpha={ALPHA}, delta={DELTA}, Bi_c={BI_C}", flush=True)
    print(f"   N={N_GRID}, t_end={T_END}, n_save={N_SAVE}", flush=True)
    print(f"   per-sim timeout: {PER_SIM_TIMEOUT/60:.0f} min", flush=True)
    print(f"   workers: {len(BI_T_VALS)} (full fan-out)", flush=True)
    print("=" * 70, flush=True)

    t_total = time.perf_counter()
    results = []

    with mp.Pool(processes=len(BI_T_VALS)) as pool:
        async_results = {}
        for Bi_T in BI_T_VALS:
            ar = pool.apply_async(run_one, args=(Bi_T,))
            async_results[Bi_T] = (ar, time.perf_counter())
            print(f"   submitted Bi_T={Bi_T}", flush=True)

        remaining = set(BI_T_VALS)
        while remaining:
            done_now = []
            for Bi_T in remaining:
                ar, t_sub = async_results[Bi_T]
                if ar.ready():
                    try:
                        r = ar.get(timeout=1)
                        results.append(r)
                        if r["error"]:
                            print(f"   [done] Bi_T={Bi_T}: ERROR  "
                                  f"{r['error'][:60]}", flush=True)
                        else:
                            print(f"   [done] Bi_T={Bi_T}: "
                                  f"osc={r['is_oscillating']}  "
                                  f"xi_LCST={r['xi_LCST']:.4f}  "
                                  f"L_T={r['L_T']:.3f}  "
                                  f"wall={r['wall_s']/60:.1f}min",
                                  flush=True)
                    except Exception as e:
                        print(f"   [done] Bi_T={Bi_T}: GET-ERR  {e}",
                              flush=True)
                        results.append({
                            "Bi_T": float(Bi_T),
                            "is_oscillating": False,
                            "xi_LCST": float("nan"),
                            "L_T": float(np.sqrt(ALPHA / Bi_T)),
                            "L_c": float(np.sqrt(DELTA / BI_C)),
                            "wall_s": float("nan"),
                            "error": f"GET-ERR: {e}",
                        })
                    done_now.append(Bi_T)
                elif time.perf_counter() - t_sub > PER_SIM_TIMEOUT:
                    print(f"   [TIMEOUT] Bi_T={Bi_T} exceeded "
                          f"{PER_SIM_TIMEOUT/60:.0f} min, killing",
                          flush=True)
                    results.append({
                        "Bi_T": float(Bi_T),
                        "is_oscillating": False,
                        "xi_LCST": float("nan"),
                        "L_T": float(np.sqrt(ALPHA / Bi_T)),
                        "L_c": float(np.sqrt(DELTA / BI_C)),
                        "wall_s": float("nan"),
                        "error": "TIMEOUT",
                    })
                    done_now.append(Bi_T)

            for Bi_T in done_now:
                remaining.discard(Bi_T)

            # Incremental save
            np.savez(OUT / "sweep_partial.npz",
                     results=np.array(results, dtype=object),
                     completed=len(results),
                     total=len(BI_T_VALS),
                     alpha=ALPHA, delta=DELTA, Bi_c=BI_C)

            if remaining:
                time.sleep(5)

        pool.terminate()
        pool.join()

    elapsed = time.perf_counter() - t_total
    print(f"\n   total wall-clock: {elapsed/60:.1f} min", flush=True)

    # Aggregate
    valid = [r for r in results if r["is_oscillating"] and not np.isnan(r["xi_LCST"])]
    valid_with_front = [r for r in valid if 0 < r["xi_LCST"] < 1]

    print(f"\n   {len(valid)}/{len(BI_T_VALS)} oscillating", flush=True)
    print(f"   {len(valid_with_front)}/{len(BI_T_VALS)} with internal front "
          f"(0 < xi < 1)", flush=True)

    summary = {
        "results": results,
        "n_valid": len(valid),
        "n_valid_with_front": len(valid_with_front),
        "verdict": None,
    }

    if len(valid_with_front) >= 3:
        ratios_LT = [(1 - r["xi_LCST"]) / r["L_T"] for r in valid_with_front]
        ratios_Lc = [(1 - r["xi_LCST"]) / r["L_c"] for r in valid_with_front]
        ratios_min = [(1 - r["xi_LCST"]) / min(r["L_T"], r["L_c"])
                      for r in valid_with_front]

        m_LT = float(np.mean(ratios_LT))
        s_LT = float(np.std(ratios_LT))
        m_Lc = float(np.mean(ratios_Lc))
        s_Lc = float(np.std(ratios_Lc))
        m_min = float(np.mean(ratios_min))
        s_min = float(np.std(ratios_min))

        print(f"\n   (1-xi)/L_T          : {m_LT:.4f} +- {s_LT:.4f}  "
              f"(spread {s_LT/m_LT:.1%})", flush=True)
        print(f"   (1-xi)/L_c          : {m_Lc:.4f} +- {s_Lc:.4f}  "
              f"(spread {s_Lc/m_Lc:.1%})", flush=True)
        print(f"   (1-xi)/min(L_T,L_c) : {m_min:.4f} +- {s_min:.4f}  "
              f"(spread {s_min/m_min:.1%})", flush=True)
        print(f"   Phase D plateau     : {PHASE_D_PLATEAU:.4f}", flush=True)
        delta_plateau = (m_min - PHASE_D_PLATEAU) / PHASE_D_PLATEAU
        print(f"   delta vs plateau    : {delta_plateau:+.1%}", flush=True)

        pass_spread = (s_min / m_min) < 0.30
        pass_plateau = abs(delta_plateau) < 0.30
        if pass_spread and pass_plateau:
            verdict = "PASS"
        elif pass_spread:
            verdict = "PARTIAL_SPREAD_OK_PLATEAU_OFF"
        elif pass_plateau:
            verdict = "PARTIAL_PLATEAU_OK_SPREAD_HIGH"
        else:
            verdict = "FAIL"
        print(f"\n   verdict: {verdict}  "
              f"(spread<30%: {pass_spread}, "
              f"plateau<30%-offset: {pass_plateau})", flush=True)

        summary.update({
            "ratios_LT_mean": m_LT, "ratios_LT_std": s_LT,
            "ratios_Lc_mean": m_Lc, "ratios_Lc_std": s_Lc,
            "ratios_min_mean": m_min, "ratios_min_std": s_min,
            "delta_vs_plateau": delta_plateau,
            "pass_spread": pass_spread,
            "pass_plateau": pass_plateau,
            "verdict": verdict,
        })
    else:
        print(f"\n   FAIL: insufficient valid points ({len(valid_with_front)} "
              f"need >=3) — cannot assess L_T scaling", flush=True)
        summary["verdict"] = "FAIL_INSUFFICIENT_POINTS"

    np.savez(OUT / "sweep_final.npz",
             results=np.array(results, dtype=object),
             alpha=ALPHA, delta=DELTA, Bi_c=BI_C,
             phase_D_plateau=PHASE_D_PLATEAU,
             **{k: v for k, v in summary.items()
                if k not in ("results",) and not isinstance(v, (list, dict))})

    with open(OUT / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   saved {OUT / 'sweep_final.npz'}", flush=True)
    print(f"   saved {OUT / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
