#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_lt_patch_fallbackB_feasibility.py — Fallback B feasibility test.

L_T.2 (alpha=0.04, delta=0.50) yielded only 1/4 oscillating: cycle window
is too narrow in Bi_T when alpha is shrunk. Try Fallback B: keep alpha
at default 0.20 but inflate delta to 2.00. This keeps thermal dynamics
normal while pushing L_c well above L_T.

  alpha=0.20, delta=2.00, Bi_c=0.70 (Bi_T=0.10 test point)
  L_c = sqrt(2.00/0.70) = 1.690
  L_T = sqrt(0.20/0.10) = 1.414  (< L_c)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "lt_patch"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = Params(
        Bi_T=0.10, S_chi=1.0,
        alpha=0.20,
        delta=2.00,
        Bi_c=0.70,
        N=301,
        t_end=400.0,
        n_save=4000,
    )
    p = finalize_params(p)

    print("=" * 70, flush=True)
    print(" L_T patch Fallback B feasibility test", flush=True)
    print(f"   Bi_T={p.Bi_T}, alpha={p.alpha}, delta={p.delta}, Bi_c={p.Bi_c}",
          flush=True)
    print(f"   N={p.N}, t_end={p.t_end}", flush=True)
    L_T = float(np.sqrt(p.alpha / p.Bi_T))
    L_c = float(np.sqrt(p.delta / p.Bi_c))
    print(f"   L_T={L_T:.4f}, L_c={L_c:.4f}, L_T/L_c={L_T/L_c:.3f}",
          flush=True)
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    r = simulate(p)
    wall = time.perf_counter() - t0
    print(f"\n   wall-clock: {wall:.1f}s ({wall/60:.2f} min)", flush=True)

    t = r["t"]
    x = r["x"]
    J = r["J"]
    theta = r["theta"]
    phi = r["phi"]

    mask = t >= 200
    J_surf = J[-1, mask]
    theta_surf = theta[-1, mask]
    phi_max = phi[:, mask].max(axis=1)

    is_oscillating = bool(np.std(J_surf) > 0.05)
    J_surf_peak = float(np.max(J_surf))
    J_surf_trough = float(np.min(J_surf))

    crossing = np.where(phi_max > 0.5)[0]
    if len(crossing) > 0:
        xi_LCST = float(x[crossing[0]])
    else:
        xi_LCST = 1.0
    if (phi_max > 0.5).all():
        xi_LCST = 0.0

    print(f"\n   is_oscillating: {is_oscillating}", flush=True)
    print(f"   J_surf range  : [{J_surf_trough:.3f}, {J_surf_peak:.3f}]  "
          f"std={np.std(J_surf):.4f}", flush=True)
    print(f"   xi_LCST       : {xi_LCST:.4f}", flush=True)
    print(f"   (1 - xi_LCST) : {1-xi_LCST:.4f}", flush=True)
    print(f"   (1-xi)/L_T    : {(1-xi_LCST)/L_T:.4f}", flush=True)
    print(f"   (1-xi)/L_c    : {(1-xi_LCST)/L_c:.4f}", flush=True)
    print(f"   (1-xi)/min(L_T,L_c) : {(1-xi_LCST)/min(L_T,L_c):.4f}",
          flush=True)

    if is_oscillating and 0 < xi_LCST < 1:
        verdict = "PASS"
    elif not is_oscillating:
        verdict = "FAIL_NOT_OSCILLATING"
    elif xi_LCST == 0:
        verdict = "FAIL_HOT_RUNAWAY"
    elif xi_LCST == 1:
        verdict = "FAIL_COLD_SS"
    else:
        verdict = "FAIL_UNKNOWN"
    print(f"\n   verdict       : {verdict}", flush=True)

    np.savez(OUT / "fallbackB_feasibility.npz",
             is_oscillating=is_oscillating,
             J_surf_peak=J_surf_peak,
             J_surf_trough=J_surf_trough,
             xi_LCST=xi_LCST,
             L_T=L_T, L_c=L_c,
             Bi_T=p.Bi_T, alpha=p.alpha, delta=p.delta, Bi_c=p.Bi_c,
             N=p.N, t_end=p.t_end,
             wall_s=wall,
             verdict=verdict,
             t=t, x=x, J=J, theta=theta, phi=phi)
    print(f"\n   saved {OUT / 'fallbackB_feasibility.npz'}", flush=True)


if __name__ == "__main__":
    main()
