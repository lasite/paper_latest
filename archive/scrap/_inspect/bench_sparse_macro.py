#!/usr/bin/env python3
"""
bench_sparse_macro.py — Compare macroscopic observables between the
old (cached) dense run and a fresh sparse-path run, both at N=41.

Pointwise relative differences in u can spuriously look enormous because
u oscillates between ~1e-12 and ~1, so a tiny phase shift gives ~100%
relative error on near-zero samples. The physically meaningful check is
whether ξ_peak, ξ_LCST, surface period, and amplitude all match.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from fig2_data import WORKING_POINT, T_START, T_END
from fig3_data import derived_from_arrays
from scan_optimized import Params, simulate


def macros(t, J, u, theta, x, p_dict):
    idx = (t >= T_START) & (t <= T_END)
    surf = J[-1, idx]
    surf_amp = float(surf.max() - surf.min())
    s = surf - surf.mean()
    pk, _ = find_peaks(s, prominence=0.1 * surf_amp, distance=3)
    tt = t[idx]
    period = float(np.median(np.diff(tt[pk]))) if len(pk) >= 2 else float("nan")
    d = derived_from_arrays(x, J[:, idx], np.maximum(u[:, idx], 1e-12),
                             theta[:, idx], p_dict)
    return dict(
        xi_peak=float(d["xi_peak"]),
        xi_LCST=float(d["xi_LCST"]),
        halo=float(d["xi_LCST"] - d["xi_peak"]),
        surf_amp=surf_amp,
        period=period,
        n_peaks=int(len(pk)),
        theta_mean=float(theta[:, idx].mean()),
        theta_max=float(theta[:, idx].max()),
        J_max=float(J[:, idx].max()),
        J_min=float(J[:, idx].min()),
    )


def main():
    cache = _HERE.parent.parent / "data" / "fig2" / "cache.npz"
    z = np.load(cache)
    p_dict = dict(WORKING_POINT)
    p_dict["N"] = int(z["J"].shape[0])
    print(f"  Old cache: N={p_dict['N']}")
    m_old = macros(z["t"], z["J"], z["u"], z["theta"], z["x"], p_dict)

    p = Params(**dict(WORKING_POINT))  # N=41 default
    res = simulate(p)
    m_new = macros(res["t"], res["J"], res["u"], res["theta"], res["x"],
                   dict(WORKING_POINT))

    print(f"\n  {'metric':>14s}  {'old (dense)':>13s}  {'new (sparse)':>13s}  "
          f"{'abs diff':>10s}  {'rel %':>8s}")
    for k in ["xi_peak", "xi_LCST", "halo", "surf_amp", "period",
              "n_peaks", "theta_mean", "theta_max", "J_max", "J_min"]:
        a, b = m_old[k], m_new[k]
        diff = abs(a - b)
        rel = diff / max(abs(a), abs(b), 1e-12) * 100.0
        print(f"  {k:>14s}  {a:>13.5g}  {b:>13.5g}  {diff:>10.4g}  {rel:>7.2f}%")


if __name__ == "__main__":
    main()
