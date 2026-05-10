#!/usr/bin/env python3
"""
xi_LCST_N_convergence_lite.py — Sequential N convergence (skips N=161
which hung the parallel version). Writes to the same cache path as
the parallel version so xi_LCST_universal.py picks it up.
"""
import os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fig2_data import WORKING_POINT, T_START, T_END
from fig3_data import derived_from_arrays


N_VALS = [21, 41, 61, 81]


def run_one(N):
    from scan_optimized import Params, simulate
    p_dict = dict(WORKING_POINT)
    p_dict["N"] = int(N)
    # Shorter run than original to keep total time manageable
    p_dict["t_end"] = 220.0
    p_dict["n_save"] = 2200
    print(f"  N={N} ...", end=" ", flush=True)
    p = Params(**p_dict)
    result = simulate(p)
    t = result["t"]; J = result["J"]
    u = np.maximum(result["u"], 1e-12)
    theta = result["theta"]; x = result["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        # use a more permissive window
        idx = t >= 0.5 * t.max()
    surf_amp = float(J[-1, idx].max() - J[-1, idx].min())
    if surf_amp < 0.20:
        print(f"not oscillating (amp={surf_amp:.3f}); skipped")
        return None
    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], p_dict)
    print(f"xi_peak={d['xi_peak']:.5f}  xi_LCST={d['xi_LCST']:.5f}")
    return dict(N=int(N), dx=1.0/int(N),
                xi_peak=float(d["xi_peak"]),
                xi_LCST=float(d["xi_LCST"]),
                surf_amp=surf_amp)


def main():
    DATA_DIR = (_HERE.parent / "data" / "fig3").resolve()
    cache = DATA_DIR / "xi_LCST_N_convergence.npz"

    print(f"  Sequential N convergence; N ∈ {N_VALS}")
    results = []
    for N in N_VALS:
        r = run_one(N)
        if r is not None:
            results.append(r)

    if not results:
        print("  No successful runs.")
        return

    Ns  = np.array([r["N"]       for r in results])
    dxs = np.array([r["dx"]      for r in results])
    xpk = np.array([r["xi_peak"] for r in results])
    xlc = np.array([r["xi_LCST"] for r in results])
    np.savez_compressed(cache,
                        N=Ns, dx=dxs, xi_peak=xpk, xi_LCST=xlc)
    print(f"\n  Saved: {cache}")
    print(f"  {'N':>4} {'dx':>7} {'xi_peak':>9} {'xi_LCST':>9}")
    for N, dx, xp, xl in zip(Ns, dxs, xpk, xlc):
        print(f"  {N:>4} {dx:>7.4f} {xp:>9.5f} {xl:>9.5f}")


if __name__ == "__main__":
    main()
