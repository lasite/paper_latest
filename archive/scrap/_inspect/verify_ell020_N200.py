#!/usr/bin/env python3
"""
verify_ell020_N200.py — single-shot numerical verification at (ℓ=0.02, N=200).

Goal: confirm that the working point at the proposed "tighter" pair
(ℓ=0.02, dx=1/200=0.005, dx/ℓ=0.25) is numerically clean and produces
sensible ξ_peak, ξ_LCST, halo, surface oscillation period and amplitude.
This is the convergence check before declaring (ℓ=0.02, N=200) the new
canonical resolution.

Reference points (already cached):
  N=41,  ℓ=0.01:  ξ_peak=0.893, ξ_LCST=0.922 (working-point)
  N=41,  ℓ=0.02:  ξ_peak=0.830, ξ_LCST=0.896, halo=0.066, T≈8.5
  N=81,  ℓ=0.01:  ξ_peak=0.908, ξ_LCST=0.911

Expected at (N=200, ℓ=0.02): a clean halo of order ℓ (~0.02-0.05),
ξ_LCST close to the (N=41, ℓ=0.02) value (since ℓ is the physical
length, dx-refinement should NOT shift it once dx ≤ ℓ).
"""
import os
import sys
import time
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

DATA_DIR = (_HERE.parent.parent / "data" / "fig3").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
def _cache_path(ell: float, N: int) -> Path:
    return DATA_DIR / f"verify_ell{int(round(ell*1000)):03d}_N{N}.npz"


def run_one(ell: float, N: int, t_end: float = 250.0, n_save: int = 4000):
    p_dict = dict(WORKING_POINT)
    p_dict["ell"] = float(ell)
    p_dict["N"]   = int(N)
    p_dict["t_end"]  = float(t_end)
    p_dict["n_save"] = int(n_save)
    print(f"  Sim: ℓ={ell:.4f}, N={N}, dx={1.0/N:.5f}, dx/ℓ={1.0/(N*ell):.3f}")
    print(f"        t_end={t_end}, n_save={n_save}")
    t0 = time.perf_counter()
    p = Params(**p_dict)
    res = simulate(p)
    dt = time.perf_counter() - t0
    print(f"        simulate(): {dt:.1f}s, nfev={res.get('nfev')}")
    return p_dict, res


def diagnose(p_dict, res):
    t = res["t"]; J = res["J"]
    u = np.maximum(res["u"], 1e-12)
    theta = res["theta"]; x = res["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return None, "short_window"
    surf = J[-1, idx]
    surf_amp = float(surf.max() - surf.min())
    if surf_amp < 0.20:
        return None, f"not_oscillating (amp={surf_amp:.3f})"
    s = surf - surf.mean()
    pk, _ = find_peaks(s, prominence=0.1 * surf_amp, distance=3)
    tt = t[idx]
    period = float(np.median(np.diff(tt[pk]))) if len(pk) >= 2 else float("nan")
    n_peaks = int(len(pk))
    cv_period = (float(np.std(np.diff(tt[pk])) / np.mean(np.diff(tt[pk])))
                 if len(pk) >= 3 else float("nan"))

    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], p_dict)

    out = dict(
        N=p_dict["N"], ell=p_dict["ell"], dx=1.0 / p_dict["N"],
        xi_peak=float(d["xi_peak"]),
        xi_LCST=float(d["xi_LCST"]),
        xi_kin=float(d["xi_kin"]),
        halo=float(d["xi_LCST"] - d["xi_peak"]),
        surf_amp=surf_amp,
        period=period,
        n_peaks=n_peaks,
        cv_period=cv_period,
        theta_mean=float(theta[:, idx].mean()),
        theta_max=float(theta[:, idx].max()),
        theta_min=float(theta[:, idx].min()),
        J_min=float(J[:, idx].min()),
        J_max=float(J[:, idx].max()),
        u_min=float(u[:, idx].min()),
        u_max=float(u[:, idx].max()),
        J_eq=float(d["J_eq"]),
        # Convergence-relevant extras
        surf_J_mean=float(surf.mean()),
        core_J_mean=float(J[0, idx].mean()),
        core_theta_mean=float(theta[0, idx].mean()),
    )
    return out, "ok"


def main():
    print("=" * 60)
    print("verify_ell020_N200 — single-shot convergence check")
    print("=" * 60)

    ELL = float(os.environ.get("VERIFY_ELL", "0.02"))
    NV  = int(os.environ.get("VERIFY_N",   "81"))
    TE  = float(os.environ.get("VERIFY_T_END", "250.0"))
    NS  = int(os.environ.get("VERIFY_N_SAVE", "4000"))
    p_dict, res = run_one(ell=ELL, N=NV, t_end=TE, n_save=NS)
    diag, status = diagnose(p_dict, res)
    print(f"  diagnose(): {status}")
    if diag is None:
        print("  ABORT: simulation did not produce usable data.")
        return

    print()
    print("  ── Diagnostics (ℓ=0.02, N=200) ──")
    for k in ["N", "ell", "dx",
              "xi_peak", "xi_LCST", "xi_kin", "halo",
              "surf_amp", "period", "n_peaks", "cv_period",
              "theta_mean", "theta_max", "theta_min",
              "J_min", "J_max", "u_min", "u_max",
              "J_eq", "surf_J_mean", "core_J_mean", "core_theta_mean"]:
        v = diag[k]
        if isinstance(v, float):
            print(f"    {k:>16s} = {v:.6g}")
        else:
            print(f"    {k:>16s} = {v}")

    cache = _cache_path(diag["ell"], diag["N"])
    np.savez_compressed(
        cache,
        x=res["x"], t=res["t"],
        J=res["J"], u=res["u"], theta=res["theta"],
        **{k: np.asarray(v) for k, v in diag.items()},
    )
    print(f"\n  Saved arrays + diagnostics: {cache}")

    # ── Sanity comparisons ────────────────────────────────────────────
    print()
    print("  ── Reference (cached, for comparison) ──")
    ell_z = np.load(DATA_DIR / "ell_scan.npz")
    iN41_e02 = int(np.argmin(np.abs(ell_z["ell"] - 0.02)))
    if ell_z["ok"][iN41_e02]:
        print(f"    N=41, ℓ=0.020 (ell_scan):"
              f"  ξ_peak={ell_z['xi_peak'][iN41_e02]:.4f}"
              f"  ξ_LCST={ell_z['xi_LCST'][iN41_e02]:.4f}"
              f"  halo={ell_z['halo'][iN41_e02]:.4f}"
              f"  T={ell_z['period'][iN41_e02]:.2f}"
              f"  amp={ell_z['surf_amp'][iN41_e02]:.2f}")

    Nc = np.load(DATA_DIR / "xi_LCST_N_convergence.npz")
    print(f"    N convergence at default ℓ=0.01:")
    for N, dx, xp, xl in zip(Nc["N"], Nc["dx"], Nc["xi_peak"], Nc["xi_LCST"]):
        print(f"      N={int(N):>3d}  dx={dx:.5f}  ξ_peak={xp:.4f}  ξ_LCST={xl:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────
    print()
    print("  ── Verdict ──")
    halo = diag["halo"]
    rel = halo / diag["ell"]
    print(f"    halo / ℓ = {rel:.2f}    "
          f"(should be O(1) if halo is genuinely interface-controlled)")
    if diag["dx"] / diag["ell"] > 0.5:
        print("    WARNING: dx/ℓ > 0.5 — interface still under-resolved.")
    else:
        print(f"    dx/ℓ = {diag['dx']/diag['ell']:.2f}  (< 0.5 → resolved).")
    if diag["surf_amp"] > 0.2 and diag["n_peaks"] >= 3:
        print(f"    Surface oscillates cleanly: amp={diag['surf_amp']:.2f}, "
              f"{diag['n_peaks']} peaks, CV(T)={diag['cv_period']:.3f}")
    else:
        print("    Surface oscillation is weak/irregular.")


if __name__ == "__main__":
    main()
