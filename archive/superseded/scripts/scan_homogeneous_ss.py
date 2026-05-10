#!/usr/bin/env python3
"""
scan_homogeneous_ss.py — RETIRED.  Superseded by
compute_homogeneous_ss_analytic.py (closed-form 1D root-count using
the explicit (Bi_T, S_chi) <- (J, theta) map; ~0.4 s vs. ~60 s,
captures the hot-runaway swollen branch that fsolve initial guesses
missed).  Kept only for reproducibility of the original fsolve cache.

Original docstring follows.

scan_homogeneous_ss.py — C1: enumerate uniform steady-state branches of
the LSA system on the (Bi_T, S_chi) plane.

For each grid point we solve
  mu(J_0, theta_0)            = mu_b
  Bi_c (1 - u_0)              = Da J_0 R(u_0, theta_0, J_0)
  Bi_T theta_0                = Da J_0 R(u_0, theta_0, J_0)
(eq:lsa_ss in main.tex) using `linear_stability_1d.find_uniform_ss`,
then classify the cell by which branches exist:

    0 = no branch found
    1 = swollen only        (single root with J >= J_THRESH)
    2 = collapsed only      (single root with J < J_THRESH)
    3 = swollen + collapsed (two stable roots)
    4 = three or more roots (e.g., with an unstable middle)

This map answers a paper-level question: where does the swollen
homogeneous SS cease to exist?  The "C-cell" region is precisely
where it has ceased to exist *but the PDE still oscillates* (the
overlay with the PDE regime map is added by the renderer).

Cache: data/fig5/homogeneous_ss_grid.npz
  Bi_T_vals, S_chi_vals       : axes
  n_roots                     : (NT, NS) integer count of roots
  swollen_J, swollen_theta    : (NT, NS) largest-J branch (NaN if absent)
  collapsed_J, collapsed_theta: (NT, NS) smallest-J branch (NaN if absent)
  has_swollen, has_collapsed  : boolean masks
  classification              : (NT, NS) integer 0..4 per legend above
  meta                        : working-point parameters used as base

Cost: ~1 minute for a 60x60 grid (no PDE; pure root-finding).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from linear_stability_1d import LSAParams, find_uniform_ss, m_bath  # noqa
from fig2_data import WORKING_POINT

DATA_DIR = _HERE.parent / "data" / "fig5"

J_THRESH = 0.55  # split between swollen (J >= J_THRESH) and collapsed branches


def _build_lsa_params(Bi_T: float, S_chi: float) -> LSAParams:
    """Construct an LSAParams matching WORKING_POINT but with overrides."""
    p = LSAParams(
        phi_p0=WORKING_POINT["phi_p0"],
        chi_inf=WORKING_POINT["chi_inf"],
        S_chi=S_chi,
        chi1=WORKING_POINT["chi1"],
        Omega_e=WORKING_POINT["Omega_e"],
        ell=WORKING_POINT["ell"],
        Da=WORKING_POINT["Da"],
        delta=WORKING_POINT["delta"],
        alpha=WORKING_POINT["alpha"],
        Gamma_A=WORKING_POINT["Gamma_A"],
        eps_T=WORKING_POINT["eps_T"],
        Bi_mu=WORKING_POINT["Bi_mu"],
        Bi_c=WORKING_POINT["Bi_c"],
        Bi_T=Bi_T,
        m_act=WORKING_POINT["m_act"],
        m_diff=WORKING_POINT["m_diff"],
        m_mob=WORKING_POINT["m_mob"],
        M0=1.0,
        D0=WORKING_POINT["D0"],
        C0=1.0,
        K0=1.0,
        J_init=WORKING_POINT.get("J_init", 1.30),
        theta_init=WORKING_POINT.get("theta_init", 0.0),
    )
    return p


def classify(roots) -> tuple[int, int, np.ndarray]:
    """Given list of (J0, u0, theta0), return (classification, n, root array).

    classification:
        0 = no branch
        1 = swollen only
        2 = collapsed only
        3 = swollen + collapsed (two branches)
        4 = three or more branches
    """
    if not roots:
        return 0, 0, np.zeros((0, 3))
    arr = np.array([(J, u, th) for J, u, th in roots])
    Js = arr[:, 0]
    n_swollen   = int(np.sum(Js >= J_THRESH))
    n_collapsed = int(np.sum(Js <  J_THRESH))
    n_total     = len(roots)
    if n_total >= 3:
        return 4, n_total, arr
    if n_swollen >= 1 and n_collapsed >= 1:
        return 3, n_total, arr
    if n_swollen >= 1:
        return 1, n_total, arr
    if n_collapsed >= 1:
        return 2, n_total, arr
    return 0, n_total, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_BiT", type=int, default=60)
    ap.add_argument("--n_Schi", type=int, default=60)
    ap.add_argument("--BiT_min", type=float, default=0.035)
    ap.add_argument("--BiT_max", type=float, default=0.40)
    ap.add_argument("--Schi_min", type=float, default=0.0)
    ap.add_argument("--Schi_max", type=float, default=2.10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "homogeneous_ss_grid.npz"
    if cache_path.exists() and not args.force:
        print(f"  Cache already present: {cache_path}")
        print(f"  Re-run with --force to overwrite.")
        return

    Bi_T_vals = np.linspace(args.BiT_min, args.BiT_max, args.n_BiT)
    S_chi_vals = np.linspace(args.Schi_min, args.Schi_max, args.n_Schi)
    NT, NS = len(Bi_T_vals), len(S_chi_vals)

    print(f"=== C1: homogeneous SS branch enumeration ===")
    print(f"  Bi_T : [{args.BiT_min:.3f}, {args.BiT_max:.3f}], n={NT}")
    print(f"  S_chi: [{args.Schi_min:.3f}, {args.Schi_max:.3f}], n={NS}")
    print(f"  Total points: {NT * NS}")
    print()

    swollen_J     = np.full((NT, NS), np.nan)
    swollen_theta = np.full((NT, NS), np.nan)
    swollen_u     = np.full((NT, NS), np.nan)
    collapsed_J   = np.full((NT, NS), np.nan)
    collapsed_theta = np.full((NT, NS), np.nan)
    collapsed_u     = np.full((NT, NS), np.nan)
    n_roots       = np.zeros((NT, NS), dtype=np.int8)
    cls_grid      = np.zeros((NT, NS), dtype=np.int8)

    t0 = time.perf_counter()
    for it, Bi_T in enumerate(Bi_T_vals):
        for js, S_chi in enumerate(S_chi_vals):
            p = _build_lsa_params(Bi_T, S_chi)
            roots = find_uniform_ss(p)
            roots = [] if roots is None else roots
            cls, n, arr = classify(roots)
            cls_grid[it, js] = cls
            n_roots[it, js] = n
            if arr.size:
                Js = arr[:, 0]
                # Swollen branch = largest J root (only count if >= J_THRESH)
                idx_max = int(np.argmax(Js))
                if Js[idx_max] >= J_THRESH:
                    swollen_J[it, js]     = Js[idx_max]
                    swollen_u[it, js]     = arr[idx_max, 1]
                    swollen_theta[it, js] = arr[idx_max, 2]
                # Collapsed branch = smallest J root (count if < J_THRESH)
                idx_min = int(np.argmin(Js))
                if Js[idx_min] < J_THRESH:
                    collapsed_J[it, js]     = Js[idx_min]
                    collapsed_u[it, js]     = arr[idx_min, 1]
                    collapsed_theta[it, js] = arr[idx_min, 2]
        if (it + 1) % 10 == 0 or it == 0:
            elapsed = time.perf_counter() - t0
            print(f"  row {it+1:>2d}/{NT}  ({elapsed:.0f}s elapsed)")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    # Summary statistics
    pct = np.round(100 * np.bincount(cls_grid.ravel(),
                                     minlength=5) / (NT * NS), 1)
    print(f"  classification breakdown:")
    print(f"    no roots         : {pct[0]:.1f}%")
    print(f"    swollen only     : {pct[1]:.1f}%")
    print(f"    collapsed only   : {pct[2]:.1f}%")
    print(f"    swollen+collapsed: {pct[3]:.1f}%")
    print(f"    >= 3 roots       : {pct[4]:.1f}%")

    np.savez_compressed(
        cache_path,
        Bi_T_vals=Bi_T_vals, S_chi_vals=S_chi_vals,
        swollen_J=swollen_J, swollen_theta=swollen_theta, swollen_u=swollen_u,
        collapsed_J=collapsed_J, collapsed_theta=collapsed_theta,
        collapsed_u=collapsed_u,
        n_roots=n_roots, classification=cls_grid,
        J_THRESH=J_THRESH,
        meta=dict(WORKING_POINT),
    )
    print(f"  Saved: {cache_path}")


if __name__ == "__main__":
    main()
