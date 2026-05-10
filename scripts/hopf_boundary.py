#!/usr/bin/env python3
"""
hopf_boundary.py — 0D linear-stability Hopf onset on a 2D parameter grid.

For each grid point we:
  1. Find the spatially-uniform steady state (J0, u0, θ0) of the
     volume-averaged ODE system.  When several SS branches exist
     (collapsed/middle/swollen), we pick the one closest to the working
     point — i.e. the largest-J (cold-swollen) branch — because that is
     the branch the full PDE actually sits on at the working point.
  2. Build the 3×3 Jacobian A₀(J0, u0, θ0) (k=0 piece of the dispersion
     relation).
  3. Diagonalize A₀ and split eigenvalues into (a) the largest-Re among
     complex-conjugate pairs (the Hopf indicator) and (b) the largest-Re
     among real eigenvalues (the saddle-node indicator).

The contour `re_max_complex = 0` is the *analytical* 0D Hopf onset.
Plotted on top of the nonlinear regime map (fig 4), it tells whether
the regime boundaries the PDE finds align with the linear prediction.

Caches to data/fig4/hopf_boundary_<param_x>_<param_y>.npz so the
boundary is recomputed only once per axis pair.
"""
import os
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
from scipy.linalg import eigvals

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from linear_stability_1d import LSAParams, find_uniform_ss, build_A0
from fig2_data import WORKING_POINT


DATA_DIR = (_HERE.parent / "data" / "fig4").resolve()


def _swollen_branch(ss_list):
    """Pick the largest-J SS (cold-swollen branch).

    `find_uniform_ss` returns SS sorted by J ascending; the swollen
    branch is the last entry. If no SS exists return None.
    """
    if not ss_list:
        return None
    return ss_list[-1]


def hopf_grid(param_x, x_vals, param_y, y_vals, base_overrides=None,
              cache_suffix=""):
    """Compute the 0D Hopf indicator on a (param_y, param_x) grid.

    Returns a dict:
      x, y                       — axis values
      re_max_complex (ny, nx)    — max Re among complex-conjugate evals
      re_max_real    (ny, nx)    — max Re among real evals
      omega          (ny, nx)    — Im part of the dominant complex eval
      J0, theta0     (ny, nx)    — swollen-branch SS coordinates
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{cache_suffix}" if cache_suffix else ""
    cache = DATA_DIR / f"hopf_boundary_{param_x}_{param_y}{suffix}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        out.update(dict(param_x=param_x, param_y=param_y))
        print(f"  Using cache: {cache}")
        return out

    NX, NY = len(x_vals), len(y_vals)
    re_max_complex = np.full((NY, NX), np.nan)
    re_max_real    = np.full((NY, NX), np.nan)
    omega          = np.full((NY, NX), np.nan)
    J0g            = np.full((NY, NX), np.nan)
    theta0g        = np.full((NY, NX), np.nan)

    base = {**WORKING_POINT, **(base_overrides or {})}
    # Convert WORKING_POINT keys → LSAParams kwargs (LSAParams is a subset)
    lsa_keys = {f.name for f in LSAParams.__dataclass_fields__.values()}
    base_lsa = {k: v for k, v in base.items() if k in lsa_keys}
    p_base = LSAParams(**base_lsa)

    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            pp = replace(p_base, **{param_x: float(xv),
                                     param_y: float(yv)})
            ss_list = find_uniform_ss(pp)
            ss = _swollen_branch(ss_list)
            if ss is None:
                continue
            J0, u0, theta0 = ss
            A = build_A0(J0, u0, theta0, pp)
            evs = eigvals(A)
            imag_nz = np.abs(evs.imag) > 1e-8
            re_c = evs.real[imag_nz] if imag_nz.any() else np.array([-np.inf])
            re_r = evs.real[~imag_nz] if (~imag_nz).any() else np.array([-np.inf])
            re_max_complex[j, i] = float(np.max(re_c))
            re_max_real[j, i]    = float(np.max(re_r))
            J0g[j, i]            = J0
            theta0g[j, i]        = theta0
            if imag_nz.any():
                evs_c = evs[imag_nz]
                idx_c = int(np.argmax(evs_c.real))
                omega[j, i] = float(np.abs(evs_c[idx_c].imag))

    np.savez_compressed(cache, x=x_vals, y=y_vals,
                        re_max_complex=re_max_complex,
                        re_max_real=re_max_real,
                        omega=omega, J0=J0g, theta0=theta0g)
    print(f"  Saved: {cache}")
    out = dict(x=x_vals, y=y_vals,
               re_max_complex=re_max_complex,
               re_max_real=re_max_real,
               omega=omega, J0=J0g, theta0=theta0g,
               param_x=param_x, param_y=param_y)
    return out


def main():
    """Compute Hopf grids on the same axes as fig4_data.build_grid."""
    from fig4_data import BI_T_VALS, S_CHI_VALS, DA_VALS

    print("=== 0D Hopf grid: Bi_T × S_chi ===")
    g_main = hopf_grid("Bi_T", BI_T_VALS, "S_chi", S_CHI_VALS)
    n_hopf = int(np.sum(g_main["re_max_complex"] > 0))
    print(f"  Hopf-unstable cells: {n_hopf}/{g_main['re_max_complex'].size}")

    print("\n=== 0D Hopf grid: Bi_T × Da ===")
    g_da = hopf_grid("Bi_T", BI_T_VALS, "Da", DA_VALS)
    n_hopf = int(np.sum(g_da["re_max_complex"] > 0))
    print(f"  Hopf-unstable cells: {n_hopf}/{g_da['re_max_complex'].size}")


if __name__ == "__main__":
    main()
