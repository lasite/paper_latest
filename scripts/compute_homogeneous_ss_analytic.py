#!/usr/bin/env python3
"""
compute_homogeneous_ss_analytic.py — analytic replacement for the
60x60 fsolve scan in scan_homogeneous_ss.py.

Strategy: at fixed S_chi*, the SS condition F1(J, theta) = 0 is linear
in S_chi and gives an EXPLICIT theta(J; S_chi*) without root-finding:
    theta(J; S_chi*) = [mu_b - mu_base(J)] / (S_chi* * phi(J)**2)
because mu_base does not depend on theta.  Substituting into the
heat balance F2 yields Bi_T_pred(J; S_chi*) — a single 1D function
of J.  Roots of the homogeneous-SS system at parameter pair
(Bi_T*, S_chi*) are exactly the J values where
    Bi_T_pred(J; S_chi*) = Bi_T*
i.e. sign changes of Bi_T_pred(J) - Bi_T*.  We classify each root
by J vs J_THRESH (swollen / collapsed).

Output: data/fig5/homogeneous_ss_analytic.npz
    Bi_T_vals, S_chi_vals : pixel centers
    classification        : 0 none, 1 swollen-only, 2 collapsed-only, 3 bistable
    n_roots               : total # of (J, theta) preimages per pixel
    has_swollen, has_collapsed : booleans
The original "5+ roots" category from the fsolve scan (~1% of cells in
the cusp tip) is folded into 'bistable' here.

Cost: ~3 s for 360 x 240 target with N_J = 2500.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from compute_fold_curve import make_params
from linear_stability_1d import m_bath

DATA_DIR = _HERE.parent / "data" / "fig5"

J_THRESH = 0.55


def main():
    p = make_params()
    mu_b = m_bath(p)

    # 1D J grid (per S_chi sweep). J_max big enough to cover hot-runaway
    # swollen branch (J ~ 5.2).  N_J set so segment-linearization error
    # is well below pixel resolution.
    N_J = 2500
    J_grid = np.linspace(p.phi_p0 * 1.01, 6.0, N_J)
    phi = p.phi_p0 / J_grid
    one_m_phi = 1.0 - phi
    mu_base_J = (
        np.log(one_m_phi) + phi
        + (p.chi_inf + p.chi1 * phi) * phi**2
        + p.Omega_e * (J_grid - 1.0 / J_grid)
    )
    inv_phi2 = 1.0 / phi**2
    J_mid = 0.5 * (J_grid[:-1] + J_grid[1:])
    swollen_seg = J_mid >= J_THRESH                # (N_J-1,)

    # Target grid in (Bi_T, S_chi)
    BiT_min, BiT_max = 0.035, 0.40
    Schi_min, Schi_max = 0.0, 2.10
    NBT, NSX = 360, 240
    BiT_edges  = np.linspace(BiT_min,  BiT_max,  NBT + 1)
    Schi_edges = np.linspace(Schi_min, Schi_max, NSX + 1)
    BiT_centers  = 0.5 * (BiT_edges[1:]  + BiT_edges[:-1])
    Schi_centers = 0.5 * (Schi_edges[1:] + Schi_edges[:-1])

    has_s = np.zeros((NBT, NSX), dtype=bool)
    has_c = np.zeros((NBT, NSX), dtype=bool)
    n_roots = np.zeros((NBT, NSX), dtype=np.int8)

    print("=== analytic homogeneous-SS (1D root-count per pixel) ===")
    print(f"  J grid: N={N_J}  J in [{J_grid[0]:.4f}, {J_grid[-1]:.4f}]")
    print(f"  target: {NBT} x {NSX} = {NBT*NSX:,} pixels")

    t0 = time.perf_counter()
    BiT_col = BiT_centers[:, None]            # (NBT, 1)

    THETA_MAX = 30.0   # cap to avoid Arrhenius overflow at J → phi_p0
    for js, S_chi_t in enumerate(Schi_centers):
        if S_chi_t <= 0:
            continue
        # theta(J; S_chi*) = (mu_b - mu_base(J)) / (S_chi* * phi**2)
        theta = (mu_b - mu_base_J) * inv_phi2 / S_chi_t
        valid = (theta > 0) & (theta < THETA_MAX)
        if not np.any(valid):
            continue
        # Bi_T_pred(J; S_chi*) — only where theta > 0 (compute safely)
        with np.errstate(over="ignore", invalid="ignore"):
            theta_safe = np.where(valid, theta, 1.0)   # placeholder
            T_th = np.exp(p.Gamma_A * theta_safe / (1.0 + p.eps_T * theta_safe))
            K = p.Da * J_grid * one_m_phi**p.m_act * T_th
            BiT_pred = K / (theta_safe * (1.0 + K / p.Bi_c))
        BiT_pred = np.where(valid, BiT_pred, np.nan)

        v0 = BiT_pred[:-1]; v1 = BiT_pred[1:]
        seg_valid = np.isfinite(v0) & np.isfinite(v1)
        lo = np.minimum(v0, v1)
        hi = np.maximum(v0, v1)

        # bracketed[ix, seg] = does target Bi_T*[ix] lie between v0, v1 of segment?
        bracketed = ((lo[None, :] <= BiT_col) & (BiT_col <= hi[None, :])
                     & seg_valid[None, :])
        # any swollen segment that brackets?
        has_s[:, js] = (bracketed & swollen_seg[None, :]).any(axis=1)
        has_c[:, js] = (bracketed & ~swollen_seg[None, :]).any(axis=1)
        # total # crossings = total # brackets per pixel (signed crossing
        # counted once per bracket since bracketed segments are non-empty)
        n_roots[:, js] = bracketed.sum(axis=1, dtype=np.int8)

    print(f"  total: {time.perf_counter()-t0:.2f}s")

    cls = np.zeros((NBT, NSX), dtype=np.int8)
    cls[has_s & ~has_c] = 1
    cls[~has_s & has_c] = 2
    cls[has_s & has_c]  = 3

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "homogeneous_ss_analytic.npz"
    np.savez_compressed(
        cache,
        Bi_T_vals=BiT_centers, S_chi_vals=Schi_centers,
        BiT_edges=BiT_edges, Schi_edges=Schi_edges,
        classification=cls,
        n_roots=n_roots,
        has_swollen=has_s, has_collapsed=has_c,
        N_J=N_J, J_max=6.0, J_THRESH=J_THRESH,
        mu_b=mu_b,
    )
    pct = np.bincount(cls.ravel(), minlength=4) / cls.size * 100
    print(f"  classification breakdown:")
    print(f"    no root        : {pct[0]:5.2f}%")
    print(f"    swollen only   : {pct[1]:5.2f}%")
    print(f"    collapsed only : {pct[2]:5.2f}%")
    print(f"    bistable       : {pct[3]:5.2f}%")
    print(f"  multiplicity tally:")
    for k in range(int(n_roots.max()) + 1):
        n = int((n_roots == k).sum())
        print(f"    {k} roots: {n:>6d} ({100*n/n_roots.size:5.2f}%)")
    print(f"  Saved: {cache}")


if __name__ == "__main__":
    main()
