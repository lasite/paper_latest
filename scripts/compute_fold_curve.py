#!/usr/bin/env python3
"""
compute_fold_curve.py — Semi-analytical saddle-node curve in (Bi_T, S_chi).

The 0D homogeneous-SS system is
    F1(J, theta) = mu(J, theta; S_chi) - mu_b              = 0
    F2(J, theta; Bi_T) = Bi_T*theta - Da*J*R(u, theta, J)   = 0
with u = 1 - Bi_T*theta/Bi_c.  At any (J, theta) the two equations
solve EXPLICITLY for the *dependent* parameters:
    S_chi(J, theta) = [mu_b - mu_base(J, theta)] / (theta*phi**2)
    Bi_T(J, theta)  = K / [theta*(1 + K/Bi_c)],   K = Da*J*(1-phi)**m_act*T(theta)
where mu_base is mu with the S_chi-coupling term removed.  So (J, theta)
parameterizes the entire 0D-SS surface.  The saddle-node (fold) locus
in (Bi_T, S_chi) is the image, under this map, of the curve
    D(J, theta) := det[ d(F1, F2) / d(J, theta) ] = 0

We compute D on a fine (J, theta) grid, extract zero contours with
matplotlib's QuadContourGenerator, and map every contour point through
the explicit (J, theta) -> (Bi_T, S_chi) formulas.

Output: data/fig5/fold_curve_BiT_Schi.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fig2_data import WORKING_POINT
from linear_stability_1d import LSAParams, m_bath

DATA_DIR = _HERE.parent / "data" / "fig5"


def make_params() -> LSAParams:
    return LSAParams(
        phi_p0=WORKING_POINT["phi_p0"],
        chi_inf=WORKING_POINT["chi_inf"],
        S_chi=1.0,                            # placeholder; not used
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
        Bi_T=WORKING_POINT["Bi_T"],           # placeholder
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


def compute_field(JJ, TT, p, mu_b):
    """Vectorized (Bi_T, S_chi, D) over a (J, theta) mesh."""
    phi = p.phi_p0 / JJ
    one_m_phi = 1.0 - phi
    T_th = np.exp(p.Gamma_A * TT / (1.0 + p.eps_T * TT))
    K = p.Da * JJ * one_m_phi**p.m_act * T_th

    # Implicit Bi_T from F2=0
    Bi_T = K / (TT * (1.0 + K / p.Bi_c))
    # u back-substituted (equiv. Bi_c / (Bi_c + K))
    u = p.Bi_c / (p.Bi_c + K)

    # mu_base = mu without the S_chi*theta term
    mu_base = (
        np.log(one_m_phi) + phi
        + (p.chi_inf + p.chi1 * phi) * phi**2
        + p.Omega_e * (JJ - 1.0 / JJ)
    )
    # Implicit S_chi from F1=0
    S_chi = (mu_b - mu_base) / (TT * phi**2)

    # Partials at this (J, theta; S_chi, Bi_T)
    chi_eff = p.chi_inf + S_chi * TT
    dF1_dJ = (
        phi / (JJ * one_m_phi)                        # d ln(1-phi)/dJ
        - phi / JJ                                    # dphi/dJ
        - (phi**2 / JJ) * (2.0 * chi_eff + 3.0 * p.chi1 * phi)
        + p.Omega_e * (1.0 + 1.0 / JJ**2)
    )
    dF1_dth = S_chi * phi**2

    R = u * one_m_phi**p.m_act * T_th
    dR_dJ_at_u = R * p.m_act * phi / (JJ * one_m_phi)

    dF2_dJ = -p.Da * R - p.Da * JJ * dR_dJ_at_u
    dF2_dth = (
        Bi_T
        + p.Da * JJ * (R / u) * (Bi_T / p.Bi_c)
        - p.Da * JJ * R * p.Gamma_A / (1.0 + p.eps_T * TT)**2
    )

    D = dF1_dJ * dF2_dth - dF1_dth * dF2_dJ
    return Bi_T, S_chi, D


def extract_zero_contours(J_grid, th_grid, D_field):
    """Return list of (M, 2) arrays of (J, theta) along D=0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.contour(J_grid, th_grid, D_field.T, levels=[0.0])
    segments = []
    for level_segs in cs.allsegs:
        for seg in level_segs:
            if seg.shape[0] >= 2:
                segments.append(np.asarray(seg))
    plt.close(fig)
    return segments


def map_segments(segs_J_th, p, mu_b,
                 BiT_range=(0.030, 0.45), Schi_range=(0.0, 2.20)):
    """For each (J, theta) segment, compute (Bi_T, S_chi); split at clip-
    boundary so plotted curves don't connect through the off-grid range."""
    out = []
    for seg in segs_J_th:
        Js  = seg[:, 0]
        ths = seg[:, 1]
        phi = p.phi_p0 / Js
        one_m_phi = 1.0 - phi
        T_th = np.exp(p.Gamma_A * ths / (1.0 + p.eps_T * ths))
        K = p.Da * Js * one_m_phi**p.m_act * T_th
        BiT_seg  = K / (ths * (1.0 + K / p.Bi_c))
        mu_base = (
            np.log(one_m_phi) + phi
            + (p.chi_inf + p.chi1 * phi) * phi**2
            + p.Omega_e * (Js - 1.0 / Js)
        )
        Schi_seg = (mu_b - mu_base) / (ths * phi**2)

        ok = (
            (BiT_seg  >= BiT_range[0])  & (BiT_seg  <= BiT_range[1]) &
            (Schi_seg >= Schi_range[0]) & (Schi_seg <= Schi_range[1])
        )
        if not np.any(ok):
            continue
        # Split into runs of contiguous True
        idx = 0
        while idx < len(ok):
            while idx < len(ok) and not ok[idx]:
                idx += 1
            j0 = idx
            while idx < len(ok) and ok[idx]:
                idx += 1
            j1 = idx
            if j1 - j0 >= 2:
                out.append(np.column_stack([
                    BiT_seg[j0:j1], Schi_seg[j0:j1],
                    Js[j0:j1], ths[j0:j1],
                ]))
    return out


def main():
    p = make_params()
    mu_b = m_bath(p)

    N_J, N_T = 800, 800
    J_grid  = np.linspace(p.phi_p0 * 1.02, 2.0, N_J)
    th_grid = np.linspace(0.05, 18.0, N_T)
    JJ, TT  = np.meshgrid(J_grid, th_grid, indexing="ij")

    print("=== fold curve (semi-analytical) ===")
    print(f"  J     [{J_grid[0]:.4f}, {J_grid[-1]:.4f}]   N={N_J}")
    print(f"  theta [{th_grid[0]:.4f}, {th_grid[-1]:.4f}]   N={N_T}")
    print(f"  mu_b = {mu_b:.6f}  (J_init={p.J_init}, theta_init={p.theta_init})")

    t0 = time.perf_counter()
    Bi_T_field, S_chi_field, D_field = compute_field(JJ, TT, p, mu_b)
    print(f"  field: {time.perf_counter()-t0:.2f}s")

    # NaN-out points where the formulas are ill-conditioned (phi >= 1, etc.)
    # which can happen if J <= phi_p0
    bad = ~np.isfinite(D_field)
    if bad.any():
        D_field = np.where(bad, np.nan, D_field)
        print(f"  {bad.sum()} ill-conditioned grid pts NaN-ed")

    segs_J_th = extract_zero_contours(J_grid, th_grid, D_field)
    print(f"  D=0 contour segments in (J, theta): {len(segs_J_th)}")

    segs_param = map_segments(segs_J_th, p, mu_b)
    n_pts = sum(s.shape[0] for s in segs_param)
    print(f"  segments after clipping to render range: {len(segs_param)} "
          f"({n_pts} points)")

    if segs_param:
        lengths = np.array([s.shape[0] for s in segs_param], dtype=np.int64)
        flat = np.concatenate(segs_param, axis=0)
    else:
        lengths = np.array([], dtype=np.int64)
        flat = np.zeros((0, 4))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "fold_curve_BiT_Schi.npz"
    np.savez_compressed(
        cache,
        flat=flat, lengths=lengths,
        J_grid=J_grid, theta_grid=th_grid,
        Bi_T_field=Bi_T_field, S_chi_field=S_chi_field,
        D_field=D_field, mu_b=mu_b,
    )
    print(f"  saved: {cache}")


if __name__ == "__main__":
    main()
