#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fold_solver.py — F-H-R chemical-potential surface fold solver.

Computes the two saddle-node folds of the slow manifold μ(J, θ) = μ_b
in the (θ, J) plane:
    θ_lo, θ_up   — fold temperatures
    J_lo*, J_up* — J at folds (where ∂μ/∂J = 0)
    J_swollen_at_up, J_collapsed_at_lo — branch endpoints (limit-cycle extrema)
    Δθ, ΔJ, ln(θ_up/θ_lo)

These quantities are PURELY thermodynamic — they depend on
(S_chi, chi_inf, chi_1, Omega_e, phi_p0) only, NOT on Da, Bi_T, Γ_A.

The bath chemical potential μ_b is taken from m_bath(p) (i.e. fixed by the
initial cold-swollen reference state J_init, theta_init=0), matching the
PDE simulation in scan_optimized.py.

Algorithm
---------
At fixed S_chi, the slow-manifold equation μ(J,θ)=μ_b is LINEAR in θ
(since χ_eff = χ_∞ + S_χ·θ + χ_1·φ enters only through the χ·φ² term).
Therefore θ(J) is single-valued and explicit:
    θ(J) = (μ_b - μ_base(J)) / (S_χ · φ(J)²)
where μ_base = μ minus the S_χ·θ·φ² term. The folds of the (J,θ) S-curve
are the local extrema of θ(J):
    local max → upper fold (θ_up at J_up*)
    local min → lower fold (θ_lo at J_lo*)
We refine with fsolve on the simultaneous system μ=μ_b, ∂_Jμ=0.

Outputs
-------
data/iv_c/folds/default.npz       — single-point result at default Params()
data/iv_c/folds/S_chi_sweep.npz   — S_chi ∈ [0.3, 2.0] sweep
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from linear_stability_1d import LSAParams, chem_pot, df_dJ, m_bath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mu_base_of_J(J: np.ndarray, p: LSAParams) -> np.ndarray:
    """μ(J, θ=0) — chemical potential without the S_chi·θ·φ² coupling."""
    phi = p.phi_p0 / J
    one_m_phi = 1.0 - phi
    m_mix = np.log(one_m_phi) + phi + (p.chi_inf + p.chi1 * phi) * phi**2
    m_el = p.Omega_e * (J - 1.0 / J)
    return m_mix + m_el


def theta_of_J(J: np.ndarray, p: LSAParams, mu_b: float) -> np.ndarray:
    """Slow-manifold parameterization θ(J) at fixed S_chi (linear in θ)."""
    phi = p.phi_p0 / J
    return (mu_b - mu_base_of_J(J, p)) / (p.S_chi * phi**2)


# ---------------------------------------------------------------------------
# Fold finder
# ---------------------------------------------------------------------------

def find_folds(p: LSAParams,
               J_min_factor: float = 1.001,
               J_max: float = 5.0,
               N_J: int = 8000) -> dict:
    """
    Locate both folds of μ(J, θ) = μ_b in (θ, J).

    Returns dict with keys
        theta_lo, theta_up, J_lo_star, J_up_star,
        J_swollen_at_up, J_collapsed_at_lo,
        delta_theta, delta_J, log_theta_ratio, mu_b
    """
    mu_b = m_bath(p)

    # 1D scan of θ(J)
    J_grid = np.linspace(p.phi_p0 * J_min_factor, J_max, N_J)
    theta = theta_of_J(J_grid, p, mu_b)

    # Filter to physical range θ > 0  (cold-reference is the boundary θ=0)
    valid = np.isfinite(theta) & (theta > 0.0)
    if valid.sum() < 50:
        raise RuntimeError("θ(J) has too few positive points; check μ_b sign.")
    J_v = J_grid[valid]
    th_v = theta[valid]

    # Find local extrema of θ(J).  Use sign changes of finite differences.
    d_th = np.diff(th_v)
    s = np.sign(d_th)
    # Local max: s flips +→-  ;  local min: s flips -→+
    idx_max = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1   # local maxima
    idx_min = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1   # local minima

    if len(idx_max) == 0 or len(idx_min) == 0:
        # Monostable regime — no S-curve.
        raise RuntimeError(
            "Slow manifold has no fold (monostable); "
            f"S_chi={p.S_chi:.3f} likely too small to produce LCST bistability."
        )

    # Pick the highest θ-max and the lowest θ-min  (handles spurious extrema)
    i_up = idx_max[np.argmax(th_v[idx_max])]
    i_lo = idx_min[np.argmin(th_v[idx_min])]

    th_up_init, J_up_init = th_v[i_up], J_v[i_up]
    th_lo_init, J_lo_init = th_v[i_lo], J_v[i_lo]

    # ------------------------------------------------------------------
    # Refine via simultaneous μ=μ_b, ∂_Jμ=0
    # ------------------------------------------------------------------
    def fold_eqs(x):
        J, theta = x
        if J <= p.phi_p0 * 1.001 or J > 1.5 * J_max:
            return [1e6, 1e6]
        mu_val = float(chem_pot(np.atleast_1d(J), theta, p)[0]
                       if np.ndim(chem_pot(1.0, 0.0, p)) > 0
                       else chem_pot(J, theta, p))
        dmu = float(df_dJ(J, theta, p))
        return [mu_val - mu_b, dmu]

    sol_up, _, ier_up, _ = fsolve(fold_eqs, [J_up_init, th_up_init],
                                   full_output=True, xtol=1e-12)
    if ier_up != 1:
        raise RuntimeError(f"Upper-fold refinement failed (ier={ier_up}).")
    J_up_star, theta_up = float(sol_up[0]), float(sol_up[1])

    sol_lo, _, ier_lo, _ = fsolve(fold_eqs, [J_lo_init, th_lo_init],
                                   full_output=True, xtol=1e-12)
    if ier_lo != 1:
        raise RuntimeError(f"Lower-fold refinement failed (ier={ier_lo}).")
    J_lo_star, theta_lo = float(sol_lo[0]), float(sol_lo[1])

    # ------------------------------------------------------------------
    # Branch endpoints (limit-cycle J-extremes)
    # The limit cycle visits:
    #   ignition end:   (theta_up, J_swollen_at_up)  — top of swollen branch
    #   collapse jump:  →  (theta_up, J_collapsed_at_up)  — top of collapsed
    #   quench end:     (theta_lo, J_collapsed_at_lo) — bottom of collapsed
    #   reswell jump:   →  (theta_lo, J_swollen_at_lo)
    # By symmetry of the S-curve geometry:
    #   At θ = θ_up: swollen branch terminates at J_swollen_at_up = J_up_star
    #     (the merging of swollen + middle); collapsed branch is the third root.
    #   At θ = θ_lo: collapsed branch terminates at J_collapsed_at_lo = J_lo_star
    #     (merging of collapsed + middle); swollen branch is the third root.
    # However, the *amplitude* ΔJ of the limit cycle = swollen_J - collapsed_J
    # measured AT THE SAME θ. The natural measure is the J-jump at the fold:
    #   At θ_up: jump from J_up_star (swollen) → J_collapsed_at_up
    #   At θ_lo: jump from J_lo_star (collapsed) → J_swollen_at_lo
    # We report all four for downstream use.
    # ------------------------------------------------------------------

    def other_J_at_theta(target_theta: float, exclude_J: float,
                         tol_exclude: float = 0.1) -> float:
        """Find the OTHER root of θ(J)=target_theta away from exclude_J."""
        sign_diff = th_v - target_theta
        crossings = np.where(np.diff(np.sign(sign_diff)) != 0)[0]
        if len(crossings) == 0:
            return float("nan")
        # Refine each crossing
        Js_cross = []
        for k in crossings:
            J0, J1 = J_v[k], J_v[k+1]
            try:
                from scipy.optimize import brentq
                J_root = brentq(
                    lambda J: theta_of_J(np.array([J]), p, mu_b)[0] - target_theta,
                    J0, J1, xtol=1e-10
                )
                Js_cross.append(J_root)
            except ValueError:
                pass
        # Filter out roots close to exclude_J
        candidates = [J for J in Js_cross if abs(J - exclude_J) > tol_exclude]
        if not candidates:
            return float("nan")
        # The "other" root is the one farthest from exclude_J (on the opposite branch)
        if exclude_J > np.mean(Js_cross):
            return min(candidates)  # opposite branch is collapsed (small J)
        else:
            return max(candidates)  # opposite branch is swollen (large J)

    # Slightly inside the bistable window for numerical safety
    eps_th = 0.001 * (theta_up - theta_lo)
    J_collapsed_at_up = other_J_at_theta(theta_up - eps_th, J_up_star)
    J_swollen_at_lo   = other_J_at_theta(theta_lo + eps_th, J_lo_star)

    # If the "other root" calc fails, fall back to fold J's
    if not np.isfinite(J_collapsed_at_up):
        J_collapsed_at_up = J_lo_star
    if not np.isfinite(J_swollen_at_lo):
        J_swollen_at_lo = J_up_star

    # Δ J amplitude conventions
    #   delta_J        = full limit-cycle amplitude
    #                  = max_cycle J  -  min_cycle J
    #                  = J_swollen_at_lo (top of swollen, just after re-swell jump)
    #                  - J_collapsed_at_up (bottom of collapsed, just after collapse)
    #     This is what Phase A's measure_cycle_amplitude returns from PDE.
    #   delta_J_jump_up = J jump at upper fold   = J_up_star - J_collapsed_at_up
    #   delta_J_jump_lo = J jump at lower fold   = J_swollen_at_lo - J_lo_star
    delta_J         = float(J_swollen_at_lo - J_collapsed_at_up)
    delta_J_jump_up = float(J_up_star       - J_collapsed_at_up)
    delta_J_jump_lo = float(J_swollen_at_lo - J_lo_star)

    return dict(
        theta_lo=theta_lo,
        theta_up=theta_up,
        J_lo_star=J_lo_star,
        J_up_star=J_up_star,
        J_swollen_at_up=float(J_up_star),
        J_collapsed_at_lo=float(J_lo_star),
        J_collapsed_at_up=float(J_collapsed_at_up),
        J_swollen_at_lo=float(J_swollen_at_lo),
        delta_theta=float(theta_up - theta_lo),
        delta_J=delta_J,
        delta_J_jump_up=delta_J_jump_up,
        delta_J_jump_lo=delta_J_jump_lo,
        log_theta_ratio=float(np.log(theta_up / theta_lo)),
        mu_b=float(mu_b),
    )


# ---------------------------------------------------------------------------
# Main: default-point + S_chi sweep
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parents[1] / "data" / "iv_c" / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Default-parameter point ─────────────────────────────────────
    p_def = LSAParams()
    print("=" * 64)
    print(" Phase P — F-H-R fold solver")
    print("=" * 64)
    print(f"\nDefault params:  S_chi={p_def.S_chi}  chi_inf={p_def.chi_inf}"
          f"  chi1={p_def.chi1}  Omega_e={p_def.Omega_e}  phi_p0={p_def.phi_p0}")
    print(f"                 J_init={p_def.J_init}  theta_init={p_def.theta_init}")

    res = find_folds(p_def)
    print(f"\nμ_b               = {res['mu_b']:.6f}")
    print(f"θ_lo              = {res['theta_lo']:.4f}   (target ~0.85)")
    print(f"θ_up              = {res['theta_up']:.4f}   (target ~3.18)")
    print(f"Δθ = θ_up - θ_lo  = {res['delta_theta']:.4f} (target ~2.3)")
    print(f"ln(θ_up/θ_lo)     = {res['log_theta_ratio']:.4f}")
    print(f"J_lo* (collapsed) = {res['J_lo_star']:.4f}")
    print(f"J_up* (swollen)   = {res['J_up_star']:.4f}")
    print(f"J_collapsed_at_up = {res['J_collapsed_at_up']:.4f}")
    print(f"J_swollen_at_lo   = {res['J_swollen_at_lo']:.4f}")
    print(f"ΔJ (full cycle)   = {res['delta_J']:.4f}   (target ~1.2)")
    print(f"ΔJ jump @ θ_up    = {res['delta_J_jump_up']:.4f}")
    print(f"ΔJ jump @ θ_lo    = {res['delta_J_jump_lo']:.4f}")

    np.savez(out_dir / "default.npz", **res)
    print(f"\nSaved {out_dir/'default.npz'}")

    # ── S_chi sweep ─────────────────────────────────────────────────
    S_chi_vals = np.linspace(0.3, 2.0, 35)
    keys = [
        "theta_lo", "theta_up", "delta_theta", "delta_J",
        "delta_J_jump_up", "delta_J_jump_lo",
        "log_theta_ratio", "J_lo_star", "J_up_star",
        "J_collapsed_at_up", "J_swollen_at_lo", "mu_b",
    ]
    folds = {k: [] for k in keys}
    folds["S_chi"] = S_chi_vals
    n_ok = 0
    n_fail = 0

    print(f"\n{'-'*64}\nS_chi sweep (n = {len(S_chi_vals)}):")
    print(f"{'S_chi':>7}  {'theta_lo':>10}  {'theta_up':>10}  {'Δθ':>8}  {'ΔJ':>8}")

    for sc in S_chi_vals:
        p = replace(p_def, S_chi=float(sc))
        try:
            r = find_folds(p)
            for k in keys:
                folds[k].append(r[k])
            print(f"{sc:7.3f}  {r['theta_lo']:10.4f}  {r['theta_up']:10.4f}"
                  f"  {r['delta_theta']:8.4f}  {r['delta_J']:8.4f}")
            n_ok += 1
        except RuntimeError as e:
            for k in keys:
                folds[k].append(np.nan)
            print(f"{sc:7.3f}  -- monostable / {str(e)[:40]}")
            n_fail += 1

    for k in keys:
        folds[k] = np.array(folds[k])
    np.savez(out_dir / "S_chi_sweep.npz", **folds)
    print(f"\n{n_ok}/{len(S_chi_vals)} succeeded, {n_fail} monostable.")
    print(f"Saved {out_dir/'S_chi_sweep.npz'}")

    # ── Phase A success-criterion preview: h = Δθ · S_chi ──────────
    h_arr = folds["delta_theta"] * S_chi_vals
    h_finite = h_arr[np.isfinite(h_arr)]
    if h_finite.size:
        print(f"\nh = Δθ · S_chi   mean = {np.nanmean(h_arr):.3f}  "
              f"std = {np.nanstd(h_arr):.3f}  "
              f"(should be ~constant ≈ 2-3 if amplitude scaling holds)")

    # ── Cache the dense θ(J) curve at default for diagnostics ──────
    J_grid = np.linspace(p_def.phi_p0 * 1.001, 4.0, 4000)
    theta_curve = theta_of_J(J_grid, p_def, m_bath(p_def))
    np.savez(out_dir / "default_theta_J_curve.npz",
             J=J_grid, theta=theta_curve, mu_b=m_bath(p_def))
    print(f"Saved {out_dir/'default_theta_J_curve.npz'}")


if __name__ == "__main__":
    main()
