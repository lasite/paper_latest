#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_stability_1d.py — Linear stability analysis for the 1D gel slab model.

Two complementary analyses:
  1. Analytical dispersion relation σ(k) around the spatially uniform steady state.
     The 3×3 eigenvalue problem A(k)·v = σ·v captures:
       - k=0: homogeneous (0D-equivalent) Hopf instability
       - k>0: diffusion-modified instabilities (Turing, traveling waves)
  2. Numerical eigenvalue analysis of the full discretized 3N×3N system.
     Finds the true (possibly non-uniform) steady state via time integration,
     then computes the Jacobian eigenvalues.

Physics: reaction-driven self-oscillation in LCST hydrogels
  State: (J, u, θ) — swelling ratio, reactant concentration, temperature
  Volume-averaged 0D equations (base state):
    dJ/dt  = -Bi_μ·(μ - μ_b)
    du/dt  = [-Bi_c·(u-1) - Da·J·R + u·Bi_μ·(μ-μ_b)] / J
    dθ/dt  = [-Bi_T·θ + Da·J·R] / C₀
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.linalg import eigvals
from dataclasses import dataclass, replace, field
from typing import Optional, Tuple, List, Dict


# ═══════════════════════════════════════════════════════════════════
# Parameters (mirroring scan_optimized.py Params)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LSAParams:
    """Dimensionless parameters for LSA. Defaults match 1D slab."""
    phi_p0: float = 0.15
    chi_inf: float = 0.60
    S_chi: float = 1.00
    chi1: float = 1.10
    Omega_e: float = 0.12
    ell: float = 0.01

    Da: float = 4.0
    delta: float = 0.08
    alpha: float = 0.20
    Gamma_A: float = 1.5
    eps_T: float = 0.03

    Bi_mu: float = 1.00
    Bi_c: float = 0.70
    Bi_T: float = 0.10

    m_act: float = 6.0
    m_diff: float = 2.0
    m_mob: float = 1.0
    M0: float = 1.0
    D0: float = 2.0
    C0: float = 1.0
    K0: float = 1.0

    J_init: float = 1.30
    theta_init: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Constitutive laws and their analytical derivatives
# ═══════════════════════════════════════════════════════════════════

def phi_of_J(J, p):
    return p.phi_p0 / J


def chem_pot(J, theta, p):
    """Local chemical potential f(J, θ)."""
    ph = phi_of_J(J, p)
    m_mix = np.log(1 - ph) + ph + (p.chi_inf + p.S_chi * theta + p.chi1 * ph) * ph**2
    m_el = p.Omega_e * (J - 1.0 / J)
    return m_mix + m_el


def m_bath(p):
    """Bath chemical potential from initial conditions."""
    return float(chem_pot(p.J_init, p.theta_init, p))


def thermal_factor(theta, p):
    """Arrhenius-type thermal activation T(θ)."""
    denom = 1.0 + p.eps_T * theta
    return np.exp(p.Gamma_A * theta / denom)


def reaction_rate(u, theta, J, p):
    """R(u, θ, J) = u · (1-φ)^m_act · T(θ)."""
    ph = phi_of_J(J, p)
    act = (1 - ph)**p.m_act
    return u * act * thermal_factor(theta, p)


# ─── Analytical derivatives ──────────────────────────────────────

def df_dJ(J, theta, p):
    """∂f/∂J — derivative of chemical potential w.r.t. J."""
    ph = phi_of_J(J, p)
    chi_eff = p.chi_inf + p.S_chi * theta
    # Each term computed from d/dJ with dφ/dJ = -φ/J:
    t1 = ph / (J * (1 - ph))               # d ln(1-φ)/dJ
    t2 = -ph / J                            # dφ/dJ
    t3 = -(ph**2 / J) * (2 * chi_eff + 3 * p.chi1 * ph)  # d[(χ+χ₁φ)φ²]/dJ
    t4 = p.Omega_e * (1 + 1.0 / J**2)      # d[Ω_e(J-1/J)]/dJ
    return t1 + t2 + t3 + t4


def df_dtheta(J, theta, p):
    """∂f/∂θ — derivative of chemical potential w.r.t. θ."""
    ph = phi_of_J(J, p)
    return p.S_chi * ph**2


def dR_du(u, theta, J, p):
    """∂R/∂u = R/u."""
    R0 = reaction_rate(u, theta, J, p)
    return R0 / u


def dR_dtheta(u, theta, J, p):
    """∂R/∂θ = R · Γ_A / (1+ε_T·θ)²."""
    R0 = reaction_rate(u, theta, J, p)
    denom = 1.0 + p.eps_T * theta
    return R0 * p.Gamma_A / denom**2


def dR_dJ(u, theta, J, p):
    """∂R/∂J = R · m_act · φ / [J(1-φ)]."""
    ph = phi_of_J(J, p)
    R0 = reaction_rate(u, theta, J, p)
    return R0 * p.m_act * ph / (J * (1 - ph))


# ═══════════════════════════════════════════════════════════════════
# Uniform steady state
# ═══════════════════════════════════════════════════════════════════

def find_uniform_ss(p, J_guess=None, theta_guess=None):
    """
    Find the spatially uniform steady state (J₀, u₀, θ₀).

    At steady state:
      μ(J₀, θ₀) = μ_b                           (solvent equilibrium)
      Bi_T·θ₀ = Da·J₀·R(u₀, θ₀, J₀)            (heat balance)
      u₀ = 1 - Bi_T·θ₀ / Bi_c                   (reactant balance)

    Returns (J₀, u₀, θ₀) or None if no valid solution found.
    """
    mu_b = m_bath(p)

    def equations(x):
        J, theta = x
        if J <= p.phi_p0 * 1.01 or J > 10.0 or theta < -0.1 or theta > 30.0:
            return [1e10, 1e10]
        u = 1.0 - p.Bi_T * theta / p.Bi_c
        if u <= 0 or u > 1.0:
            return [1e10, 1e10]
        eq1 = chem_pot(J, theta, p) - mu_b
        eq2 = p.Bi_T * theta - p.Da * J * reaction_rate(u, theta, J, p)
        return [eq1, eq2]

    # Try multiple initial guesses
    guesses = []
    if J_guess is not None and theta_guess is not None:
        guesses.append((J_guess, theta_guess))
    guesses += [
        (p.J_init, 0.1), (p.J_init, 1.0), (p.J_init, 3.0), (p.J_init, 5.0),
        (0.5, 0.5), (0.3, 1.0), (0.3, 5.0), (0.5, 3.0),
        (1.0, 0.5), (1.0, 2.0), (1.5, 0.1),
        (0.2, 2.0), (0.2, 5.0), (0.2, 10.0),
    ]

    solutions = []
    for Jg, tg in guesses:
        try:
            sol, info, ier, msg = fsolve(equations, [Jg, tg], full_output=True)
            if ier != 1:
                continue
            J0, theta0 = sol
            if J0 <= p.phi_p0 * 1.01 or theta0 < -0.01:
                continue
            u0 = 1.0 - p.Bi_T * theta0 / p.Bi_c
            if u0 <= 0 or u0 > 1.001:
                continue
            # Check residual
            res = np.linalg.norm(equations(sol))
            if res > 1e-8:
                continue
            # Dedup
            is_dup = False
            for s in solutions:
                if abs(J0 - s[0]) < 1e-4 and abs(theta0 - s[3]) < 1e-4:
                    is_dup = True
                    break
            if not is_dup:
                solutions.append((J0, u0, theta0, theta0, res))
        except Exception:
            continue

    if not solutions:
        return None

    # Return all unique solutions, sorted by J
    results = [(s[0], s[1], s[2]) for s in solutions]
    results.sort(key=lambda x: x[0])
    return results


# ═══════════════════════════════════════════════════════════════════
# Dispersion relation: A(k) = A₀ + k²·D₂ + k⁴·D₄
# ═══════════════════════════════════════════════════════════════════

def build_A0(J0, u0, theta0, p):
    """
    3×3 Jacobian of the volume-averaged (0D) model.
    State: (δJ, δu, δθ).
    """
    fJ = df_dJ(J0, theta0, p)
    ft = df_dtheta(J0, theta0, p)
    R0 = reaction_rate(u0, theta0, J0, p)
    Ru = dR_du(u0, theta0, J0, p)
    Rt = dR_dtheta(u0, theta0, J0, p)
    RJ = dR_dJ(u0, theta0, J0, p)

    A0 = np.zeros((3, 3))

    # dJ/dt = -Bi_μ·(μ - μ_b) → linearized
    A0[0, 0] = -p.Bi_mu * fJ
    A0[0, 1] = 0.0
    A0[0, 2] = -p.Bi_mu * ft

    # du/dt = [-Bi_c(u-1) - Da·J·R + u·Bi_μ·(μ-μ_b)] / J
    A0[1, 0] = (u0 * p.Bi_mu * fJ - p.Da * (R0 + J0 * RJ)) / J0
    A0[1, 1] = (-p.Bi_c - p.Da * J0 * Ru) / J0
    A0[1, 2] = (u0 * p.Bi_mu * ft - p.Da * J0 * Rt) / J0

    # dθ/dt = [-Bi_T·θ + Da·J·R] / C₀
    A0[2, 0] = p.Da * (R0 + J0 * RJ) / p.C0
    A0[2, 1] = p.Da * J0 * Ru / p.C0
    A0[2, 2] = (-p.Bi_T + p.Da * J0 * Rt) / p.C0

    return A0


def build_D2(J0, u0, theta0, p):
    """
    k² coefficient matrix (diffusion contributions).
    Negative-definite for physical stability.
    """
    ph0 = phi_of_J(J0, p)
    fJ = df_dJ(J0, theta0, p)
    ft = df_dtheta(J0, theta0, p)
    M0 = p.M0 * (1 - ph0)**p.m_mob
    D_eff = p.delta * p.D0 * (1 - ph0)**p.m_diff

    D2 = np.zeros((3, 3))
    D2[0, 0] = -M0 * fJ           # J diffusion via chemical potential
    D2[0, 2] = -M0 * ft           # J-θ coupling via χ(θ)
    D2[1, 1] = -D_eff / J0        # reactant diffusion
    D2[2, 2] = -p.alpha * p.K0 / p.C0   # thermal diffusion
    return D2


def build_D4(J0, u0, theta0, p):
    """k⁴ coefficient (Cahn-Hilliard regularization). Negative → stabilizes short λ."""
    ph0 = phi_of_J(J0, p)
    M0 = p.M0 * (1 - ph0)**p.m_mob
    D4 = np.zeros((3, 3))
    D4[0, 0] = -M0 * p.ell**2   # ∂²(-ℓ²∂²J/∂x²)/∂x² → -ℓ²k⁴
    return D4


def dispersion_matrix(k, J0, u0, theta0, p):
    """A(k) = A₀ + k²·D₂ + k⁴·D₄."""
    A0 = build_A0(J0, u0, theta0, p)
    D2 = build_D2(J0, u0, theta0, p)
    D4 = build_D4(J0, u0, theta0, p)
    return A0 + k**2 * D2 + k**4 * D4


def dispersion_eigenvalues(k_arr, J0, u0, theta0, p):
    """
    Compute eigenvalues σ(k) for each wavenumber.
    Returns array shape (len(k_arr), 3) of complex eigenvalues,
    sorted by descending Re(σ) at each k.
    """
    n_k = len(k_arr)
    sigmas = np.zeros((n_k, 3), dtype=complex)
    for i, k in enumerate(k_arr):
        A = dispersion_matrix(k, J0, u0, theta0, p)
        evals = eigvals(A)
        idx = np.argsort(-np.real(evals))
        sigmas[i] = evals[idx]
    return sigmas


def max_growth_rate(k_arr, J0, u0, theta0, p):
    """max_k Re(σ₁(k)) — the maximum growth rate over all wavenumbers."""
    sigmas = dispersion_eigenvalues(k_arr, J0, u0, theta0, p)
    return np.max(np.real(sigmas[:, 0]))


# ═══════════════════════════════════════════════════════════════════
# Parameter scans for stability boundaries
# ═══════════════════════════════════════════════════════════════════

def stability_scan_1d(param_name, param_values, p_base, k_arr):
    """
    Scan one parameter and compute stability properties at each value.

    Returns dict with:
      'values': parameter values
      'max_re_sigma': max Re(σ) over all k for leading eigenvalue
      'sigma_k0': 3 eigenvalues at k=0
      'k_star': wavenumber of max growth rate
      'omega_star': frequency at max growth rate
      'J0', 'u0', 'theta0': base state
    """
    nv = len(param_values)
    result = {
        'values': param_values,
        'max_re_sigma': np.full(nv, np.nan),
        'sigma_k0': np.full((nv, 3), np.nan, dtype=complex),
        'k_star': np.full(nv, np.nan),
        'omega_star': np.full(nv, np.nan),
        'J0': np.full(nv, np.nan),
        'u0': np.full(nv, np.nan),
        'theta0': np.full(nv, np.nan),
    }

    for i, val in enumerate(param_values):
        pp = replace(p_base, **{param_name: val})
        ss_list = find_uniform_ss(pp)
        if ss_list is None:
            continue

        # Use smallest-J steady state (collapsed branch, most likely to be Hopf unstable)
        J0, u0, theta0 = ss_list[0]
        result['J0'][i] = J0
        result['u0'][i] = u0
        result['theta0'][i] = theta0

        sigmas = dispersion_eigenvalues(k_arr, J0, u0, theta0, pp)
        re_max = np.real(sigmas[:, 0])
        idx_max = np.argmax(re_max)

        result['max_re_sigma'][i] = re_max[idx_max]
        result['sigma_k0'][i] = sigmas[0]  # k=0 eigenvalues
        result['k_star'][i] = k_arr[idx_max]
        result['omega_star'][i] = np.abs(np.imag(sigmas[idx_max, 0]))

    return result


def stability_scan_2d(xname, xvals, yname, yvals, p_base, k_arr):
    """
    2D parameter scan: max Re(σ) map.
    Returns dict with 'X', 'Y' meshgrids and 'max_re_sigma' 2D array.
    """
    nx, ny = len(xvals), len(yvals)
    max_re = np.full((ny, nx), np.nan)
    k_star = np.full((ny, nx), np.nan)
    omega_star = np.full((ny, nx), np.nan)

    for j, yv in enumerate(yvals):
        for i, xv in enumerate(xvals):
            pp = replace(p_base, **{xname: xv, yname: yv})
            ss_list = find_uniform_ss(pp)
            if ss_list is None:
                continue
            J0, u0, theta0 = ss_list[0]
            sigmas = dispersion_eigenvalues(k_arr, J0, u0, theta0, pp)
            re_lead = np.real(sigmas[:, 0])
            idx = np.argmax(re_lead)
            max_re[j, i] = re_lead[idx]
            k_star[j, i] = k_arr[idx]
            omega_star[j, i] = np.abs(np.imag(sigmas[idx, 0]))

    X, Y = np.meshgrid(xvals, yvals)
    return {'X': X, 'Y': Y, 'max_re_sigma': max_re,
            'k_star': k_star, 'omega_star': omega_star}


# ═══════════════════════════════════════════════════════════════════
# Numerical eigenvalue analysis (full 3N×3N system)
# ═══════════════════════════════════════════════════════════════════

def find_numerical_ss(p_scan, N=51, t_end=500.0):
    """
    Find the (possibly non-uniform) steady state by time-integrating the
    full 1D PDE to steady state, then using the final state.

    Uses scan_optimized.py machinery.

    Returns dict with 'J', 'u', 'theta', 'x' arrays (shape N), or None.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from scan_optimized import (
        Params, finalize_params, initial_state, rhs_mol_logJ,
        make_jac_sparsity, simulate, _LOG_J_MAX
    )

    # Build matching Params
    sp = Params(
        N=N, t_end=t_end, n_save=max(500, int(t_end * 5)),
        phi_p0=p_scan.phi_p0, chi_inf=p_scan.chi_inf,
        S_chi=p_scan.S_chi, chi1=p_scan.chi1,
        Omega_e=p_scan.Omega_e, ell=p_scan.ell,
        Da=p_scan.Da, delta=p_scan.delta, alpha=p_scan.alpha,
        Gamma_A=p_scan.Gamma_A, eps_T=p_scan.eps_T,
        Bi_mu=p_scan.Bi_mu, Bi_c=p_scan.Bi_c, Bi_T=p_scan.Bi_T,
        m_act=p_scan.m_act, m_diff=p_scan.m_diff, m_mob=p_scan.m_mob,
        M0=p_scan.M0, D0=p_scan.D0, C0=p_scan.C0, K0=p_scan.K0,
        J_init=p_scan.J_init, theta_init=p_scan.theta_init,
    )
    sp = finalize_params(sp)

    try:
        data = simulate(sp)
    except Exception as e:
        print(f"  Simulation failed: {e}")
        return None

    # Use final state
    return {
        'J': data['J'][:, -1],
        'u': data['u'][:, -1],
        'theta': data['theta'][:, -1],
        'x': data['x'],
        'params': sp,
        'data': data,
    }


def numerical_eigenvalues(ss, p_scan, n_eigs=20):
    """
    Compute leading eigenvalues of the 3N×3N Jacobian at the steady state.

    Uses finite-difference Jacobian (same as simulation).
    Returns complex eigenvalues sorted by descending Re(σ).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from scan_optimized import (
        Params, finalize_params, rhs_mol_logJ, _LOG_J_MAX
    )

    sp = ss['params']
    N = sp.N
    n3 = 3 * N

    # Build state vector from SS
    J = ss['J']
    u = ss['u']
    theta = ss['theta']
    logJ = np.log(J)
    W = J * u
    y_ss = np.concatenate([logJ, W, theta])

    # Compute Jacobian via FD
    rhs_fn = lambda t, y: rhs_mol_logJ(t, y, sp)
    f0 = rhs_fn(0, y_ss)

    Jac = np.zeros((n3, n3))
    for i in range(n3):
        hi = max(1e-8, 1e-6 * abs(y_ss[i]))
        yp = y_ss.copy()
        yp[i] += hi
        Jac[:, i] = (rhs_fn(0, yp) - f0) / hi

    # Full eigenvalue decomposition
    evals = eigvals(Jac)

    # Sort by descending Re
    idx = np.argsort(-np.real(evals))
    return evals[idx]


# ═══════════════════════════════════════════════════════════════════
# Diagnostics / printing
# ═══════════════════════════════════════════════════════════════════

def print_base_state(J0, u0, theta0, p):
    """Print base state and key quantities."""
    ph = phi_of_J(J0, p)
    R0 = reaction_rate(u0, theta0, J0, p)
    mu = chem_pot(J0, theta0, p)
    mu_b = m_bath(p)

    print(f"  J₀ = {J0:.4f},  u₀ = {u0:.4f},  θ₀ = {theta0:.4f}")
    print(f"  φ₀ = {ph:.4f},  R₀ = {R0:.6f}")
    print(f"  μ₀ = {mu:.6f},  μ_b = {mu_b:.6f},  |μ₀-μ_b| = {abs(mu-mu_b):.2e}")
    print(f"  Da·J₀·R₀ = {p.Da*J0*R0:.6f},  Bi_T·θ₀ = {p.Bi_T*theta0:.6f}")
    print(f"  Bi_c·(1-u₀) = {p.Bi_c*(1-u0):.6f}")


def print_stability(A0, J0, u0, theta0, p):
    """Print stability analysis of A₀."""
    evals = eigvals(A0)
    idx = np.argsort(-np.real(evals))
    evals = evals[idx]
    print(f"  A₀ eigenvalues:")
    for i, ev in enumerate(evals):
        print(f"    σ_{i+1} = {ev.real:+.6f} {ev.imag:+.6f}j")
    tr = np.trace(A0)
    det = np.linalg.det(A0)
    print(f"  tr(A₀) = {tr:.6f}, det(A₀) = {det:.6f}")
    if np.any(np.real(evals) > 0):
        print(f"  → UNSTABLE (positive Re eigenvalue)")
        if np.any(np.abs(np.imag(evals[np.real(evals) > 0])) > 1e-6):
            print(f"  → Hopf instability (oscillatory)")
        else:
            print(f"  → Monotone instability")
    else:
        print(f"  → Stable (all Re < 0)")


# ═══════════════════════════════════════════════════════════════════
# Main diagnostic
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = LSAParams()
    print("=" * 60)
    print("1D Gel Slab — Linear Stability Analysis (Dispersion)")
    print("=" * 60)
    print(f"\nParameters: Da={p.Da}, Bi_c={p.Bi_c}, Bi_T={p.Bi_T}, "
          f"S_χ={p.S_chi}, Γ_A={p.Gamma_A}")
    print(f"  χ∞={p.chi_inf}, χ₁={p.chi1}, Ω_e={p.Omega_e}")
    print(f"  δ={p.delta}, α={p.alpha}, ℓ={p.ell}")
    print(f"  m_act={p.m_act}, m_diff={p.m_diff}, m_mob={p.m_mob}")
    print(f"  M₀={p.M0}, D₀={p.D0}")

    # ─── Find all uniform steady states ───
    print("\n── Uniform Steady States ──")
    ss_list = find_uniform_ss(p)
    if ss_list is None:
        print("  No steady state found!")
    else:
        print(f"  Found {len(ss_list)} steady state(s):")
        for i, (J0, u0, theta0) in enumerate(ss_list):
            print(f"\n  SS #{i+1}:")
            print_base_state(J0, u0, theta0, p)
            A0 = build_A0(J0, u0, theta0, p)
            print_stability(A0, J0, u0, theta0, p)

    # ─── Matrix diagnostics ───
    if ss_list is not None:
        for i, (J0, u0, theta0) in enumerate(ss_list):
            print(f"\n── Matrix Diagnostics (SS #{i+1}) ──")
            D2 = build_D2(J0, u0, theta0, p)
            D4 = build_D4(J0, u0, theta0, p)
            fJ_val = df_dJ(J0, theta0, p)
            ft_val = df_dtheta(J0, theta0, p)
            ph0 = phi_of_J(J0, p)
            M0_val = p.M0 * (1 - ph0)**p.m_mob
            D_eff = p.delta * p.D0 * (1 - ph0)**p.m_diff
            print(f"  f_J = {fJ_val:.6f}  ({'spinodal' if fJ_val < 0 else 'stable'})")
            print(f"  f_θ = {ft_val:.6f}")
            print(f"  M₀_eff = {M0_val:.6f}")
            print(f"  D_eff = {D_eff:.6f}")
            print(f"  D₂ diag: [{D2[0,0]:.4f}, {D2[1,1]:.4f}, {D2[2,2]:.4f}]")
            print(f"  D₂[0,2] = {D2[0,2]:.6f}")
            print(f"  D₄[0,0] = {D4[0,0]:.8f}")
            if fJ_val < 0:
                k_spin = np.sqrt(abs(fJ_val) / (2 * p.ell**2))
                print(f"  Spinodal peak wavenumber k* = √(|f_J|/2ℓ²) = {k_spin:.1f}")
                sigma_spin = M0_val * fJ_val**2 / (4 * p.ell**2)
                print(f"  Spinodal max growth ≈ M₀f_J²/(4ℓ²) = {sigma_spin:.1f}")

    # ─── Dispersion relation for each SS ───
    k_arr = np.linspace(0, 200, 2000)
    if ss_list is not None:
        print("\n── Dispersion Relation ──")
        for i, (J0, u0, theta0) in enumerate(ss_list):
            sigmas = dispersion_eigenvalues(k_arr, J0, u0, theta0, p)
            re_max = np.max(np.real(sigmas[:, 0]))
            idx_max = np.argmax(np.real(sigmas[:, 0]))
            k_max = k_arr[idx_max]
            print(f"\n  SS #{i+1} (J₀={J0:.4f}):")
            print(f"    max Re(σ₁) = {re_max:.6f} at k* = {k_max:.3f}")
            print(f"    ω* = |Im(σ₁(k*))| = {abs(np.imag(sigmas[idx_max, 0])):.4f}")
            # Print k=0 eigenvalues
            print(f"    k=0 eigenvalues: ", end="")
            for s in sigmas[0]:
                print(f"  {s.real:+.4f}{s.imag:+.4f}j", end="")
            print()
            if re_max > 0:
                print(f"    → UNSTABLE (max growth at k*={k_max:.2f})")
            else:
                print(f"    → STABLE for all k")

    # ─── Da scan ───
    print("\n── Da Scan (k=0 Hopf analysis) ──")
    Da_vals = np.linspace(1.0, 20.0, 40)
    k_arr_scan = np.array([0.0])  # k=0 only for Hopf
    scan_da = stability_scan_1d('Da', Da_vals, p, k_arr_scan)
    for i, Da in enumerate(Da_vals):
        ev = scan_da['sigma_k0'][i]
        if not np.isnan(ev[0].real):
            re_lead = max(ev[0].real, ev[1].real, ev[2].real)
            tag = "UNSTABLE" if re_lead > 0 else "stable"
            im_lead = ev[0].imag
            print(f"  Da={Da:5.1f}: σ = {ev[0].real:+8.4f}±{abs(ev[0].imag):.4f}j, "
                  f"{ev[2].real:+8.4f}, J₀={scan_da['J0'][i]:.3f} [{tag}]")

    print("\nDone.")
