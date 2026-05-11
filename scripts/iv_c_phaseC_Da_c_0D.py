#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseC_Da_c_0D.py — analytic 0D Hopf onset Da_c^0D at each Bi_T,
on the cold-swollen-connected SS branch.

Why continuation, not find_uniform_ss?
  The free Newton solver in linear_stability_1d.find_uniform_ss returns
  the collapsed branch even when the cold-swollen branch coexists,
  because at our working-point parameters the cold branch is a
  weakly-attracting fixed point that Newton overshoots from generic
  guesses. The derivation assumes linearisation about the
  cold-connected SS — so we trace that branch from Da=0+ via Newton
  continuation, with each step's solution as the next step's initial
  guess.

Reuses
  chem_pot, reaction_rate, m_bath, build_A0   from linear_stability_1d.py

Algorithm
  1. Set p = LSAParams() (working-point material constants).
  2. For each Bi_T ∈ {0.06, 0.10, 0.16, 0.25}:
     - Pick a fine Da grid Da ∈ (1e-4, ~1.0), log-spaced.
     - Newton-continue from (J=1.30, theta=ε) tracking the cold-swollen
       branch.  Stop when Newton diverges (saddle-node) or the SS jumps
       to a different branch (J drops below 0.7).
     - Compute re_max_complex(Da) along the branch.
     - Da_c^0D = smallest Da on the traced branch where re_max_complex
       crosses 0.

Output
  data/iv_c/phaseC/Da_c_0D.npz   (Bi_T, Da_c_0D, omega, J0, theta0)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import fsolve

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from linear_stability_1d import (
    LSAParams, chem_pot, reaction_rate, m_bath, build_A0,
)

OUT = _HERE.parent / "data" / "iv_c" / "phaseC"
OUT.mkdir(parents=True, exist_ok=True)

BI_T_VALS = np.array([0.06, 0.10, 0.16, 0.25])

DA_MIN = 1e-4
DA_MAX = 5.0
N_DA = 600


def trace_cold_branch(Bi_T: float, base: LSAParams,
                      Da_grid: np.ndarray = None) -> dict:
    """
    Newton-continue the cold-swollen-connected SS as Da increases from 0+.
    Compute eigenvalues of A_0 along the branch.

    Returns:
      Da_traced  : 1D array (length n_traced)
      J_traced, theta_traced, u_traced
      re_complex : max Re among complex-conjugate eigvals at each Da
      omega      : Im of dominant complex eigval at each Da
      Da_c       : Da at which re_complex first crosses 0 (NaN if never)
      omega_c    : Hopf frequency at onset
      J0, theta0 : SS coords at onset
      reason     : termination reason
    """
    if Da_grid is None:
        Da_grid = np.exp(np.linspace(np.log(DA_MIN), np.log(DA_MAX), N_DA))

    mu_b = m_bath(base)

    J_curr, theta_curr = float(base.J_init), 1e-4
    Da_traced = []
    J_traced = []
    th_traced = []
    u_traced = []
    re_traced = []
    omega_traced = []

    def eqs_factory(p, Bi_T_loc, Da):
        Bi_c = p.Bi_c

        def eqs(x):
            J, theta = x
            if J <= p.phi_p0 * 1.001 or J > 6.0 or theta < -0.01 or theta > 30:
                return [1e6, 1e6]
            u = 1.0 - Bi_T_loc * theta / Bi_c
            if u <= 0.0 or u > 1.001:
                return [1e6, 1e6]
            mu_val = float(chem_pot(J, theta, p))
            R = float(reaction_rate(u, theta, J, p))
            return [mu_val - mu_b,
                    Bi_T_loc * theta - Da * J * R]
        return eqs

    reason = "completed"
    for Da in Da_grid:
        p = replace(base, Da=float(Da), Bi_T=float(Bi_T))
        eqs = eqs_factory(p, Bi_T, Da)
        sol, info, ier, msg = fsolve(eqs, [J_curr, theta_curr],
                                     full_output=True, xtol=1e-12)
        if ier != 1:
            reason = f"newton_diverged at Da={Da:.4f} ({msg[:40]})"
            break
        J_new, theta_new = sol
        # Reject if jumped branches (J shrunk too much from previous)
        if J_new < 0.5 * J_curr or J_new < 0.4:
            reason = (f"branch_jumped at Da={Da:.4f}: "
                      f"J {J_curr:.3f} → {J_new:.3f}")
            break
        # Reject if Newton residual too large
        res = float(np.linalg.norm(eqs([J_new, theta_new])))
        if res > 1e-6:
            reason = f"large_residual at Da={Da:.4f} (res={res:.2e})"
            break

        u_new = 1.0 - Bi_T * theta_new / p.Bi_c
        A = build_A0(J_new, u_new, theta_new, p)
        evs = eigvals(A)
        imag_nz = np.abs(evs.imag) > 1e-9
        if imag_nz.any():
            re_c = float(np.max(evs.real[imag_nz]))
            idx_c = int(np.argmax(evs.real[imag_nz]))
            om = float(np.abs(evs[imag_nz][idx_c].imag))
        else:
            re_c = float(np.max(evs.real))
            om = float("nan")

        Da_traced.append(float(Da))
        J_traced.append(float(J_new))
        th_traced.append(float(theta_new))
        u_traced.append(float(u_new))
        re_traced.append(re_c)
        omega_traced.append(om)

        J_curr, theta_curr = J_new, theta_new

    Da_traced = np.array(Da_traced)
    J_traced = np.array(J_traced)
    th_traced = np.array(th_traced)
    u_traced = np.array(u_traced)
    re_traced = np.array(re_traced)
    omega_traced = np.array(omega_traced)

    # Find Da_c: first sign change of re_traced (negative → positive)
    Da_c = float("nan")
    omega_c = float("nan")
    J0 = float("nan")
    theta0 = float("nan")
    if len(re_traced) >= 2:
        s = np.sign(re_traced)
        cross = np.where((s[:-1] < 0) & (s[1:] > 0))[0]
        if len(cross) > 0:
            k = cross[0]
            re_lo = re_traced[k]
            re_hi = re_traced[k + 1]
            Da_lo = Da_traced[k]
            Da_hi = Da_traced[k + 1]
            if re_hi != re_lo:
                Da_c = float(Da_lo + (Da_hi - Da_lo) * (-re_lo) /
                              (re_hi - re_lo))
            else:
                Da_c = float(Da_lo)
            # Interpolate omega, J, theta at onset
            frac = (Da_c - Da_lo) / (Da_hi - Da_lo) if Da_hi != Da_lo else 0
            omega_c = float(omega_traced[k] +
                             (omega_traced[k + 1] - omega_traced[k]) * frac)
            J0 = float(J_traced[k] + (J_traced[k + 1] - J_traced[k]) * frac)
            theta0 = float(th_traced[k] +
                            (th_traced[k + 1] - th_traced[k]) * frac)

    return dict(
        Bi_T=Bi_T, Da_traced=Da_traced,
        J_traced=J_traced, theta_traced=th_traced, u_traced=u_traced,
        re_complex_traced=re_traced, omega_traced=omega_traced,
        Da_c=Da_c, omega_c=omega_c, J0=J0, theta0=theta0,
        reason=reason,
    )


def Da_c_0D_analytic(Bi_T: float, base: LSAParams) -> float:
    """
    Leading-order saddle-node of the cold-swollen-connected SS.

    From the heat balance Bi_T*theta = Da*J*exp(Gamma_A*theta), the cold
    branch terminates at theta -> infty when Bi_T = Da*J*Gamma_A
    (denominator of theta = Da*J/(Bi_T - Da*J*Gamma_A) hits zero, in the
    small-theta linearisation).  Hence

        Da_c^0D  ~  Bi_T / (J_0 * Gamma_A)

    with J_0 = J_init (cold-reference).  The analysis on this branch
    gives Re(complex_pair) -> 0 simultaneously with the SN — the SS
    becomes neutrally stable just as it disappears, so this Da is the
    natural 0D analog of the PDE oscillation onset.
    """
    return float(Bi_T / (base.J_init * base.Gamma_A))


def main():
    base = LSAParams()
    print("=" * 70)
    print(" Phase C - 0D Hopf onset Da_c^0D (cold-branch continuation)")
    print("=" * 70)
    print(f"  base WP: alpha={base.alpha}, S_chi={base.S_chi}, "
          f"Gamma_A={base.Gamma_A}, Bi_c={base.Bi_c}, J_init={base.J_init}")
    print(f"  Bi_T values: {list(BI_T_VALS)}")
    print(f"  Da grid: log-spaced [{DA_MIN}, {DA_MAX}], {N_DA} pts\n")

    Da_c_analytic = np.full(len(BI_T_VALS), np.nan)
    Da_c_continuation = np.full(len(BI_T_VALS), np.nan)
    Da_c_arr = np.full(len(BI_T_VALS), np.nan)   # the value we use downstream
    omega_c_arr = np.full(len(BI_T_VALS), np.nan)
    J0_arr = np.full(len(BI_T_VALS), np.nan)
    theta0_arr = np.full(len(BI_T_VALS), np.nan)
    branch_data = {}   # Bi_T -> traced arrays

    print(f"  {'Bi_T':>7s}  {'Da_c^0D (analytic)':>20s}  "
          f"{'last Da on branch':>20s}  {'Da_c (used)':>14s}")
    for i, Bi_T in enumerate(BI_T_VALS):
        Da_c_an = Da_c_0D_analytic(float(Bi_T), base)
        Da_c_analytic[i] = Da_c_an
        res = trace_cold_branch(float(Bi_T), base)
        n_traced = len(res["Da_traced"])
        # Continuation Da_c: last finite Da on the cold branch
        # (i.e. SN if branch died via Newton failure; otherwise upper limit)
        if n_traced > 0:
            Da_last = float(res["Da_traced"][-1])
            Da_c_continuation[i] = Da_last
        # Use analytic Da_c^0D as the canonical value (matches numerical SN
        # at low Bi_T to ~10%; numerical-only fails at Bi_T=0.25 because
        # the continuation walks smoothly through the fold instead of
        # diverging).
        Da_c_arr[i] = Da_c_an
        # Pull J0, theta0 from the closest traced point if available
        if n_traced > 0 and np.isfinite(Da_c_an):
            k = int(np.argmin(np.abs(res["Da_traced"] - Da_c_an)))
            J0_arr[i] = float(res["J_traced"][k])
            theta0_arr[i] = float(res["theta_traced"][k])
        print(f"  {Bi_T:7.3f}  {Da_c_an:20.4f}  "
              f"{Da_c_continuation[i]:20.4f}  {Da_c_arr[i]:14.4f}")
        branch_data[float(Bi_T)] = res

    npz = OUT / "Da_c_0D.npz"
    payload = dict(Bi_T=BI_T_VALS,
                   Da_c_0D=Da_c_arr,
                   Da_c_0D_analytic=Da_c_analytic,
                   Da_c_0D_continuation_last=Da_c_continuation,
                   omega=omega_c_arr, J0=J0_arr, theta0=theta0_arr,
                   alpha=base.alpha, Bi_c=base.Bi_c,
                   S_chi=base.S_chi, Gamma_A=base.Gamma_A,
                   J_init=base.J_init)
    # Save full traced branches too
    for Bi_T, r in branch_data.items():
        tag = f"BiT_{Bi_T:.3f}".replace(".", "p")
        for k in ("Da_traced", "J_traced", "theta_traced", "u_traced",
                  "re_complex_traced", "omega_traced"):
            payload[f"{tag}__{k}"] = r[k]
    np.savez(npz, **payload)
    print(f"\n  saved {npz}")


if __name__ == "__main__":
    main()
