#!/usr/bin/env python3
"""
hard_excitation_demo.py — IC-path hysteresis test for the "LCST
bistability rewrites Hopf" main story.

At a C-cell (LSA stable around the homogeneous SS, PDE oscillates with
J_amp ~ 2.2), run two simulations at IDENTICAL parameters but starting
from very different ICs:

  IC-A: cold-swollen start  (J=1.30, θ=0, u=0.02)
        — the default fig4-sweep IC; we know it reaches the lcst_front
          limit cycle (J_amp=2.244 in the existing fig4 cache).

  IC-B: pre-collapsed start (J=0.30, θ=0.72, u=0.94)
        — values of the homogeneous collapsed SS (the only SS that
          find_uniform_ss returns at this point). 0D LSA says this SS
          is locally stable (re_max_complex = -inf, no Hopf signature).

Outcomes:
  - Both reach the same attractor → single global attractor, no
    hysteresis. The PDE-level state space is convex.
  - IC-A reaches cycle, IC-B settles to a different state → the PDE
    has TWO COEXISTING attractors at this parameter point, which is
    the hard-excitation / fold-of-cycles signature LSA cannot detect.

Computational economy: N=121 (vs 301 in fig4) and t_end=100 (vs 200);
each sim ~2-3 min single-core. Wall-time event caps any single sim
at 600s as a safety net against SNIC slow flow.
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from scan_optimized import (
    Params, finalize_params, rhs_mol_logJ, make_jac_sparsity,
    make_sparse_fd_jac, cell_centers, _LOG_J_MAX,
)
from fig2_data import WORKING_POINT


DEMO_BI_T  = 0.0590
DEMO_S_CHI = 1.80
DEMO_N     = 121
DEMO_T_END = 200.0
DEMO_NSAVE = 4000
MAX_WALL_SEC = 900.0  # 15 min cap per sim


def simulate_with_ic(p: Params, y0: np.ndarray, label: str):
    p = finalize_params(p)
    n3 = 3 * p.N
    rhs_fn = lambda t, y: rhs_mol_logJ(t, y, p)
    S = make_jac_sparsity(p.N)
    jac_sparse, _ = make_sparse_fd_jac(rhs_fn, S, n3)

    t0_wall = time.perf_counter()

    def wall_event(t, y):
        return MAX_WALL_SEC - (time.perf_counter() - t0_wall)
    wall_event.terminal = True
    wall_event.direction = -1

    sol = solve_ivp(
        fun=rhs_fn, jac=jac_sparse,
        t_span=(0.0, p.t_end), y0=y0,
        t_eval=np.linspace(0, p.t_end, p.n_save),
        method=p.method, rtol=p.rtol, atol=p.atol,
        max_step=p.max_step, events=wall_event,
    )
    dt = time.perf_counter() - t0_wall
    timed_out = (sol.status == 1)
    print(f"  [{label}] {dt:.0f}s, success={sol.success}, "
          f"nfev={sol.nfev}, timed_out={timed_out}, "
          f"reached_t={sol.t[-1]:.1f}/{p.t_end:.1f}")

    n = p.N
    log_J_min = np.log(p.phi_p0 * 1.02)
    J = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
    W = sol.y[n:2*n]
    theta = sol.y[2*n:]
    u = np.maximum(W / J, p.u_floor)
    return dict(t=sol.t, x=cell_centers(p.N), J=J, u=u, theta=theta,
                timed_out=timed_out, label=label)


def build_ic_cold(p: Params):
    """Default fig4-sweep IC: J=J_init, θ=0, u=u_init (cold-swollen guess)."""
    x = cell_centers(p.N)
    n = p.N
    Jvec = np.maximum(p.J_init + p.eps_J * np.cos(np.pi * x),
                      np.exp(np.log(p.phi_p0 * 1.02)) + 1e-6)
    uvec = np.maximum(p.u_init + p.eps_u * np.cos(np.pi * x), p.u_floor)
    thvec = p.theta_init + p.eps_theta * x
    Wvec = Jvec * uvec
    return np.concatenate([np.log(Jvec), Wvec, thvec])


def build_ic_collapsed(p: Params, J_pre=0.30, theta_pre=0.72, u_pre=0.94):
    """Pre-collapsed IC at the homogeneous collapsed SS values."""
    x = cell_centers(p.N)
    n = p.N
    Jvec = np.full(n, J_pre) + 1e-3 * np.cos(np.pi * x)
    uvec = np.full(n, u_pre) + 1e-3 * np.cos(np.pi * x)
    thvec = np.full(n, theta_pre) + 1e-3 * np.cos(np.pi * x)
    Wvec = Jvec * uvec
    return np.concatenate([np.log(Jvec), Wvec, thvec])


def main():
    p_dict = dict(WORKING_POINT)
    p_dict.update(Bi_T=DEMO_BI_T, S_chi=DEMO_S_CHI,
                  N=DEMO_N, t_end=DEMO_T_END, n_save=DEMO_NSAVE)
    p = Params(**p_dict)

    print(f"=== Hard-excitation IC-path test ===")
    print(f"  Bi_T={DEMO_BI_T}, S_chi={DEMO_S_CHI}")
    print(f"  N={DEMO_N}, t_end={DEMO_T_END}")
    print(f"  Wall-cap per sim: {MAX_WALL_SEC:.0f}s")
    print()

    print("  >>> RUN A: cold-swollen IC (J=1.30, θ=0, u=0.02)")
    y0_A = build_ic_cold(p)
    out_A = simulate_with_ic(p, y0_A, "cold-start")

    print("\n  >>> RUN B: pre-collapsed IC (J=0.30, θ=0.72, u=0.94)")
    y0_B = build_ic_collapsed(p)
    out_B = simulate_with_ic(p, y0_B, "collapsed-start")

    # Diagnostics on last 30%: detect cycle via peak-finding (more robust
    # than raw amplitude because slow drift onto a SS reads as "non-zero
    # amp" but is not a cycle), and record terminal mean-J / mean-θ to
    # show whether the two ICs reach the same attractor.
    from scipy.signal import find_peaks

    def _osc_metrics(d, frac0=0.7):
        n_t = len(d["t"])
        sl = slice(int(frac0 * n_t), n_t)
        tt = d["t"][sl]
        Jw = d["J"][:, sl]
        Jm = Jw.mean(axis=0)
        amp = float(Jm.max() - Jm.min())
        prom = max(0.05 * amp, 0.02)
        peaks, _ = find_peaks(Jm, prominence=prom, distance=3)
        is_cycle = len(peaks) >= 3
        # terminal state (mean over last 5%)
        n5 = max(int(0.05 * n_t), 10)
        return dict(
            amp=amp,
            n_peaks=len(peaks),
            is_cycle=is_cycle,
            J_mean_term=float(d["J"][:, -n5:].mean()),
            theta_mean_term=float(d["theta"][:, -n5:].mean()),
            J_surf_term=float(d["J"][-1, -n5:].mean()),
            theta_surf_term=float(d["theta"][-1, -n5:].mean()),
        )

    mA = _osc_metrics(out_A)
    mB = _osc_metrics(out_B)
    print()
    print(f"  --- last-30% diagnostics (proper peak-detection) ---")
    print(f"  cold-start:      <J>_amp={mA['amp']:.3f}  n_peaks={mA['n_peaks']}  "
          f"is_cycle={mA['is_cycle']}")
    print(f"                   terminal: <J>={mA['J_mean_term']:.3f}  "
          f"<θ>={mA['theta_mean_term']:.3f}  "
          f"J_surf={mA['J_surf_term']:.3f}  θ_surf={mA['theta_surf_term']:.3f}")
    print(f"  collapsed-start: <J>_amp={mB['amp']:.3f}  n_peaks={mB['n_peaks']}  "
          f"is_cycle={mB['is_cycle']}")
    print(f"                   terminal: <J>={mB['J_mean_term']:.3f}  "
          f"<θ>={mB['theta_mean_term']:.3f}  "
          f"J_surf={mB['J_surf_term']:.3f}  θ_surf={mB['theta_surf_term']:.3f}")

    same_state = (abs(mA["J_mean_term"] - mB["J_mean_term"]) < 0.2
                  and abs(mA["theta_mean_term"] - mB["theta_mean_term"]) < 0.2)
    if mA["is_cycle"] and mB["is_cycle"] and same_state:
        print(f"  >>> Single global cycle attractor — no hysteresis.")
    elif (not same_state):
        print(f"  >>> HYSTERESIS CONFIRMED: two ICs settle to DIFFERENT "
              f"attractors. <J> differs by {abs(mA['J_mean_term']-mB['J_mean_term']):.3f}, "
              f"<θ> by {abs(mA['theta_mean_term']-mB['theta_mean_term']):.3f}. "
              f"Coexisting attractors at the same parameter point — the "
              f"smoking gun for nonlinear bistability that LSA cannot detect.")
    else:
        print(f"  >>> Inconclusive — extend t_end to verify terminal states.")

    # Plot
    out_dir = _HERE.parent.parent / "Figure" / "pub"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    for d, lbl, c in [(out_A, "cold-start IC", "#3676b8"),
                      (out_B, "collapsed-start IC", "#c0392b")]:
        axes[0, 0].plot(d["t"], d["J"][-1], color=c, lw=1.0, label=lbl)
        axes[1, 0].plot(d["t"], d["theta"][-1], color=c, lw=1.0)
    axes[0, 0].set_ylabel(r"$J_{\rm surf}(t)$")
    axes[1, 0].set_ylabel(r"$\theta_{\rm surf}(t)$")
    axes[1, 0].set_xlabel(r"$t$")
    axes[0, 0].set_title(
        f"Bi_T={DEMO_BI_T}, S_χ={DEMO_S_CHI}: surface trajectory")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)

    # Phase plane <J> vs <θ>
    for d, lbl, c in [(out_A, "cold-start", "#3676b8"),
                      (out_B, "collapsed-start", "#c0392b")]:
        Jm = d["J"].mean(axis=0); thm = d["theta"].mean(axis=0)
        # color trajectory by time (lighter near t=0, dark at t=t_end)
        n = len(d["t"])
        axes[0, 1].plot(Jm, thm, color=c, lw=0.7, alpha=0.85, label=lbl)
        axes[0, 1].plot(Jm[-1], thm[-1], "o", color=c, ms=8,
                        markeredgecolor="k")
    axes[0, 1].set_xlabel(r"$\langle J\rangle$")
    axes[0, 1].set_ylabel(r"$\langle\theta\rangle$")
    axes[0, 1].set_title("Phase plane (• = terminal state)")
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # phi_surf shows LCST hopping
    p_p0 = p_dict["phi_p0"]
    for d, lbl, c in [(out_A, "cold-start", "#3676b8"),
                      (out_B, "collapsed-start", "#c0392b")]:
        phi_surf = p_p0 / d["J"][-1]
        axes[1, 1].plot(d["t"], phi_surf, color=c, lw=1.0, label=lbl)
    axes[1, 1].axhline(0.5, color="grey", lw=0.7, ls=":", label="LCST (φ=0.5)")
    axes[1, 1].set_xlabel(r"$t$")
    axes[1, 1].set_ylabel(r"$\varphi_{\rm surf}(t)$")
    axes[1, 1].set_title("Surface φ: LCST crossings")
    axes[1, 1].legend(loc="best", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "fig6.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    cache = Path(_HERE.parent.parent / "data" / "fig4" / "hard_excitation_demo.npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        Bi_T=DEMO_BI_T, S_chi=DEMO_S_CHI, N=DEMO_N, t_end=DEMO_T_END,
        t_A=out_A["t"], J_A=out_A["J"], theta_A=out_A["theta"], x_A=out_A["x"],
        t_B=out_B["t"], J_B=out_B["J"], theta_B=out_B["theta"], x_B=out_B["x"],
        timed_out_A=out_A["timed_out"], timed_out_B=out_B["timed_out"],
        J_mean_term_A=mA["J_mean_term"], theta_mean_term_A=mA["theta_mean_term"],
        J_mean_term_B=mB["J_mean_term"], theta_mean_term_B=mB["theta_mean_term"],
    )
    print(f"  Saved: {cache}")


if __name__ == "__main__":
    main()
