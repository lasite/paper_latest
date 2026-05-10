#!/usr/bin/env python3
"""
spinodal_demo.py — Demonstrate spinodal decomposition in the gel model.

Physics:
  When the gel is quenched into the spinodal region (f_J < 0),
  spatially uniform states are unstable to finite-wavelength
  perturbations. The Cahn-Hilliard regularization (ℓ² k⁴ term)
  selects a characteristic wavelength k* ≈ sqrt(-f_J / (2ℓ²)).

Setup:
  - Start from collapsed uniform state J₀ ≈ 0.30, θ₀ ≈ 1.5 (above LCST)
  - Add small random perturbation to J
  - Da = 0 (no reaction) to isolate thermodynamic phase separation
  - Fine mesh N = 301, ℓ = 0.01 (matches the appendix's predicted
    fastest-growing wavenumber k* ≈ sqrt(-f_J/(2 ℓ²)) ≈ 94)
  - Observe spontaneous formation of swollen/collapsed domains

Output:
  - Kymograph of J(x,t) showing domain formation
  - Spatial profiles at selected times
  - Growth rate comparison with linear theory
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csc_matrix
from dataclasses import dataclass, replace

# Import model components
from scan_optimized import (
    Params, phi_from_J, harmonic_mean, laplacian_neumann,
    local_chem_pot, finalize_params, cell_centers,
)


# ══════════════════════════════════════════════════════════════════
# 1. Verify spinodal condition: compute f_J at the working point
# ══════════════════════════════════════════════════════════════════

def compute_f_J(J, theta, p):
    """Compute ∂f/∂J (chemical potential derivative) numerically."""
    dJ = 1e-6
    f_plus = local_chem_pot(np.array([J + dJ]), np.array([theta]), p)[0]
    f_minus = local_chem_pot(np.array([J - dJ]), np.array([theta]), p)[0]
    return (f_plus - f_minus) / (2 * dJ)


def compute_f_JJ(J, theta, p):
    """Compute ∂²f/∂J² to verify spinodal (negative = inside spinodal)."""
    dJ = 1e-5
    f_plus = local_chem_pot(np.array([J + dJ]), np.array([theta]), p)[0]
    f_0 = local_chem_pot(np.array([J]), np.array([theta]), p)[0]
    f_minus = local_chem_pot(np.array([J - dJ]), np.array([theta]), p)[0]
    return (f_plus - 2 * f_0 + f_minus) / dJ**2


def find_spinodal_info(p, J0, theta0):
    """Compute spinodal characteristics at a given state."""
    f_J = compute_f_J(J0, theta0, p)
    f_JJ = compute_f_JJ(J0, theta0, p)

    phi = p.phi_p0 / J0
    M0 = p.M0 * max(1 - phi, 1e-12)**p.m_mob

    print(f"  State: J={J0:.4f}, θ={theta0:.4f}, φ={phi:.4f}")
    print(f"  f_J  = {f_J:.4f}")
    print(f"  f_JJ = {f_JJ:.4f}  ({'INSIDE spinodal' if f_J < 0 else 'outside spinodal'})")

    if f_J < 0:
        # Predicted fastest-growing wavenumber
        k_star = np.sqrt(-f_J / (2 * p.ell**2))
        lambda_star = 2 * np.pi / k_star
        # Growth rate at k*
        sigma_star = M0 * f_J**2 / (4 * p.ell**2)
        print(f"  k*   = {k_star:.1f}")
        print(f"  λ*   = {lambda_star:.4f} H₀")
        print(f"  σ*   = {sigma_star:.1f} (growth rate)")
        print(f"  Need N ≥ {int(1.0 / lambda_star * 10)} for 10 wavelengths")
        return {"f_J": f_J, "M0": M0, "k_star": k_star,
                "lambda_star": lambda_star, "sigma_star": sigma_star}
    return {"f_J": f_J, "M0": M0}


# ══════════════════════════════════════════════════════════════════
# 2. Simplified RHS: J equation only (Cahn-Hilliard), θ fixed
# ══════════════════════════════════════════════════════════════════

def rhs_spinodal(t, y, p, theta_fixed):
    """
    Pure Cahn-Hilliard dynamics for J with fixed θ:
      J_t = ∂_x [ M(J) ∂_x μ ]
      μ = f(J,θ) - ℓ² J_xx
    No reaction, no reactant transport, no thermal evolution.
    """
    N = len(y)
    dx = 1.0 / N

    J = np.exp(np.clip(y, np.log(p.phi_p0 * 1.02), np.log(8.0)))
    theta = np.full(N, theta_fixed)
    phi = phi_from_J(J, p)

    # Chemical potential
    m_local = local_chem_pot(J, theta, p)
    m = m_local - p.ell**2 * laplacian_neumann(J, dx)

    # Swelling flux with Neumann BC (no-flux at both ends for spinodal)
    M_cell = p.M0 * np.maximum(1 - phi, 1e-12)**p.m_mob
    M_face = harmonic_mean(M_cell[:-1], M_cell[1:])

    q = np.zeros(N + 1)
    q[1:N] = -M_face * (m[1:] - m[:-1]) / dx
    # Both boundaries: no flux (closed system for spinodal)
    q[0] = 0.0
    q[N] = 0.0

    # logJ evolution
    logJ_t = -(q[1:] - q[:-1]) / (dx * J)
    return logJ_t


# ══════════════════════════════════════════════════════════════════
# 3. Full coupled RHS with Da=0 (alternative: see thermal relaxation)
# ══════════════════════════════════════════════════════════════════

def rhs_full_no_reaction(t, y, p):
    """
    Full 3-field model with Da=0: isolate spinodal from reaction.
    State: [logJ_0..N-1, W_0..N-1, theta_0..N-1]
    """
    from scan_optimized import rhs_mol_logJ
    p_no_rxn = replace(p, Da=0.0)
    return rhs_mol_logJ(t, y, p_no_rxn)


# ══════════════════════════════════════════════════════════════════
# 4. Run spinodal simulation
# ══════════════════════════════════════════════════════════════════

def run_spinodal(N=201, ell=0.005, J0=0.30, theta0=1.5,
                 perturbation=0.02, t_end=0.5, seed=42):
    """
    Run a Cahn-Hilliard spinodal decomposition simulation.

    Parameters:
      N: grid points (need enough to resolve k*)
      ell: interface parameter (smaller → sharper domains)
      J0: initial uniform swelling ratio (must be in spinodal: f_J < 0)
      theta0: fixed temperature (above LCST)
      perturbation: amplitude of random initial perturbation
      t_end: simulation time
      seed: random seed
    """
    p = Params(
        N=N, ell=ell,
        Da=0.0,  # no reaction
        phi_p0=0.15, chi_inf=0.60, S_chi=1.0, chi1=1.10,
        Omega_e=0.12, m_mob=1.0, M0=1.0,
        # These don't matter for pure CH but set them anyway
        Bi_T=0.10, Bi_c=0.70, Gamma_A=1.5,
        t_end=t_end, rtol=1e-8, atol=1e-10,
    )

    print(f"\n{'='*60}")
    print(f"Spinodal decomposition simulation")
    print(f"  N={N}, ℓ={ell}, J₀={J0}, θ={theta0}")
    print(f"{'='*60}")

    # Check spinodal condition
    info = find_spinodal_info(p, J0, theta0)
    if info["f_J"] >= 0:
        print("WARNING: Not in spinodal region! f_J >= 0")
        print("Try lower J0 or higher theta0.")
        return None

    # Initial condition: uniform + random noise
    rng = np.random.default_rng(seed)
    x = cell_centers(N)
    logJ0 = np.log(J0) + perturbation * rng.standard_normal(N)

    # Time points: very dense in the early linear-growth regime so we
    # can resolve several decades of exp(2σ* τ) before saturation.
    # With σ* ~ 3850 the linear regime ends at τ ~ 1e-3.
    t_eval = np.unique(np.concatenate([
        np.linspace(0, t_end * 0.01, 200),   # 0 .. 0.005 (dt=2.5e-5)
        np.linspace(t_end * 0.01, t_end, 300),  # 0.005 .. 0.5 (linear)
    ]))

    print(f"\n  Integrating (pure Cahn-Hilliard, θ fixed at {theta0})...")
    sol = solve_ivp(
        fun=lambda t, y: rhs_spinodal(t, y, p, theta0),
        t_span=(0, t_end),
        y0=logJ0,
        t_eval=t_eval,
        method="BDF",
        rtol=p.rtol, atol=p.atol,
        max_step=t_end / 200,
    )

    if not sol.success:
        print(f"  Integration failed: {sol.message}")
        return None

    print(f"  Done. nfev={sol.nfev}, {len(sol.t)} time points.")

    J = np.exp(np.clip(sol.y, np.log(p.phi_p0 * 1.02), np.log(8.0)))
    return {"x": x, "t": sol.t, "J": J, "p": p, "info": info,
            "theta0": theta0, "J0": J0}


# ══════════════════════════════════════════════════════════════════
# 5. Scan J0 and theta0 to find good spinodal parameters
# ══════════════════════════════════════════════════════════════════

def scan_spinodal_region():
    """Find which (J, θ) combinations are inside the spinodal."""
    p = Params(phi_p0=0.15, chi_inf=0.60, S_chi=1.0, chi1=1.10,
               Omega_e=0.12)

    print("\nSpinodal region scan: f_J(J, θ)")
    print(f"{'J':>8s}", end="")
    for theta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"  θ={theta:.1f}", end="")
    print()

    for J in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.30]:
        print(f"{J:8.2f}", end="")
        for theta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            fJ = compute_f_J(J, theta, p)
            marker = f"{fJ:7.2f}" if fJ >= 0 else f"\033[91m{fJ:7.2f}\033[0m"
            print(f"  {marker}", end="")
        print()


# ══════════════════════════════════════════════════════════════════
# 6. Plotting
# ══════════════════════════════════════════════════════════════════

def plot_spinodal(result, outdir="figures_pub"):
    """Generate spinodal decomposition figure."""
    os.makedirs(outdir, exist_ok=True)

    x = result["x"]
    t = result["t"]
    J = result["J"]
    info = result["info"]
    J0 = result["J0"]
    theta0 = result["theta0"]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
    fig.subplots_adjust(hspace=0.55, wspace=0.42,
                        left=0.09, right=0.96, top=0.95, bottom=0.10)

    label_kw = dict(fontsize=11, fontweight="bold")
    panel_label_pad = (-0.18, 1.05)

    def add_label(ax, s):
        ax.text(panel_label_pad[0], panel_label_pad[1], s,
                transform=ax.transAxes, **label_kw)

    # (a) Kymograph J(x,t)
    ax = axes[0, 0]
    vmin = np.percentile(J, 2)
    vmax = np.percentile(J, 98)
    im = ax.imshow(J, origin="lower", aspect="auto",
                   extent=[t[0], t[-1], x[0], x[-1]],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045, label="$J$")
    ax.set_xlabel(r"Time $\tau$", fontsize=8)
    ax.set_ylabel(r"$x/H_0$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Kymograph", fontsize=9)
    add_label(ax, "(a)")

    # (b) Spatial profiles at selected times
    ax = axes[0, 1]
    n_profiles = 6
    t_indices = np.linspace(0, len(t) - 1, n_profiles).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    for i, idx in enumerate(t_indices):
        ax.plot(x, J[:, idx], color=colors[i], lw=0.8,
                label=rf"$\tau$={t[idx]:.3f}")
    ax.axhline(J0, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$x/H_0$", fontsize=8)
    ax.set_ylabel(r"$J(x)$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.85,
              handlelength=1.3, borderpad=0.3)
    ax.set_title("Spatial profiles", fontsize=9)
    add_label(ax, "(b)")

    # (c) Band-averaged Fourier amplitude at k ≈ k* (matches paper text)
    ax = axes[1, 0]
    Nx = len(x)
    k_arr = np.fft.rfftfreq(Nx, d=1.0/Nx) * 2 * np.pi
    if "k_star" in info:
        kstar = info["k_star"]
        band_lo, band_hi = 0.7 * kstar, 1.3 * kstar
        band_mask = (k_arr >= band_lo) & (k_arr <= band_hi)
    else:
        kstar = None
        band_mask = np.ones_like(k_arr, dtype=bool)
        band_mask[0] = False  # exclude DC

    amp_band = np.zeros(len(t))
    for i in range(len(t)):
        Jc = J[:, i] - J[:, i].mean()
        spec = np.abs(np.fft.rfft(Jc))**2
        amp_band[i] = float(spec[band_mask].mean()) if band_mask.any() else 0.0
    amp_band = np.maximum(amp_band, 1e-300)

    ax.semilogy(t, amp_band, "b-", lw=1.0, label=r"sim")

    # Overlay theoretical exponential growth, clipped to data range
    if "sigma_star" in info and amp_band.max() > 0:
        sigma = info["sigma_star"]
        sat = amp_band.max()
        # Anchor at the first time the spectral amplitude has clearly
        # risen above noise (3× the early floor)
        floor = amp_band[:max(1, len(t)//100)].mean()
        rising = np.where(amp_band > 3 * floor)[0]
        if rising.size > 0:
            i_anchor = int(rising[0])
        else:
            i_anchor = max(1, len(t) // 20)
        t_anchor = t[i_anchor]
        amp_anchor = amp_band[i_anchor]
        # Draw the line until it would overshoot saturation by ~3×
        if sigma > 0 and amp_anchor > 0:
            t_line_end = t_anchor + np.log(3 * sat / amp_anchor) / (2 * sigma)
        else:
            t_line_end = t[-1]
        mask = (t >= t_anchor) & (t <= min(t_line_end, t[-1]))
        if mask.sum() >= 2:
            t_theory = t[mask]
            amp_theory = amp_anchor * np.exp(2 * sigma * (t_theory - t_anchor))
            ax.semilogy(t_theory, amp_theory, "r--", lw=1.0,
                        label=rf"$\exp(2\sigma^*\tau)$, $\sigma^*$={sigma:.0f}")
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)

    # Pin y-limits to the data range (avoid overflow autoscale)
    y_lo = max(amp_band[amp_band > 0].min() / 10.0, 1e-12)
    y_hi = amp_band.max() * 10.0
    ax.set_ylim(y_lo, y_hi)
    # Zoom x-axis to the exponential-growth window
    # (t_end=0.5, but linear regime ends at ≈ 0.002 with σ*≈3850).
    # Show 0 to 5× the saturation time so both growth and saturation
    # onset are visible.
    # Show ≳11 e-folds (≈5 decades of growth) plus the saturation knee.
    if "sigma_star" in info and info["sigma_star"] > 0:
        t_zoom = min(11.0 / info["sigma_star"], t[-1])
    else:
        t_zoom = t[-1]
    ax.set_xlim(0, t_zoom)
    ax.set_xlabel(r"Time $\tau$", fontsize=8)
    ax.set_ylabel(rf"$\langle |\hat J(k)|^2\rangle_{{[0.7,1.3]\,k^*}}$",
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(r"Spectral growth at $k\approx k^*$", fontsize=9)
    add_label(ax, "(c)")

    # (d) Fourier power spectrum |Ĵ(k)|² at selected times
    ax = axes[1, 1]
    for i, idx in enumerate(t_indices[1:]):  # skip τ=0
        Jc = J[:, idx] - np.mean(J[:, idx])
        spectrum = np.abs(np.fft.rfft(Jc))**2
        ax.semilogy(k_arr[1:], spectrum[1:], color=colors[i+1], lw=0.7,
                    label=rf"$\tau$={t[idx]:.3f}")
    if kstar is not None:
        ax.axvline(kstar, color="r", ls=":", lw=0.8,
                   label=rf"$k^*$={kstar:.0f}")
    ax.set_xlabel(r"Wavenumber $k$", fontsize=8)
    ax.set_ylabel(r"$|\hat J(k)|^2$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", framealpha=0.85,
              handlelength=1.3, borderpad=0.3)
    ax.set_title("Fourier spectrum", fontsize=9)
    ax.set_xlim(0, min(3 * kstar if kstar else 300, k_arr[-1]))
    add_label(ax, "(d)")

    path = os.path.join(outdir, "fig7.pdf")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: Survey the spinodal region
    scan_spinodal_region()

    # Step 2: Run simulation at a good point
    # Parameters chosen to match the analytical predictions reported in
    # the appendix: ℓ=0.01 ⇒ k* ≈ 94, λ* ≈ 0.067 H₀, σ* ≈ 3850.
    # N=301 gives dx/ℓ ≈ 0.33 (well-resolved interface).
    result = run_spinodal(
        N=301,           # fine mesh; dx/ℓ ≈ 0.33 (well-resolved)
        ell=0.01,        # canonical interface parameter, matches main text
        J0=0.30,         # collapsed state (inside spinodal)
        theta0=1.5,      # above LCST
        perturbation=0.001,  # δJ/J₀ ≈ 0.1%, matches main text claim
        t_end=0.5,       # short time (spinodal is fast, σ* ~ 10³)
    )

    if result is not None:
        # Save into the paper's figure tree so main.tex picks it up.
        out = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "Figure", "pub"))
        plot_spinodal(result, outdir=out)
        print("\nDone.")
