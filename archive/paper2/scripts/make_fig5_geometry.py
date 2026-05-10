#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig5_geometry.py — Fig.5: Geometry effect: slab (oscillates) vs sphere (steady).

Layout: double-column (6.875 in), 2 rows.
  Row 1: (a) Geometry schematic (full width, drawn with matplotlib)
  Row 2: (b) ε_diff scan — sphere oscillation amplitude vs internal diffusion coupling
          (c) Slab kymograph J(x,τ) — Regime I (oscillatory reference)
          (d) Sphere kymograph J(r,τ) at ε_diff=0 (oscillates)
          (e) Sphere kymograph J(r,τ) at ε_diff=1e-3 (steady swollen)

Sphere model: embedded simplified version parametrized by eps_diff.
  eps_diff = 0     → no internal coupling → surface-only reaction → OSCILLATES (Δθ≈0.23)
  eps_diff = 1e-3  → normal diffusion → front propagates inward → STEADY SWOLLEN
  Critical ε* ≈ 3×10⁻⁴

Reference data (N=21, t_end=800): eps: 0→0.227, 1e-6→0.203, 1e-5→0.121,
    1e-4→0.085, 1e-3→0.001 (stable), ε* ≈ 3e-4.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Wedge
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import lil_matrix, csc_matrix

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, kymo_show, save)
from scan_optimized import Params, simulate

set_style()

_LOG_J_MAX = np.log(8.0)
_LOG_J_MIN = np.log(0.153)


# ══════════════════════════════════════════════════════════════════
# Embedded sphere model (parametrized by eps_diff)
# ══════════════════════════════════════════════════════════════════

class _Grid:
    def __init__(self, N):
        self.N   = N
        self.dR  = 1.0 / N
        self.R_f = np.linspace(0, 1, N + 1)
        self.R_c = 0.5 * (self.R_f[:-1] + self.R_f[1:])
        self.R2_f = self.R_f ** 2
        self.V   = (self.R_f[1:] ** 3 - self.R_f[:-1] ** 3) / 3.0
        self.Vs  = np.maximum(self.V, 1e-30)

    def div(self, f):
        return (self.R2_f[1:] * f[1:] - self.R2_f[:-1] * f[:-1]) / self.Vs


def _chem_pot(J, theta):
    phi = np.clip(0.15 / np.maximum(J, 0.153), 0, 0.999)
    chi = 0.40 + 0.50 * theta + 0.80 * phi
    return (np.log(np.maximum(1 - phi, 1e-15)) + phi
            + chi * phi ** 2 + 0.10 * (J ** (-1.0/3.0) - 1.0 / J))


def _reaction_rate(u, theta, J):
    phi = np.clip(0.15 / np.maximum(J, 0.153), 0, 0.999)
    denom = 1.0 + 0.03 * np.maximum(theta, -30.0)
    return (np.maximum(u, 1e-12)
            * np.maximum(1 - phi, 1e-12) ** 4
            * np.exp(np.clip(theta / denom, -50, 50)))


_M_B = float(_chem_pot(np.array([1.5]), np.array([0.0]))[0])


def _rhs_sphere(t, y, g, eps_diff, Da=2.5, Bi_T=0.80, Bi_c=1.0):
    """
    Sphere RHS matching the original coupling_transition model exactly.

    Exchange with the bath is implemented as VOLUME-DISTRIBUTED bulk terms
    (mean-field surface exchange):
      - solvent: -J23*(m - m_b)/J  (drives J toward equilibrium)
      - reactant: +J23*(1 - u)     (replenishes from bath at concentration=1)
      - heat:    -Bi_T*J23*theta   (Newton cooling)

    eps_diff controls INTERNAL spatial coupling (diffusion between cells).
    All face fluxes at r=0 and r=R are zero (no Robin flux BCs needed).
    """
    N  = g.N
    lJ = np.clip(y[:N], _LOG_J_MIN, _LOG_J_MAX)
    W  = y[N:2*N]
    th = np.clip(y[2*N:], -10, 25)
    J  = np.exp(lJ)
    u  = np.maximum(W / J, 1e-12)
    m  = _chem_pot(J, th)
    J23 = J ** (2.0 / 3.0)
    R_  = _reaction_rate(u, th, J)

    # Solvent: internal diffusion only; zero flux at r=0 and r=R
    q = np.zeros(N + 1)
    q[1:N] = -eps_diff * (m[1:] - m[:-1]) / g.dR

    # Reactant: internal diffusion only
    nf = np.zeros(N + 1)
    nf[1:N] = -eps_diff * 0.1 * (u[1:] - u[:-1]) / g.dR

    # Heat: internal diffusion only
    h = np.zeros(N + 1)
    h[1:N] = -eps_diff * 0.3 * (th[1:] - th[:-1]) / g.dR

    # Volume-distributed exchange with bath (mean-field BCs)
    lJt = -g.div(q) / J - J23 * (m - _M_B) / J
    Wt  = -g.div(nf) - Da * J * R_  + Bi_c * J23 * (1.0 - u)
    tht = -g.div(h)  + Da * R_       - Bi_T * J23 * th

    return np.concatenate([lJt, Wt, tht])


def _make_jac_sparsity(N):
    sz = 3 * N
    S  = lil_matrix((sz, sz))
    for kr, kc, w in [(0, 0, 2), (0, 2, 1),
                       (1, 0, 2), (1, 1, 1), (1, 2, 1),
                       (2, 2, 1)]:
        for i in range(N):
            for dj in range(-w, w + 1):
                j = i + dj
                if 0 <= j < N:
                    S[kr*N + i, kc*N + j] = 1
    return csc_matrix(S)


def run_sphere(eps_diff, N=31, t_end=800, n_pts=6000):
    g   = _Grid(N)
    R   = g.R_c
    J0  = 0.35 + 0.01 * np.cos(np.pi * R)
    u0  = 0.50 + 0.02 * (1 - R)
    th0 = 1.70 + 0.02 * (1 - R ** 2)
    y0  = np.concatenate([
        np.log(np.maximum(J0, 0.16)),
        np.maximum(J0, 0.16) * np.maximum(u0, 1e-12),
        th0,
    ])
    Sp = _make_jac_sparsity(N)
    sol = solve_ivp(
        lambda t, y: _rhs_sphere(t, y, g, eps_diff),
        (0, t_end), y0,
        method="BDF",
        jac_sparsity=Sp,
        t_eval=np.linspace(0, t_end, n_pts),
        rtol=1e-6, atol=1e-8, max_step=0.4,
    )
    if not sol.success:
        print(f"    eps={eps_diff:.0e}: FAILED — {sol.message}")
        return None
    J  = np.exp(np.clip(sol.y[:N], _LOG_J_MIN, _LOG_J_MAX))
    th = sol.y[2*N:]
    return {"t": sol.t, "J": J, "theta": th, "r": R}


def osc_amplitude(t, signal, frac=0.5):
    i0  = int(frac * len(t))
    sig = signal[i0:]
    sig_d = sig - np.polyval(np.polyfit(t[i0:], sig, 1), t[i0:])
    amp = float(np.max(sig_d) - np.min(sig_d))
    pks, _ = find_peaks(sig_d, prominence=max(0.1 * amp, 1e-3),
                        distance=max(3, len(sig_d) // 20))
    return amp, len(pks) >= 2


# ══════════════════════════════════════════════════════════════════
# Panel (a): Geometry schematic
# ══════════════════════════════════════════════════════════════════

def draw_geometry_schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── LEFT: Slab ────────────────────────────────────────────────
    # Bath (left of slab at x=0, not shown since symmetric)
    slab = FancyBboxPatch((0.2, 0.0), 2.0, 2.2,
                          boxstyle="round,pad=0.05",
                          facecolor="#c8e6fa", edgecolor="#1f77b4", lw=1.2)
    ax.add_patch(slab)
    # Reaction zone (right edge of slab)
    rxn_slab = FancyBboxPatch((1.8, 0.0), 0.40, 2.2,
                              boxstyle="square,pad=0",
                              facecolor="#ffd699", edgecolor="none", alpha=0.75)
    ax.add_patch(rxn_slab)
    ax.text(1.2, 1.1, "slab", ha="center", va="center",
            fontsize=8, color="#1f77b4", fontweight="bold")

    # Surface exchange arrows (only at x=H₀, right side)
    for y in [0.4, 1.1, 1.8]:
        ax.annotate("", xy=(2.20, y), xytext=(2.55, y),
                    arrowprops=dict(arrowstyle="<|-", color="#2ca02c",
                                    lw=0.9, mutation_scale=8))
    ax.text(2.75, 1.1, r"H$_2$O$_2$", ha="left", va="center",
            fontsize=6.5, color="#2ca02c")
    # Symmetry (left)
    ax.plot([0.2, 0.2], [0.0, 2.2], color="#555", lw=0.8, ls="--")
    ax.text(0.2, -0.4, r"$x=0$ (sym.)", ha="center", fontsize=6.5, color="#555")
    ax.text(2.2, -0.4, r"$x=H_0$", ha="center", fontsize=6.5, color="#1f77b4")
    ax.text(1.2, 2.45, "→ OSCILLATES", ha="center", fontsize=7.5,
            color="#2ca02c", fontweight="bold")
    ax.text(1.2, -0.75, r"negative feedback $\checkmark$", ha="center", fontsize=6.5,
            color="#2ca02c", style="italic")

    # ── Divider ───────────────────────────────────────────────────
    ax.plot([5.0, 5.0], [-0.6, 2.8], color="lightgray", lw=0.8, ls="-")
    ax.text(5.0, 2.9, "vs", ha="center", va="center", fontsize=8, color="gray")

    # ── RIGHT: Sphere ─────────────────────────────────────────────
    cx, cy = 7.5, 1.1
    sphere_outer = Circle((cx, cy), 1.1, facecolor="#c8e6fa",
                           edgecolor="#1f77b4", lw=1.2)
    sphere_rxn = Wedge((cx, cy), 1.1, 0, 360, width=0.35,
                        facecolor="#ffd699", edgecolor="none", alpha=0.75)
    ax.add_patch(sphere_outer)
    ax.add_patch(sphere_rxn)
    ax.text(cx, cy, "sphere", ha="center", va="center",
            fontsize=8, color="#1f77b4", fontweight="bold")

    # Arrows from all directions
    for ang in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.deg2rad(ang)
        x1 = cx + 1.45 * np.cos(rad)
        y1 = cy + 1.45 * np.sin(rad)
        x2 = cx + 1.15 * np.cos(rad)
        y2 = cy + 1.15 * np.sin(rad)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                    lw=0.7, mutation_scale=7))
    ax.text(cx, 2.45, "→ STEADY", ha="center", fontsize=7.5,
            color="#d62728", fontweight="bold")
    ax.text(cx, -0.75, r"negative feedback $\times$", ha="center", fontsize=6.5,
            color="#d62728", style="italic")


# ══════════════════════════════════════════════════════════════════
# Panel (b): ε_diff scan
# ══════════════════════════════════════════════════════════════════

def panel_eps_scan(ax):
    # Known reference data from transcript (N=21, t_end=800)
    eps_ref  = [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    amp_ref  = [0.203, 0.121, 0.085, 0.001, 0.000, 0.000, 0.000]
    osc_ref  = [True,  True,  True,  False,  False, False, False]

    # Run a few new points at higher N for publication quality
    print("  Running ε_diff scan (N=31, t_end=600) ...")
    eps_scan = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2]
    amp_scan = []
    for eps in eps_scan:
        d = run_sphere(eps, N=31, t_end=600, n_pts=4000)
        if d is None:
            amp_scan.append(0.0)
            continue
        wt   = (d["J"].shape[0]) ** (-1)       # uniform weight (simplified)
        g    = _Grid(31)
        wt   = g.V / g.V.sum()
        th_m = np.sum(d["theta"] * wt[:, None], axis=0)
        amp, is_osc = osc_amplitude(d["t"], th_m)
        amp_scan.append(amp if is_osc else 0.0)
        tag  = "OSC" if is_osc else "stab"
        print(f"    eps={eps:.0e}: {tag}  Δθ={amp:.3f}")
    amp_scan = np.array(amp_scan)

    # Plot
    eps_plot = np.array([max(e, 1e-7) for e in eps_scan])
    ax.semilogx(eps_plot, amp_scan, color=C[0], lw=1.2, marker="o",
                ms=4, label="1D sphere model")
    ax.axhline(0, color="k", lw=0.5)

    # critical eps — transition near 3e-4
    crit = 3e-4
    ax.axvline(crit, color="gray", ls="--", lw=0.8)
    ax.text(crit * 1.8, amp_scan.max() * 0.5 if amp_scan.max() > 0 else 0.1,
            r"$\varepsilon^*\approx 3\times10^{-4}$",
            fontsize=6.5, color="gray", va="center")
    ax.fill_betweenx([0, amp_scan.max() * 1.1 if amp_scan.max() > 0 else 0.3],
                     1e-7, crit, alpha=0.08, color=C[0])
    ax.text(1e-6, amp_scan.max() * 0.85 if amp_scan.max() > 0 else 0.2,
            "Oscillatory", fontsize=6.5, color=C[0])
    ax.text(3e-3, amp_scan.max() * 0.85 if amp_scan.max() > 0 else 0.2,
            "Steady\nswollen", fontsize=6.5, color=C[1], multialignment="center")

    ax.set_xlabel(r"Internal coupling $\varepsilon_\mathrm{diff}$")
    ax.set_ylabel(r"Oscillation amplitude $\Delta\theta$")
    ax.set_title(r"Sphere: coupling suppresses oscillation", fontsize=8)
    print("  ε_diff scan done.")
    return amp_scan


# ══════════════════════════════════════════════════════════════════
# Panels (c,d,e): Kymographs
# ══════════════════════════════════════════════════════════════════

_SLAB_CACHE = os.path.join(os.path.dirname(__file__),
                           "figures_pub/fig5_slab_cache.npz")
_SLAB_CACHE_VERSION = 2


def plot_slab_kymo(ax):
    if os.path.exists(_SLAB_CACHE):
        print("  Loading slab cache ...")
        d = np.load(_SLAB_CACHE)
        if {"x", "t", "J", "cache_version"}.issubset(d.files) and int(np.atleast_1d(d["cache_version"])[0]) == _SLAB_CACHE_VERSION:
            x, t, J = d["x"], d["t"], d["J"]
        else:
            d = None
    else:
        d = None
    if d is None:
        print("  Running slab Regime I for kymograph (Da=4.0, S_chi=0.7, N=40) ...")
        p = Params(Da=4.0, S_chi=0.7, Bi_c=0.70, Bi_T=0.10, Gamma_A=1.5,
                   N=40, t_end=400, n_save=4000,
                   arrh_exp_cap=30.0, max_step=0.25)
        data = simulate(p)
        print(f"  Slab done. nfev={data['nfev']}")
        x, t, J = data["x"], data["t"], data["J"]
        np.savez(_SLAB_CACHE, x=x, t=t, J=J, cache_version=np.array([_SLAB_CACHE_VERSION], dtype=int))

    i0 = int(0.65 * len(t))
    t_tail = t[i0:] - t[i0]
    J_tail = J[:, i0:]

    vmin = np.nanpercentile(J_tail, 2)
    vmax = np.nanpercentile(J_tail, 98)
    kymo_show(ax, J_tail, x, t_tail, cmap="RdBu_r",
              label=r"$J(x,\tau)$", vmin=vmin, vmax=vmax)
    ax.set_title("Slab (oscillating)", fontsize=8)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$x/H_0$")


def plot_sphere_kymo(ax, eps_diff, label=""):
    print(f"  Running sphere (ε_diff={eps_diff:.0e}) ...")
    d = run_sphere(eps_diff, N=31, t_end=600, n_pts=4000)
    if d is None:
        ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes, ha="center")
        return

    i0 = int(0.60 * len(d["t"]))
    t_tail = d["t"][i0:] - d["t"][i0]
    J_tail = d["J"][:, i0:]

    vmin = np.nanpercentile(J_tail, 2)
    vmax = np.nanpercentile(J_tail, 98)
    kymo_show(ax, J_tail, d["r"], t_tail, cmap="RdBu_r",
              label=r"$J(r,\tau)$", vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=8)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$r/R_0$")
    print(f"  Sphere kymograph done.")


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.5
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()

    fig = plt.figure(figsize=(PRE_DOUBLE, 6.2))
    gs  = fig.add_gridspec(
        2, 4,
        height_ratios=[0.9, 1.0],
        hspace=0.52, wspace=0.50,
        left=0.09, right=0.97, top=0.97, bottom=0.09,
    )

    # Row 1: full-width schematic (spanning all 4 cols)
    ax_a = fig.add_subplot(gs[0, :])
    draw_geometry_schematic(ax_a)
    add_panel_label(ax_a, "a", x=-0.02, y=1.04)

    # Row 2: 4 sub-panels
    ax_b  = fig.add_subplot(gs[1, 0])
    ax_c  = fig.add_subplot(gs[1, 1])
    ax_d  = fig.add_subplot(gs[1, 2])
    ax_e  = fig.add_subplot(gs[1, 3])

    panel_eps_scan(ax_b)
    add_panel_label(ax_b, "b")

    plot_slab_kymo(ax_c)
    add_panel_label(ax_c, "c")

    plot_sphere_kymo(ax_d, eps_diff=0.0,
                     label=r"Sphere: $\varepsilon_\mathrm{diff}=0$ (OSC)")
    add_panel_label(ax_d, "d")

    plot_sphere_kymo(ax_e, eps_diff=1e-3,
                     label=r"Sphere: $\varepsilon_\mathrm{diff}=10^{-3}$ (steady)")
    add_panel_label(ax_e, "e")

    save(fig, "fig5_geometry")
    print("Fig.5 done.")


if __name__ == "__main__":
    main()
