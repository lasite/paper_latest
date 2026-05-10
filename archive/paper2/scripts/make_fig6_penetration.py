#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig6_penetration.py — Fig.6: Reactant penetration analysis & optimization.

Layout: double-column (6.875 in), 2×2 panels.
  (a) [top-left]     u_ctr vs Da: cold-phase (orange) and heating-phase min (blue)
  (b) [top-right]    Accessibility front depth x* vs Da + Thiele scaling
  (c) [bottom-left]  Da × Gamma_A mechanism map (accessibility vs depletion dominated)
  (d) [bottom-right] Spatial profiles u(ξ): original vs optimized parameters

Data source: 1D slab simulate() with varying Da and Gamma_A.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, save)
from scan_optimized import Params, simulate, finalize_params

set_style()


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def run_and_analyze(Da, Gamma_A=1.5, D0=2.0, Bi_c=0.70, Bi_T=0.10,
                    S_chi=1.0, N=21, t_end=150, n_save=1500):
    """Run a simulation and extract accessibility metrics."""
    p = Params(Da=Da, S_chi=S_chi, Bi_c=Bi_c, Bi_T=Bi_T,
               Gamma_A=Gamma_A, D0=D0, N=N, t_end=t_end, n_save=n_save)
    try:
        d = simulate(p)
    except Exception as e:
        return None

    t = d["t"]
    J = d["J"]       # (N, n_save)
    u = d["u"]
    theta = d["theta"]
    x = d["x"]

    # Use last 40% for analysis
    i0 = int(0.60 * len(t))
    J_tail = J[:, i0:]
    u_tail = u[:, i0:]
    theta_tail = theta[:, i0:]
    t_tail = t[i0:]

    # Volume-averaged theta
    theta_mean = np.mean(theta_tail, axis=0)

    # Check oscillation
    peaks, _ = find_peaks(theta_mean, prominence=0.05)
    if len(peaks) < 2:
        return {"oscillates": False, "Da": Da, "Gamma_A": Gamma_A}

    # Centre node (index 0)
    u_ctr = u_tail[0, :]
    u_ctr_cold_p90 = np.percentile(u_ctr, 90)
    u_ctr_heat_min = np.min(u_ctr)

    # Surface node (index -1)
    u_surf = u_tail[-1, :]

    # Accessibility front depth x*: fraction with u > 0.05 at collapse
    troughs, _ = find_peaks(-theta_mean, prominence=0.05)
    if len(troughs) > 0:
        i_collapse = troughs[-1]
        u_at_collapse = u_tail[:, i_collapse]
        x_star = np.sum(u_at_collapse > 0.05) / len(x)
    else:
        x_star = 0.0

    # J amplitude
    J_mean = np.mean(J_tail, axis=0)
    J_amp = np.max(J_mean) - np.min(J_mean)

    return {
        "oscillates": True,
        "Da": Da,
        "Gamma_A": Gamma_A,
        "u_ctr_cold_p90": u_ctr_cold_p90,
        "u_ctr_heat_min": u_ctr_heat_min,
        "x_star": x_star,
        "J_amp": J_amp,
        "u_surf_mean": np.mean(u_surf),
    }


# ══════════════════════════════════════════════════════════════════
# Panel (a): u_ctr vs Da
# ══════════════════════════════════════════════════════════════════

def panel_uctr_vs_Da(ax, results_Da):
    Da_arr = []
    uctr_cold = []
    uctr_heat = []
    no_osc_Da = []

    for r in results_Da:
        if r is None or not r["oscillates"]:
            if r is not None:
                no_osc_Da.append(r["Da"])
            continue
        Da_arr.append(r["Da"])
        uctr_cold.append(r["u_ctr_cold_p90"])
        uctr_heat.append(r["u_ctr_heat_min"])

    ax.semilogy(Da_arr, uctr_cold, "o-", color=C[3], ms=4, lw=1.0,
                label=r"$u_\mathrm{ctr}$ cold (P90)")
    ax.semilogy(Da_arr, uctr_heat, "s-", color=C[0], ms=4, lw=1.0,
                label=r"$u_\mathrm{ctr}$ heat (min)")
    for d in no_osc_Da:
        ax.axvline(d, color="0.7", lw=0.5, ls=":")

    # Analytical Da*
    phi0 = 0.15; J_ctr = 1.15; m_act = 6; delta = 0.08; D0 = 2.0
    Gamma_A = 1.5; theta_h = 1.5
    Da_star = delta * D0 / ((1 - phi0/J_ctr)**m_act *
              np.exp(Gamma_A * theta_h / (1 + theta_h)))
    ax.axvline(Da_star, color=C[1], ls=":", lw=0.8,
               label=rf"$\mathrm{{Da}}^*={Da_star:.2f}$")

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"Centre reactant $u_\mathrm{ctr}$")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(bottom=1e-8)


# ══════════════════════════════════════════════════════════════════
# Panel (b): x* vs Da
# ══════════════════════════════════════════════════════════════════

def panel_xstar_vs_Da(ax, results_Da):
    Da_arr = []
    xstar_arr = []

    for r in results_Da:
        if r is None or not r["oscillates"]:
            continue
        Da_arr.append(r["Da"])
        xstar_arr.append(r["x_star"])

    ax.plot(Da_arr, xstar_arr, "o-", color=C[0], ms=4, lw=1.0,
            label=r"$x^*$ (simulation)")

    # Analytical Thiele scaling
    Da_fit = np.linspace(0.2, max(Da_arr), 100)
    delta = 0.08; D0 = 2.0
    alpha_h = 0.8; f_h = 2.0  # approximate mid-heating values
    xstar_fit = np.sqrt(delta * D0 / (Da_fit * alpha_h * f_h))
    xstar_fit = np.clip(xstar_fit, 0, 1)
    ax.plot(Da_fit, xstar_fit, "--", color=C[1], lw=0.8,
            label=r"$\Phi^{-1}$ scaling")

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"Front depth $x^*/H_0$")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(bottom=0)


# ══════════════════════════════════════════════════════════════════
# Panel (c): Da × Gamma_A mechanism map
# ══════════════════════════════════════════════════════════════════

def panel_Da_GA_map(ax, results_2d, Da_vals, GA_vals):
    nDa = len(Da_vals)
    nGA = len(GA_vals)

    # Classification: 0=no osc, 1=accessibility, 2=depletion
    img = np.full((nGA, nDa), 0)

    for i, ga in enumerate(GA_vals):
        for j, da in enumerate(Da_vals):
            r = results_2d.get((da, ga))
            if r is None or not r["oscillates"]:
                img[i, j] = 0
            elif r["u_ctr_cold_p90"] > 0.05:
                img[i, j] = 1  # accessibility-dominated
            else:
                img[i, j] = 2  # depletion-dominated

    cmap = ListedColormap([C[1], C[3], C[2]])  # red, orange, green
    extent = [Da_vals[0], Da_vals[-1], GA_vals[0], GA_vals[-1]]
    ax.imshow(img, origin="lower", aspect="auto", extent=extent,
              cmap=cmap, vmin=0, vmax=2, rasterized=True,
              interpolation="nearest")

    # Analytical Da* boundary
    phi0 = 0.15; J_ctr = 1.15; m_act = 6; delta = 0.08; D0 = 2.0
    theta_h = 1.5
    GA_line = np.linspace(GA_vals[0], GA_vals[-1], 100)
    Da_star_line = delta * D0 / ((1 - phi0/J_ctr)**m_act *
                   np.exp(GA_line * theta_h / (1 + theta_h)))
    ax.plot(Da_star_line, GA_line, "k--", lw=1.0,
            label=r"$\mathrm{Da}^*(\Gamma_A)$")

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"$\Gamma_A$")
    ax.set_title("Mechanism map", fontsize=8)

    # Manual legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=C[1], label="No oscillation"),
        mpatches.Patch(color=C[3], label="Accessibility-dom."),
        mpatches.Patch(color=C[2], label="Depletion-dom."),
    ]
    ax.legend(handles=patches, fontsize=5.5, loc="upper right")


# ══════════════════════════════════════════════════════════════════
# Panel (d): u(x) profiles: original vs optimized
# ══════════════════════════════════════════════════════════════════

def panel_u_profiles(ax):
    """Compare u(x) spatial profiles for original vs optimized params."""
    configs = [
        {"label": r"Original ($\mathrm{Da}=14$, $D_0=1$)",
         "Da": 14.0, "D0": 1.0, "Bi_c": 0.35, "color": C[1], "ls": "--"},
        {"label": r"Optimized ($\mathrm{Da}=4$, $D_0=2$)",
         "Da": 4.0, "D0": 2.0, "Bi_c": 0.70, "color": C[0], "ls": "-"},
    ]

    for cfg in configs:
        p = Params(Da=cfg["Da"], D0=cfg["D0"], Bi_c=cfg["Bi_c"],
                   S_chi=1.0, Bi_T=0.10, Gamma_A=1.5,
                   N=21, t_end=150, n_save=1500)
        try:
            d = simulate(p)
            x = d["x"]
            u = d["u"]
            # Time-average over last 40%
            i0 = int(0.60 * u.shape[1])
            u_mean = np.mean(u[:, i0:], axis=1)
            u_peak = np.max(u[:, i0:], axis=1)
            ax.semilogy(x, u_peak, color=cfg["color"], ls=cfg["ls"],
                        lw=1.0, label=cfg["label"])
        except Exception as e:
            print(f"  Profile sim failed: {e}")

    ax.set_xlabel(r"$\xi = x/H_0$")
    ax.set_ylabel(r"Peak $u(\xi)$")
    ax.set_title(r"$590\times$ enhancement", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    ax.set_ylim(bottom=1e-7)


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.6
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()

    # ── 1D Da scan ────────────────────────────────────────────────
    Da_scan = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
               7.0, 10.0]
    print(f"  Running Da scan ({len(Da_scan)} points)...")
    results_Da = []
    for da in Da_scan:
        print(f"    Da={da:.2f}...", end=" ", flush=True)
        r = run_and_analyze(da, Gamma_A=1.5)
        results_Da.append(r)
        if r and r["oscillates"]:
            print(f"osc, u_ctr={r['u_ctr_cold_p90']:.2e}, x*={r['x_star']:.3f}")
        else:
            print("no osc" if r else "failed")

    # ── 2D Da × Gamma_A map ──────────────────────────────────────
    Da_2d = [0.5, 2.0, 4.0, 6.0, 8.0]
    GA_2d = [0.5, 1.5, 3.0, 5.0]
    print(f"  Running Da×GA map ({len(Da_2d)}×{len(GA_2d)} = "
          f"{len(Da_2d)*len(GA_2d)} points)...")
    results_2d = {}
    for ga in GA_2d:
        for da in Da_2d:
            print(f"    Da={da:.1f}, GA={ga:.1f}...", end=" ", flush=True)
            r = run_and_analyze(da, Gamma_A=ga)
            results_2d[(da, ga)] = r
            if r and r["oscillates"]:
                print(f"osc, u_ctr={r['u_ctr_cold_p90']:.2e}")
            else:
                print("no osc" if r else "failed")

    # ── Assemble figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(PRE_DOUBLE, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40,
                          left=0.08, right=0.97, top=0.96, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    print("  Plotting panels...")
    panel_uctr_vs_Da(ax_a, results_Da)
    add_panel_label(ax_a, "a")

    panel_xstar_vs_Da(ax_b, results_Da)
    add_panel_label(ax_b, "b")

    panel_Da_GA_map(ax_c, results_2d, Da_2d, GA_2d)
    add_panel_label(ax_c, "c")

    panel_u_profiles(ax_d)
    add_panel_label(ax_d, "d")

    save(fig, "fig6_penetration")
    print("Fig.6 done.")


if __name__ == "__main__":
    main()
