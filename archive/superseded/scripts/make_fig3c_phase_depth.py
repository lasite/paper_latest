#!/usr/bin/env python3
"""
make_fig3c_phase_depth.py — Fig 3(c): limit cycles overlaid on the LCST
bistable equilibrium curve.

Background: the locus μ(J, θ) = μ_b (cold-bath chemical potential), drawn
in the (J/J_eq, θ) plane. The curve is the bistable S-shape of an LCST
gel — a swollen branch (lower θ, large J), an unstable middle branch
(short, dashed), and a collapsed branch (upper θ, small J), joined at
two folds.

Foreground: single-cycle (J/J_eq, θ) trajectories at ξ ∈ {0, 0.4, 0.7,
0.9, 1.0}. The shrinking loops as ξ decreases make the spatial split
tangible:
  • core (small ξ): tiny, near-harmonic loop on the swollen branch;
  • mechanical halo (ξ ≈ ξ_peak): elongated loop that overshoots toward
    the upper fold but does not jump branches;
  • collapse front (ξ → 1): full relaxation orbit that bridges both
    branches — the trajectory traverses the unstable middle region as
    the gel periodically commits to a full LCST collapse.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from style_pub import set_style, add_panel_label
from fig3_data import panel_data, save_panel, PHASE_XI
set_style()


def _one_cycle(t, J_seg, th_seg):
    """Restrict (t, J, θ) to a single peak-to-peak cycle in J at the surface."""
    pks, _ = find_peaks(J_seg, prominence=0.05, distance=20)
    if len(pks) >= 2:
        i0, i1 = pks[0], pks[1] + 1
        return t[i0:i1], J_seg[i0:i1], th_seg[i0:i1]
    return t, J_seg, th_seg


def _nearest_index(x, xi):
    return int(np.argmin(np.abs(x - xi)))


def panel_c(ax, x, t, J, theta, J_eq, eq_curve=None, depths=PHASE_XI,
            label_fs=9, tick_fs=7, legend_fs=6, cmap_name="cividis"):
    """Overlay limit cycles on the LCST bistable equilibrium curve.

    Returns the list of indices used (for caller annotations).
    """
    cmap = plt.get_cmap(cmap_name)

    # ── Background: bistable equilibrium locus μ(J,θ)=μ_b ─────────────
    if eq_curve is not None:
        th_eq = eq_curve["theta"]
        sw = eq_curve["swollen"]    / J_eq
        md = eq_curve["middle"]     / J_eq
        co = eq_curve["collapsed"]  / J_eq
        # Stable branches solid (dark grey); unstable middle dotted.
        # No legend entry — annotated inline below.
        ax.plot(sw, th_eq, color="0.30", lw=1.4, ls="-", zorder=10,
                alpha=0.85)
        ax.plot(co, th_eq, color="0.30", lw=1.4, ls="-", zorder=10,
                alpha=0.85)
        ax.plot(md, th_eq, color="0.30", lw=1.1, ls=(0, (1.5, 1.5)),
                zorder=10, alpha=0.85)

    # ── Limit cycles at requested depths ──────────────────────────────
    t_seg, J_surf_seg, th_surf_seg = _one_cycle(t, J[-1], theta[-1])
    i0 = int(np.searchsorted(t, t_seg[0], side="left"))
    i1 = i0 + len(t_seg)

    used_idx = []
    n = len(depths)
    for k, xi_target in enumerate(depths):
        i = _nearest_index(x, xi_target)
        used_idx.append(i)
        c = cmap(0.10 + 0.80 * k / max(1, n - 1))
        if xi_target <= 0.5:
            lbl = rf"$\xi=0$ (core)"
        else:
            lbl = rf"$\xi=1$ (surface)"
        ax.plot(J[i, i0:i1] / J_eq, theta[i, i0:i1],
                color=c, lw=1.6, alpha=0.95, zorder=3 + k,
                label=lbl)

    # Cold-bath equilibrium reference at J/J_eq = 1 (θ = 0)
    ax.axvline(1.0, color="k", lw=0.5, ls=":", alpha=0.5, zorder=2)

    ax.set_xlabel(r"$J/J_\mathrm{eq}$", fontsize=label_fs)
    ax.set_ylabel(r"$\theta$", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs, direction="out", length=2.5)

    # View window — include the equilibrium curve and all limit cycles.
    Jn_min = 0.05
    Jn_max = float(np.max([(J[i, i0:i1] / J_eq).max() for i in used_idx])) + 0.10
    th_min_v = 0.0
    th_max_v = float(np.max([theta[i, i0:i1].max() for i in used_idx])) + 0.30
    ax.set_xlim(Jn_min, Jn_max)
    ax.set_ylim(th_min_v, th_max_v)

    ax.legend(fontsize=legend_fs, loc="upper right",
              handlelength=1.2, framealpha=0.85, borderpad=0.2,
              labelspacing=0.22, ncol=1)
    return used_idx


def main():
    d = panel_data()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.16)
    panel_c(ax, d["x"], d["t"], d["J"], d["theta"], d["J_eq"],
            eq_curve=d["eq_curve"])
    add_panel_label(ax, "c")
    save_panel(fig, "fig3c_phase_depth")
    plt.close(fig)


if __name__ == "__main__":
    main()
