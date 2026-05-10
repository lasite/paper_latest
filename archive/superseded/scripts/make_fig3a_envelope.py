#!/usr/bin/env python3
"""
make_fig3a_envelope.py — Fig 3(a): J/J_eq envelope and the three-zone split.

Plots min/max of J(ξ,τ) over the analysis window normalized by the cold-
bath equilibrium  J_eq = J(θ=0, μ=μ_b).  In these units, J/J_eq=1
corresponds to the gel sitting in equilibrium with the bath at θ=0.

Three zones are shaded:
  core  (ξ < ξ_peak):   passive thermal mode, swelling small
  halo  (ξ_peak ≤ ξ ≤ ξ_LCST):  mechanical halo where the LCST collapse
                                front overshoots inward but the gel
                                does not itself cross LCST
  front (ξ > ξ_LCST):   propagating LCST collapse front (φ→1 each cycle)

The star at ξ_peak marks the position where J_max(ξ) is maximal — the
inner reach of the mechanical halo, a coordinate-invariant critical
point that does not depend on the kinematic reference state.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_pub import set_style, add_panel_label
from fig3_data import panel_data, save_panel
set_style()


CORE_FACE  = "#cfe3f6"   # pale blue   — passive core
HALO_FACE  = "#fff0c2"   # pale yellow — mechanical halo
FRONT_FACE = "#fde0d3"   # pale red    — collapse front
JMAX_C     = "#1f5fa3"   # blue, J_max curve
JMIN_C     = "#a23e1c"   # red,  J_min curve


def panel_a(ax, x, J_min, J_max, J_eq, xi_peak, xi_LCST,
            label_fs=9, tick_fs=7, annot_fs=7):
    """Draw the J/J_eq envelope panel with three zone shading."""
    Jmn = J_min / J_eq
    Jmx = J_max / J_eq

    ax.axvspan(0.0, xi_peak,            color=CORE_FACE,  alpha=0.55, lw=0, zorder=0)
    ax.axvspan(xi_peak, xi_LCST,        color=HALO_FACE,  alpha=0.65, lw=0, zorder=0)
    ax.axvspan(xi_LCST, 1.0,            color=FRONT_FACE, alpha=0.65, lw=0, zorder=0)

    ax.fill_between(x, Jmn, Jmx, color="0.78", alpha=0.55, lw=0, zorder=1)
    ax.plot(x, Jmx, color=JMAX_C, lw=1.4,
            label=r"$J_\max(\xi)/J_\mathrm{eq}$", zorder=3)
    ax.plot(x, Jmn, color=JMIN_C, lw=1.4,
            label=r"$J_\min(\xi)/J_\mathrm{eq}$", zorder=3)

    # Cold-equilibrium reference
    ax.axhline(1.0, color="k", lw=0.6, ls=":", zorder=2)

    # Two boundary guides
    ax.axvline(xi_peak, color="k",       lw=0.7, ls="--", zorder=2)
    ax.axvline(xi_LCST, color="0.35",    lw=0.7, ls=":",  zorder=2)

    # Star at the J_max(ξ) peak — the mechanical critical point
    j_peak_norm = float(np.max(Jmx))
    ax.plot([xi_peak], [j_peak_norm], marker="*", ms=9,
            color="#222", mec="k", mew=0.4, zorder=5)
    ax.annotate(rf"$\xi_\mathrm{{peak}}\!\approx\!{xi_peak:.2f}$",
                xy=(xi_peak, j_peak_norm),
                xytext=(xi_peak - 0.30, 0.40 * j_peak_norm),
                fontsize=annot_fs, ha="center",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5))

    # Zone labels
    y_top = max(1.7, j_peak_norm * 1.05)
    ax.text(0.5 * xi_peak, y_top * 0.95, "core",
            ha="center", fontsize=annot_fs, color="#1a4d77", weight="bold")
    ax.text(0.5 * (xi_peak + xi_LCST), y_top * 0.90, "halo",
            ha="center", fontsize=annot_fs - 0.5, color="#9b6e16",
            weight="bold")
    ax.text(0.5 * (xi_LCST + 1.0), y_top * 0.95, "front",
            ha="center", fontsize=annot_fs, color="#a23e1c", weight="bold")

    ax.set_xlabel(r"$\xi$", fontsize=label_fs)
    ax.set_ylabel(r"$J/J_\mathrm{eq}$", fontsize=label_fs)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.7, j_peak_norm * 1.10))
    ax.tick_params(labelsize=tick_fs, direction="out", length=2.5)
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9,
              handlelength=1.4, borderpad=0.3)


def main():
    d = panel_data()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.18)
    panel_a(ax, d["x"], d["J_min"], d["J_max"], d["J_eq"],
            d["xi_peak"], d["xi_LCST"])
    add_panel_label(ax, "a")
    save_panel(fig, "fig3a_envelope")
    plt.close(fig)


if __name__ == "__main__":
    main()
