#!/usr/bin/env python3
"""
make_fig3e_heat.py — Fig 3(e): heat source and reactant profiles vs ξ.

Time-averaged local heat source ⟨Da·J·R⟩(ξ) on the right axis (log) and
time-averaged reactant ⟨u⟩(ξ) on the left axis (log) over the analysis
window. Together they show that the reaction is geometrically confined
to the outer ~20 % of the gel, that the same shell is the only region
where reactant ever reaches appreciable concentration, and that the
temperature oscillation observed in the core (panel b) is therefore
sustained by inward heat conduction from this thin shell rather than by
local reaction.
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


C_HEAT = "#a23e1c"
C_U    = "#137a73"


def panel_e(ax, x, u_mean, heat_mean, xi_peak, xi_LCST=None,
            label_fs=9, tick_fs=7, legend_fs=6.5, floor=1e-8):
    """Plot ⟨u⟩(ξ) and ⟨Da·J·R⟩(ξ) on twin log axes."""
    u_p = np.maximum(u_mean, floor)
    h_p = np.maximum(heat_mean, floor)

    l1, = ax.plot(x, u_p, color=C_U, lw=1.5, label=r"$\langle u\rangle$")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(floor, 2.0)
    ax.set_xlabel(r"$\xi$", fontsize=label_fs)
    ax.set_ylabel(r"$\langle u\rangle$", color=C_U, fontsize=label_fs)
    ax.tick_params(axis="y", colors=C_U, labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=tick_fs, direction="out", length=2.5)

    ax2 = ax.twinx()
    l2, = ax2.plot(x, h_p, color=C_HEAT, lw=1.5,
                   label=r"$\langle\mathrm{Da}\,J\,R\rangle$")
    ax2.set_yscale("log")
    ax2.set_ylim(floor, max(5.0, h_p.max() * 1.5))
    ax2.set_ylabel(r"$\langle\mathrm{Da}\,J\,R\rangle$",
                   color=C_HEAT, fontsize=label_fs,
                   rotation=270, labelpad=12)
    ax2.tick_params(axis="y", colors=C_HEAT, labelsize=tick_fs)

    ax.axvline(xi_peak, color="k", lw=0.7, ls="--", zorder=5)
    if xi_LCST is not None:
        ax.axvline(xi_LCST, color="0.35", lw=0.7, ls=":", zorder=5)

    ax.legend(handles=[l1, l2], fontsize=legend_fs, loc="upper left",
              framealpha=0.9, handlelength=1.4, borderpad=0.3,
              labelspacing=0.25)


def main():
    d = panel_data()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.subplots_adjust(left=0.18, right=0.83, top=0.95, bottom=0.18)
    panel_e(ax, d["x"], d["u_mean"], d["heat_mean"],
            d["xi_peak"], d["xi_LCST"])
    add_panel_label(ax, "e")
    save_panel(fig, "fig3e_heat")
    plt.close(fig)


if __name__ == "__main__":
    main()
