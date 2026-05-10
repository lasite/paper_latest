#!/usr/bin/env python3
"""
make_fig3b_envelopes_thu.py — Fig 3(b): θ(ξ) and u(ξ) envelopes.

Stacked twin-axis envelope plot complementing panel (a). The temperature
envelope (left, linear) shows that θ oscillates throughout the gel — even
in the reactant-starved core — because heat conducts in from the surface
reaction zone. The reactant envelope (right, log) shows that u is close to
the bath value only in a thin outer skin and decays exponentially toward
the centre, giving a quantitative picture of the reactant-starvation
mechanism.
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


TH_C = "#d6722e"   # warm orange for θ
U_C  = "#137a73"   # teal for u


def panel_b(ax, x, th_min, th_max, u_min, u_max, xi_peak, xi_LCST=None,
            label_fs=9, tick_fs=7, legend_fs=6.5):
    """Draw the θ/u envelopes on twin y-axes. Returns (ax, ax2).

    The dashed vertical guide at ξ_peak is the J_max-peak (mechanical
    halo inner edge). If xi_LCST is provided, it is drawn as a finer
    dotted line marking the LCST collapse-front locus.
    """
    # θ envelope on the left (linear)
    ax.fill_between(x, th_min, th_max, color=TH_C, alpha=0.30, lw=0)
    ax.plot(x, th_max, color=TH_C, lw=1.3, label=r"$\theta_\max$")
    ax.plot(x, th_min, color=TH_C, lw=1.0, ls="--", label=r"$\theta_\min$")
    ax.set_xlabel(r"$\xi$", fontsize=label_fs)
    ax.set_ylabel(r"$\theta$", color=TH_C, fontsize=label_fs)
    ax.tick_params(axis="y", colors=TH_C, labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=tick_fs, direction="out", length=2.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.05, th_max.max() * 1.05))

    # u envelope on the right (log, since u_min ~ floor = 1e-12)
    ax2 = ax.twinx()
    ax2.fill_between(x, np.maximum(u_min, 1e-12), u_max,
                     color=U_C, alpha=0.18, lw=0)
    ax2.plot(x, u_max, color=U_C, lw=1.3, label=r"$u_\max$")
    ax2.plot(x, np.maximum(u_min, 1e-12),
             color=U_C, lw=1.0, ls="--", label=r"$u_\min$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-12, 1.5)
    ax2.set_ylabel(r"$u$", color=U_C, fontsize=label_fs,
                   rotation=270, labelpad=10)
    ax2.tick_params(axis="y", colors=U_C, labelsize=tick_fs)

    # ξ_peak guide (matches panel a) and optional ξ_LCST
    ax.axvline(xi_peak, color="k", lw=0.7, ls="--", zorder=5)
    if xi_LCST is not None:
        ax.axvline(xi_LCST, color="0.35", lw=0.7, ls=":", zorder=5)

    # Combined legend at lower-right (clear of the rising envelopes which
    # peak in the upper-right; θ_min has its lowest values near ξ=1 too,
    # so the lower-left of the panel is the genuinely empty region).
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=legend_fs, loc="upper left",
              framealpha=0.9, handlelength=1.3, borderpad=0.3,
              labelspacing=0.25, ncol=2, columnspacing=0.6,
              handletextpad=0.4)
    return ax, ax2


def main():
    d = panel_data()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.subplots_adjust(left=0.16, right=0.84, top=0.95, bottom=0.18)
    panel_b(ax, d["x"], d["th_min"], d["th_max"],
            d["u_min"], d["u_max"], d["xi_peak"], d["xi_LCST"])
    add_panel_label(ax, "b")
    save_panel(fig, "fig3b_envelopes_thu")
    plt.close(fig)


if __name__ == "__main__":
    main()
