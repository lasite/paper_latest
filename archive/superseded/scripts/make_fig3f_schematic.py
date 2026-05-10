#!/usr/bin/env python3
"""
make_fig3f_schematic.py — Fig 3(f): two-zone mechanism schematic.

Stylised cross-section of the slab from ξ=0 (centre) to ξ=1 (free surface).
The split at ξ_c separates a passive, reactant-starved core where θ
oscillates only because heat conducts inward, from an active outer shell
where the local rate is gated by accessibility (1−φ)^{m_act} swinging
between fully open and fully closed each cycle. Annotations label the
dominant feedback in each zone and the fluxes that couple them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

from style_pub import set_style, add_panel_label
from fig3_data import panel_data, save_panel
set_style()


CORE_FACE  = "#cfe3f6"
SHELL_FACE = "#fde0d3"
CORE_EDGE  = "#1f5fa3"
SHELL_EDGE = "#a23e1c"


def draw_schematic(ax, xi_c=0.81, fontsize_scale=1.0):
    """Draw the two-zone mechanism schematic on `ax`.

    Compact layout that scales down to the 2-inch composite cell:
      0.00–0.10 : ξ axis labels
      0.10–0.32 : slab band
      0.36–0.58 : inter-zone flux arrows
      0.62–0.97 : two title + bullet-list zone descriptions
    """
    fs = fontsize_scale
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    ax.set_axis_off()

    # ── Slab cross-section (taller for a stronger visual) ─────────
    slab_y0, slab_y1 = 0.10, 0.32
    ax.add_patch(Rectangle((0.0, slab_y0), xi_c, slab_y1 - slab_y0,
                           facecolor=CORE_FACE, edgecolor=CORE_EDGE,
                           lw=0.8, zorder=2))
    ax.add_patch(Rectangle((xi_c, slab_y0), 1.0 - xi_c, slab_y1 - slab_y0,
                           facecolor=SHELL_FACE, edgecolor=SHELL_EDGE,
                           lw=0.8, zorder=2))
    ax.plot([1.0, 1.0], [slab_y0 - 0.012, slab_y1 + 0.012],
            color="k", lw=1.2, zorder=3)
    ax.text(1.012, 0.5 * (slab_y0 + slab_y1), "bath",
            fontsize=7 * fs, ha="left", va="center", color="0.25")
    ax.plot([xi_c, xi_c], [slab_y0 - 0.005, slab_y1 + 0.005],
            color="k", lw=0.8, ls="--", zorder=4)
    ax.text(0.5 * xi_c, 0.5 * (slab_y0 + slab_y1), "core",
            fontsize=8.5 * fs, ha="center", va="center",
            color=CORE_EDGE, weight="bold")
    ax.text(0.5 * (xi_c + 1.0), 0.5 * (slab_y0 + slab_y1), "shell",
            fontsize=8.5 * fs, ha="center", va="center",
            color=SHELL_EDGE, weight="bold")

    ax.text(0.0, slab_y0 - 0.04, r"$\xi=0$",
            fontsize=7 * fs, ha="left", va="top", color="0.25")
    # Place ξ_c label slightly below to avoid colliding with the ξ=1 marker.
    ax.text(xi_c, slab_y0 - 0.04, rf"$\xi_c\!\approx\!{xi_c:.2f}$",
            fontsize=7 * fs, ha="right", va="top", color="0.15",
            weight="bold")
    ax.text(1.0, slab_y0 - 0.04, r"$\xi=1$",
            fontsize=7 * fs, ha="right", va="top", color="0.25")

    # ── Inter-zone fluxes (between slab and zone descriptions) ────
    y_react = slab_y1 + 0.08
    ax.add_patch(FancyArrowPatch(
        (1.005, y_react), (xi_c + 0.05, y_react),
        arrowstyle="-|>", mutation_scale=9, color="#137a73", lw=1.0,
        zorder=4))
    ax.text(0.5 * (xi_c + 0.05 + 1.005), y_react + 0.022,
            "reactant in",
            fontsize=6 * fs, ha="center", va="bottom", color="#137a73")

    y_heat = slab_y1 + 0.18
    ax.add_patch(FancyArrowPatch(
        (xi_c + 0.05, y_heat), (xi_c - 0.32, y_heat),
        arrowstyle="-|>", mutation_scale=9, color="#a23e1c", lw=1.0,
        zorder=4))
    ax.text(0.5 * (xi_c - 0.32 + xi_c + 0.05), y_heat + 0.022,
            "heat inward",
            fontsize=6 * fs, ha="center", va="bottom", color="#a23e1c")

    # ── Vertically stacked headers + key mechanism (full-width) ──
    # Core block (top)
    ax.text(0.0, 0.97, "core",
            fontsize=8.5 * fs, ha="left", va="top",
            color=CORE_EDGE, weight="bold")
    ax.text(0.16, 0.97,
            r"$u\!\approx\!0,\;R\!\approx\!0$;  "
            r"$\theta$ slaved to inward heat;",
            fontsize=6.5 * fs, ha="left", va="top", color="0.15")
    ax.text(0.16, 0.89,
            r"small $J$ swing via $\chi(\theta)$ coupling.",
            fontsize=6.5 * fs, ha="left", va="top", color="0.15")

    # Shell block (middle)
    ax.text(0.0, 0.78, "shell",
            fontsize=8.5 * fs, ha="left", va="top",
            color=SHELL_EDGE, weight="bold")
    ax.text(0.16, 0.78,
            r"$\theta\!\uparrow\!\Rightarrow\!\chi\!\uparrow\!\Rightarrow\!J\!\downarrow$;  "
            r"collapse: $\phi\!\to\!1$,",
            fontsize=6.5 * fs, ha="left", va="top", color="0.15")
    ax.text(0.16, 0.70,
            r"$(1\!-\!\phi)^{m}\!\to\!0$ quenches $R$; re-swell,",
            fontsize=6.5 * fs, ha="left", va="top", color="0.15")
    ax.text(0.16, 0.62,
            "refill, restart.",
            fontsize=6.5 * fs, ha="left", va="top", color="0.15")


def main():
    d = panel_data()
    xi_c = d["xi_LCST"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    draw_schematic(ax, xi_c=xi_c)
    add_panel_label(ax, "f")
    save_panel(fig, "fig3f_schematic")
    plt.close(fig)


if __name__ == "__main__":
    main()
