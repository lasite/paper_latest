#!/usr/bin/env python3
"""
make_fig2a_schematic.py — Fig 2(a): slab + bath schematic with three Biot fluxes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style_pub import set_style, add_panel_label
from fig2_data import save_panel
set_style()


def draw_schematic(ax, fontsize_scale=1.0):
    """Draw the gel + bath schematic.

    fontsize_scale shrinks all text sizes for tiled use (composite figure).
    """
    fs = fontsize_scale
    ax.set_xlim(-0.3, 2.95)
    ax.set_ylim(-0.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    gel = FancyBboxPatch((0, 0), 1.6, 1.8, boxstyle="round,pad=0.04",
                         facecolor="#c8e6fa", edgecolor="#1f77b4", lw=1.2)
    ax.add_patch(gel)
    ax.text(0.8, 1.30, "gel", ha="center", va="center", fontsize=12*fs,
            color="#1f77b4", weight="bold")
    ax.text(0.8, 0.30, "catalyst", ha="center", va="center",
            fontsize=9*fs, color="#1f77b4", style="italic")

    bath = FancyBboxPatch((1.82, 0), 0.95, 1.8, boxstyle="round,pad=0.04",
                          facecolor="#eaf5e9", edgecolor="#2ca02c",
                          lw=1.0, ls="--")
    ax.add_patch(bath)
    ax.text(2.30, 1.60, "bath", ha="center", fontsize=10*fs, color="#2ca02c")

    ax.plot([0, 0], [0, 1.8], color="#555", lw=1.0, ls="--")
    ax.text(0,   -0.22, r"$\xi=0$",   ha="center", va="top", fontsize=9*fs)
    ax.text(0,   -0.50, "symmetry",   ha="center", va="top", fontsize=8*fs,
            color="#555")
    ax.text(1.6, -0.22, r"$\xi=1$",   ha="center", va="top", fontsize=9*fs)
    ax.text(1.6, -0.50, "surface",    ha="center", va="top", fontsize=8*fs,
            color="#555")

    # Bi_mu: bidirectional (solvent flows in or out depending on μ - μ_b)
    ax.annotate("", xy=(2.05, 1.25), xytext=(1.50, 1.25),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.6))
    ax.text(2.35, 1.25, r"$\mathrm{Bi}_\mu$", ha="left", va="center",
            fontsize=10*fs, color="#1f77b4")

    # Bi_c: reactant flows from bath → gel
    ax.annotate("", xy=(1.50, 0.80), xytext=(2.05, 0.80),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6))
    ax.text(2.35, 0.80, r"$\mathrm{Bi}_c$", ha="left", va="center",
            fontsize=10*fs, color="#2ca02c")

    # Bi_T: heat loss from gel → bath
    ax.annotate("", xy=(2.05, 0.35), xytext=(1.50, 0.35),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
    ax.text(2.35, 0.35, r"$\mathrm{Bi}_T$", ha="left", va="center",
            fontsize=10*fs, color="#d62728")


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    draw_schematic(ax)
    add_panel_label(ax, 'a')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    save_panel(fig, "fig2a_schematic")
    plt.close(fig)


if __name__ == "__main__":
    main()
