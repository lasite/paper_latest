"""Quick preview of the fig4 phase grid — coloured regime map only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from fig4_data import (build_grid, REGIME_NAMES, REG_FAILED, REG_STEADY_COLD,
                       REG_BULK_HOPF, REG_LCST_FRONT, REG_GLOBAL_COLLAPSE,
                       REG_STEADY_COLLAPSED, REG_STEADY_FRONT)
from fig2_data import WORKING_POINT

REG_COLORS = {
    REG_FAILED:           "#888888",
    REG_STEADY_COLD:      "#cfe7ff",
    REG_BULK_HOPF:        "#7fbf7b",
    REG_LCST_FRONT:       "#d6604d",
    REG_GLOBAL_COLLAPSE:  "#762a83",
    REG_STEADY_COLLAPSED: "#3a1f73",
    REG_STEADY_FRONT:     "#fed98e",
}


def main():
    g = build_grid()  # uses cache
    x = g["x"]; y = g["y"]; reg = g["regime"]

    codes = sorted(REG_COLORS.keys())
    cmap = ListedColormap([REG_COLORS[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.subplots_adjust(left=0.13, right=0.78, top=0.95, bottom=0.13)
    im = ax.pcolormesh(x, y, reg, cmap=cmap, norm=norm, shading="nearest")

    ax.scatter([WORKING_POINT["Bi_T"]], [WORKING_POINT["S_chi"]],
               s=80, marker="*", color="white", edgecolor="k",
               linewidth=1.0, zorder=5, label="WP")
    ax.set_xlabel(r"$\mathrm{Bi}_T$")
    ax.set_ylabel(r"$S_\chi$")
    ax.set_title("fig 4 phase grid (preview)")
    ax.legend(loc="upper right", fontsize=8)

    # Legend with regime swatches (only for regimes actually present)
    present = sorted(set(reg.flatten().tolist()))
    handles = []
    for c in present:
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     facecolor=REG_COLORS[c],
                                     edgecolor="0.4",
                                     label=REGIME_NAMES[int(c)]))
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(0.99, 0.5), fontsize=8,
               framealpha=0.9, handlelength=1.2)

    out = os.path.join(os.path.dirname(__file__), "fig4_grid_preview.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
