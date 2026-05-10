#!/usr/bin/env python3
"""
make_basin_C.py — D1 renderer.

Reads scan_basin_C_dense.py's cache (NJ x NT IC sweep at the C-cell
classified into cycle / runaway terminal attractor) and renders a
basin map.

Output: figures_pub/basin_C.{pdf,png}
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from style_pub import set_style, PRE_DOUBLE, save
set_style()

DATA_DIR = _HERE.parent / "data" / "fig6"

C_CYCLE   = "#c0392b"
C_RUNAWAY = "#7c3aa8"


def main():
    z = np.load(DATA_DIR / "basin_C_dense.npz")
    J0_vals     = z["J0_vals"]
    theta0_vals = z["theta0_vals"]
    Jm_term     = z["Jm_term"]
    BiT  = float(z["cell_Bi_T"])
    Sx   = float(z["cell_S_chi"])

    # Tri-class on Jm_term:
    #   class 0 = frozen / partial collapse   (Jm_term < 1.0)
    #   class 1 = LCST-front cycle            (1.0 <= Jm_term < 3.0)
    #   class 2 = hot-runaway                 (Jm_term >= 3.0)
    cls = np.full_like(Jm_term, -1, dtype=np.int8)
    cls[(Jm_term < 1.0) & np.isfinite(Jm_term)] = 0
    cls[(Jm_term >= 1.0) & (Jm_term < 3.0)] = 1
    cls[Jm_term >= 3.0] = 2
    print("  class counts: frozen={}, cycle={}, runaway={}, fail={}"
          .format(int((cls==0).sum()), int((cls==1).sum()),
                  int((cls==2).sum()), int((cls==-1).sum())),
          flush=True)

    C_FROZEN = "#d49a3f"  # frozen-front color from existing Fig.6

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(0.55 * PRE_DOUBLE, 3.3))

    cmap = ListedColormap(["#dddddd", C_FROZEN, C_CYCLE, C_RUNAWAY])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    pcm = ax.pcolormesh(J0_vals, theta0_vals, cls,
                        cmap=cmap, norm=norm, shading="auto",
                        zorder=1)

    # Markers for the canonical seeds in the D2 sweep
    ax.plot(1.30, 0.0, "*", color="white", mec="k", mew=0.8,
            ms=12, zorder=10)
    ax.plot(0.20, 0.0, "^", color="white", mec="k", mew=0.8,
            ms=8, zorder=10)
    ax.text(1.30, 0.4, "cold seed\n(D2)", fontsize=6, color="k",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                      alpha=0.85))
    ax.text(0.20, 0.4, "hot seed\n(D2)", fontsize=6, color="k",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                      alpha=0.85))

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_FROZEN, ec="0.4",
                      label="frozen / partial collapse"),
        plt.Rectangle((0, 0), 1, 1, fc=C_CYCLE, ec="0.4",
                      label="LCST-front cycle basin"),
        plt.Rectangle((0, 0), 1, 1, fc=C_RUNAWAY, ec="0.4",
                      label="hot-runaway basin"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              framealpha=0.95, handlelength=1.4, borderpad=0.4)

    ax.set_xlabel(r"$J_0$ (initial swelling)")
    ax.set_ylabel(r"$\theta_0$ (initial thermal pulse)")
    ax.set_xlim(J0_vals.min(), J0_vals.max())
    ax.set_ylim(theta0_vals.min(), theta0_vals.max())
    ax.tick_params(direction="out", length=2.5, labelsize=7)
    ax.set_title(rf"C-cell basins of attraction "
                 rf"($\mathrm{{Bi}}_T={BiT:.3f}$, $S_\chi={Sx:.2f}$)",
                 fontsize=8)

    save(fig, "basin_C")


if __name__ == "__main__":
    main()
