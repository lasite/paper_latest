#!/usr/bin/env python3
"""
make_homogeneous_ss.py — C1 renderer (single panel).

Reads the analytic homogeneous-SS classification cache and the
semi-analytical saddle-node fold curve, then produces a single-panel
figure of the SS branch existence map on (Bi_T, S_chi) with the
analytic fold overlay and the working-point and C-cell markers.

The original two-panel layout (with PDE oscillation overlay) was
retired: the 25x25 PDE regime grid is too coarse to draw smooth
contours, and the LSA-vs-PDE physical comparison is already a direct
corollary of the slow-manifold construction in Sec. IV.B.

Cache-only renderer; cheap to re-run.
Output: figures_pub/homogeneous_ss.{pdf,png}
"""
from __future__ import annotations

import sys
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

DATA_DIR_LSA = _HERE.parent / "data" / "fig5"

# Working point + C-cell markers
WP_BI_T  = 0.10
WP_S_CHI = 1.00
C_BI_T   = 0.059
C_S_CHI  = 1.80


def load_lsa():
    """Load the analytic SS classification (cls in {0,1,2,3}).

    0 = no homogeneous SS in the considered (J, theta) box
    1 = swollen only
    2 = collapsed only
    3 = bistable (>= 2 roots; the original '>= 3 roots' sliver
        is folded into this category)
    """
    z = np.load(DATA_DIR_LSA / "homogeneous_ss_analytic.npz",
                allow_pickle=True)
    return {
        "Bi_T":  z["Bi_T_vals"],
        "S_chi": z["S_chi_vals"],
        "cls":   z["classification"],
        "n_roots": z["n_roots"],
    }


def load_fold():
    """Load semi-analytical SN fold curve. Returns list of (M, 4)
    arrays with columns [Bi_T, S_chi, J, theta]; empty list if cache
    absent."""
    cache = DATA_DIR_LSA / "fold_curve_BiT_Schi.npz"
    if not cache.exists():
        return []
    z = np.load(cache)
    flat = z["flat"]; lens = z["lengths"]
    segs, k = [], 0
    for L in lens:
        segs.append(flat[k:k + int(L)])
        k += int(L)
    return segs


def panel_branches(ax, lsa, fold_segs=None):
    """SS branch existence map (4 categories) + analytic fold overlay."""
    Bi_T, S_chi = lsa["Bi_T"], lsa["S_chi"]
    cls = lsa["cls"]                # rows = Bi_T, cols = S_chi

    # Categorical colors:
    #   0 = no roots          (light grey)
    #   1 = swollen only      (sky blue)
    #   2 = collapsed only    (terra cotta)
    #   3 = bistable          (purple)
    colors = ["#dadada", "#a6cee3", "#e6705a", "#9b6cb8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    ax.pcolormesh(Bi_T, S_chi, cls.T, cmap=cmap, norm=norm,
                  shading="auto")

    # Semi-analytical fold curve overlay (saddle-node locus from
    # det J(F1, F2) = 0 mapped through the explicit (Bi_T, S_chi)
    # parameterization).  This is the same boundary as the cls
    # transitions, but drawn from the smooth 800x800 (J, theta)
    # field — no pixel staircase.
    if fold_segs:
        for k, seg in enumerate(fold_segs):
            label = "fold (analytic)" if k == 0 else None
            ax.plot(seg[:, 0], seg[:, 1], "-", color="black",
                    lw=1.0, alpha=0.95, zorder=8, label=label)

    # Working point + C-cell markers
    ax.plot(WP_BI_T, WP_S_CHI, "*", color="white", mec="k",
            mew=0.8, ms=11, zorder=10)
    ax.plot(C_BI_T, C_S_CHI, "^", color="white", mec="k",
            mew=0.8, ms=7, zorder=10)
    ax.text(WP_BI_T * 1.18, WP_S_CHI, "WP", fontsize=6, va="center",
            ha="left", color="k", zorder=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                      alpha=0.85))
    ax.text(C_BI_T * 1.20, C_S_CHI, "C-cell", fontsize=6, va="center",
            ha="left", color="k", zorder=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                      alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$")
    ax.set_ylabel(r"$S_\chi$")
    ax.set_xlim(Bi_T.min(), Bi_T.max())
    ax.set_ylim(S_chi.min(), S_chi.max())
    ax.tick_params(direction="out", length=2.5)

    # Discrete legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[1], ec="0.4",
                      label="swollen only"),
        plt.Rectangle((0, 0), 1, 1, fc=colors[2], ec="0.4",
                      label="collapsed only"),
        plt.Rectangle((0, 0), 1, 1, fc=colors[3], ec="0.4",
                      label="bistable"),
        plt.Rectangle((0, 0), 1, 1, fc=colors[0], ec="0.4",
                      label="no root"),
    ]
    if fold_segs:
        handles.append(plt.Line2D([], [], color="black", lw=1.0,
                                  label="fold (analytic)"))
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              framealpha=0.95, handlelength=1.2, ncol=1, borderpad=0.4)


def main():
    lsa = load_lsa()
    fold_segs = load_fold()
    if fold_segs:
        n_pts = sum(s.shape[0] for s in fold_segs)
        print(f"  fold-curve overlay: {len(fold_segs)} segs, {n_pts} pts")

    # Single-column APS width (PRE_DOUBLE / 2 ≈ 3.44") with a tight
    # square aspect so the (Bi_T, S_chi) plane reads clearly at the
    # printed size.
    fig = plt.figure(figsize=(PRE_DOUBLE / 2.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    panel_branches(ax, lsa, fold_segs=fold_segs)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.91, bottom=0.18)
    ax.set_title("homogeneous SS branches", fontsize=8)

    save(fig, "homogeneous_ss")


if __name__ == "__main__":
    main()
