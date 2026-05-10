#!/usr/bin/env python3
"""
make_fig_lsa_vs_nl.py — Single-panel classification of where the
0D linear stability prediction (re_max_complex > 0 at the swollen
homogeneous SS) agrees or disagrees with the nonlinear PDE outcome on
the (Bi_T, S_chi) grid that fig4 uses.

Two physical disagreement modes:
  - B-cells (LSA Hopf-unstable but NL steady): subcritical Hopf escape.
  - C-cells (LSA stable but NL oscillates): the swollen homogeneous SS
    does not exist; the PDE finds a spatially driven LCST-front cycle
    that has no 0D analog.

The summary text panels and the (Bi_T, Da) duplicate are intentionally
omitted; aggregate counts are printed to stdout for inline citation in
the main text.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from style_pub import set_style

PRE_SINGLE = 3.375  # APS single-column width (inches)
from fig2_data import WORKING_POINT
from fig4_data import (build_grid, BI_T_VALS, S_CHI_VALS,
                       REG_FAILED, REG_BULK_HOPF, REG_LCST_FRONT,
                       REG_GLOBAL_COLLAPSE, REG_STEADY_COLD,
                       REG_STEADY_COLLAPSED, REG_STEADY_FRONT)
from hopf_boundary import hopf_grid

set_style()

LBL_FS  = 8
TICK_FS = 6.5
LEG_FS  = 6.0

# ── classification codes ─────────────────────────────────────────────
A_CODE, B_CODE, C_CODE, D_CODE, F_CODE = 0, 1, 2, 3, -1

CELL_COLORS = {
    A_CODE: "#cfe7ff",
    B_CODE: "#fdb863",
    C_CODE: "#b2182b",
    D_CODE: "#7fbf7b",
    F_CODE: "#888888",
}
CELL_LABEL = {
    A_CODE: "A: LSA stable, NL steady",
    B_CODE: "B: LSA Hopf, NL steady (subcritical)",
    C_CODE: "C: LSA stable, NL osc (no 0D analog)",
    D_CODE: "D: LSA Hopf, NL oscillating",
}

OSC_REGIMES    = {REG_BULK_HOPF, REG_LCST_FRONT, REG_GLOBAL_COLLAPSE}
STEADY_REGIMES = {REG_STEADY_COLD, REG_STEADY_COLLAPSED, REG_STEADY_FRONT}


def categorize(g, h):
    reg = g["regime"]
    re_c = h["re_max_complex"]
    lsa_hopf = np.isfinite(re_c) & (re_c > 0.0)
    nl_osc = np.isin(reg, list(OSC_REGIMES))
    nl_steady = np.isin(reg, list(STEADY_REGIMES))
    nl_failed = (reg == REG_FAILED)
    out = np.full(reg.shape, F_CODE, dtype=int)
    out[(~lsa_hopf) & nl_steady] = A_CODE
    out[lsa_hopf & nl_steady] = B_CODE
    out[(~lsa_hopf) & nl_osc] = C_CODE
    out[lsa_hopf & nl_osc] = D_CODE
    out[nl_failed] = F_CODE
    return out


def main():
    g_main = build_grid(param_x="Bi_T", x_vals=BI_T_VALS,
                        param_y="S_chi", y_vals=S_CHI_VALS)
    h_main = hopf_grid("Bi_T", BI_T_VALS, "S_chi", S_CHI_VALS)
    cat = categorize(g_main, h_main)

    # Diagnostic counts (printed to stdout for inline citation in §IV.D)
    n = cat.size
    counts = {c: int(np.sum(cat == c)) for c in
              (A_CODE, B_CODE, C_CODE, D_CODE, F_CODE)}
    n_eff = n - counts[F_CODE]
    agree = counts[A_CODE] + counts[D_CODE]
    disagree = counts[B_CODE] + counts[C_CODE]
    # Fraction of cells where LSA's swollen homogeneous SS does not exist
    re_c = h_main["re_max_complex"]
    no_swollen_ss = int(np.sum(~np.isfinite(re_c)))
    pct_no_ss = 100 * no_swollen_ss / n

    print("="*60)
    print(f"  LSA-vs-NL classification on (Bi_T, S_chi) grid:")
    print(f"    n_cells              = {n}")
    print(f"    n_failed (NL)        = {counts[F_CODE]}")
    print(f"    A (agreed quiet)     = {counts[A_CODE]:>4}  "
          f"({100*counts[A_CODE]/n_eff:.1f}%)")
    print(f"    D (agreed osc)       = {counts[D_CODE]:>4}  "
          f"({100*counts[D_CODE]/n_eff:.1f}%)")
    print(f"    B (subcritical Hopf) = {counts[B_CODE]:>4}  "
          f"({100*counts[B_CODE]/n_eff:.1f}%)")
    print(f"    C (no 0D analog)     = {counts[C_CODE]:>4}  "
          f"({100*counts[C_CODE]/n_eff:.1f}%)")
    print(f"    agreement total      = {agree:>4}  "
          f"({100*agree/n_eff:.1f}%)")
    print(f"    disagreement total   = {disagree:>4}  "
          f"({100*disagree/n_eff:.1f}%)")
    print(f"    cells without swollen homog SS = {no_swollen_ss}/{n} "
          f"({pct_no_ss:.1f}%)")
    print("="*60)

    # ── Single-panel figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(PRE_SINGLE, PRE_SINGLE * 0.95))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.30)

    codes = sorted(CELL_COLORS.keys())
    cmap = ListedColormap([CELL_COLORS[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    ax.pcolormesh(g_main["x"], g_main["y"], cat,
                  cmap=cmap, norm=norm,
                  shading="nearest", rasterized=True)
    ax.set_xscale("log")

    # Working point marker
    ax.scatter([WORKING_POINT["Bi_T"]], [WORKING_POINT["S_chi"]],
               s=60, marker="*", color="white", edgecolor="k",
               linewidth=0.8, zorder=8)
    ax.annotate("WP", (WORKING_POINT["Bi_T"], WORKING_POINT["S_chi"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6, color="0.10")

    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)

    # Inline horizontal legend below the panel (4 categories)
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=CELL_COLORS[c],
                             edgecolor="0.3",
                             label=CELL_LABEL[c])
               for c in (A_CODE, D_CODE, B_CODE, C_CODE)]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.005),
                     ncol=2, fontsize=LEG_FS,
                     framealpha=0.95, handlelength=1.2,
                     borderpad=0.4, columnspacing=1.0,
                     handletextpad=0.5)

    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            "..", "Figure", "pub"))
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "fig5.pdf")
    png = os.path.join(out_dir, "fig5.png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


if __name__ == "__main__":
    main()
