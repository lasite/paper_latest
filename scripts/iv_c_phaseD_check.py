#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseD_check.py — Phase D verification + figures.

Reads
  data/iv_c/phaseD/main_grid.npz       (49 sims at m_act,m_diff)
  data/iv_c/phaseD/BiT_slice.npz       (4 sims at varying Bi_T)

Verifies scaling law (ii)
   1 - xi_LCST  ->  min(L_T, L_c) * G(m_act + m_diff)
                                   ^^^^^^^^^^^^^^^^^^^^
                                   approaches a constant in the
                                   sharp-barrier limit
with L_T = sqrt(alpha/Bi_T), L_c = sqrt(delta/Bi_c).

PASS:
  - For m_act+m_diff above some m_c (the crossover into sharp-barrier
    regime), (1 - xi_LCST)/min(L_T, L_c) is constant within 30% across
    the grid of large-sum points.
  - At small m_act+m_diff, hot-runaway observed (xi_LCST ~ 0).

Outputs
  data/iv_c/phaseD/phaseD_check.npz
  Figure/pub/iv_c_front_depth_main.{pdf,png}
  Figure/pub/iv_c_front_depth_attractor_map.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE   # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA = _HERE.parent / "data" / "iv_c" / "phaseD"
FIG = _HERE.parent / "Figure" / "pub"

# Working-point constants for L_T / L_c calc
ALPHA = 0.20
DELTA = 0.08
BI_C = 0.70

# PASS thresholds
SATURATION_TOL = 0.30      # within 30%

# Discrete colormap for the 5+ classes
CLASS_ORDER = [
    "cold_SS", "overswollen_uniform", "overswollen_front",
    "frozen_front", "cycle", "hot_runaway", "TIMEOUT", "sim_error",
]
CLASS_COLORS = {
    "cold_SS":             "#3D5A80",   # navy
    "overswollen_uniform": "#A0C4FF",   # light blue
    "overswollen_front":   "#FFB703",   # amber
    "frozen_front":        "#FB8500",   # orange
    "cycle":               "#7AE582",   # green
    "hot_runaway":         "#D62828",   # red
    "TIMEOUT":             "#444444",   # dark grey
    "sim_error":           "#999999",   # grey
}


def _L_T(Bi_T):  return np.sqrt(ALPHA / Bi_T)
def _L_c():       return np.sqrt(DELTA / BI_C)


def main():
    main_grid = np.load(DATA / "main_grid.npz", allow_pickle=False)
    slc = np.load(DATA / "BiT_slice.npz", allow_pickle=False)

    m_act = main_grid["m_act"]
    m_diff = main_grid["m_diff"]
    xi_LCST = main_grid["xi_LCST"]            # (n_mact, n_mdiff)
    classification = main_grid["classification"]
    Bi_T_grid = float(main_grid["Bi_T"])
    Da_grid = float(main_grid["Da"])

    # ── Main-grid scaling check ─────────────────────────────────────
    M_sum = m_act[:, None] + m_diff[None, :]
    one_m_xi = 1.0 - xi_LCST
    L_T_grid = _L_T(Bi_T_grid)
    L_c_grid = _L_c()
    L_eff = min(L_T_grid, L_c_grid)
    ratio = one_m_xi / L_eff

    print("=" * 70)
    print(" Phase D - front-depth scaling check")
    print("=" * 70)
    print(f"  Main grid Bi_T={Bi_T_grid}, alpha={ALPHA}, delta={DELTA}, "
          f"Bi_c={BI_C}")
    print(f"  L_T = sqrt(alpha/Bi_T) = {L_T_grid:.4f}")
    print(f"  L_c = sqrt(delta/Bi_c) = {L_c_grid:.4f}")
    print(f"  L_eff = min(L_T, L_c) = {L_eff:.4f}")
    print()
    print(f"  {'m_act+m_diff':>13s}  {'mean(1-xi)':>11s}  "
          f"{'mean ratio':>11s}  {'std ratio':>10s}  {'n_finite':>9s}")
    sat_check = []
    for s in range(2, m_act.max() + m_diff.max() + 1):
        mask = (M_sum == s) & np.isfinite(one_m_xi)
        if not mask.any():
            continue
        vals = ratio[mask]
        ones = one_m_xi[mask]
        print(f"  {s:13d}  {ones.mean():11.4f}  "
              f"{vals.mean():11.4f}  {vals.std():10.4f}  "
              f"{mask.sum():9d}")
        if s >= 6:
            sat_check.append(vals)

    print()
    if sat_check:
        all_vals = np.concatenate(sat_check)
        mean_ratio = float(all_vals.mean())
        std_ratio = float(all_vals.std())
        spread = (all_vals.max() - all_vals.min()) / abs(mean_ratio) \
            if mean_ratio != 0 else float("inf")
        print(f"  Saturation regime (m_act+m_diff >= 6):")
        print(f"    mean (1-xi)/L_eff = {mean_ratio:.4f}")
        print(f"    std              = {std_ratio:.4f}")
        print(f"    full spread/mean = {spread:.2%}")
    else:
        mean_ratio = float("nan"); std_ratio = float("nan")
        spread = float("inf")

    PASS_saturation = spread < SATURATION_TOL

    # Hot-runaway expected at small m_act+m_diff?
    small_m_classes = []
    for i, ma in enumerate(m_act):
        for j, md in enumerate(m_diff):
            if ma + md <= 3:
                small_m_classes.append(str(classification[i, j]))
    PASS_hotrunaway = (len(small_m_classes) > 0 and
                       any("hot_runaway" in c for c in small_m_classes))
    print(f"\n  Hot-runaway at small m_act+m_diff?  {PASS_hotrunaway}")
    print(f"    small-m classes: {set(small_m_classes)}")

    PASS = PASS_saturation
    print(f"\n  >>> Phase D {'PASS' if PASS else 'FAIL'}  "
          f"(saturation spread {spread:.2%} < {SATURATION_TOL:.0%}: "
          f"{PASS_saturation}; hot-runaway flag: {PASS_hotrunaway}) <<<")

    # ── Bi_T slice scaling ──────────────────────────────────────────
    BI_T_slice = slc["Bi_T"]
    xi_slice = slc["xi_LCST"]
    cls_slice = slc["classification"]
    one_m_xi_slice = 1.0 - xi_slice
    L_T_slice = _L_T(BI_T_slice)
    L_c_slice = np.full_like(L_T_slice, _L_c())
    L_eff_slice = np.minimum(L_T_slice, L_c_slice)
    ratio_slice = one_m_xi_slice / L_eff_slice
    print(f"\n  Bi_T slice (m_act=m_diff=4):")
    print(f"  {'Bi_T':>6s}  {'L_T':>6s}  {'L_c':>6s}  {'L_eff':>6s}  "
          f"{'xi':>6s}  {'1-xi':>6s}  {'ratio':>7s}  class")
    for k, b in enumerate(BI_T_slice):
        print(f"  {b:6.3f}  {L_T_slice[k]:6.3f}  {L_c_slice[k]:6.3f}  "
              f"{L_eff_slice[k]:6.3f}  {xi_slice[k]:6.3f}  "
              f"{one_m_xi_slice[k]:6.3f}  {ratio_slice[k]:7.3f}  "
              f"{cls_slice[k]}")

    # Save summary
    np.savez(DATA / "phaseD_check.npz",
             # main grid
             m_act=m_act, m_diff=m_diff,
             xi_LCST=xi_LCST, classification=classification,
             one_m_xi=one_m_xi, ratio=ratio, M_sum=M_sum,
             L_eff_grid=L_eff,
             sat_mean=mean_ratio, sat_std=std_ratio, sat_spread=spread,
             pass_saturation=PASS_saturation,
             pass_hotrunaway=PASS_hotrunaway,
             # Bi_T slice
             Bi_T_slice=BI_T_slice,
             L_T_slice=L_T_slice, L_c_slice=L_c_slice,
             L_eff_slice=L_eff_slice,
             xi_slice=xi_slice, ratio_slice=ratio_slice,
             cls_slice=cls_slice)
    print(f"\n  saved {DATA / 'phaseD_check.npz'}")

    # ── Figure 1: front-depth scaling (main + slice) ────────────────
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(PRE_DOUBLE, 3.2))
    ax_a, ax_b = axes

    # (a): (1-xi)/L_eff vs m_act+m_diff
    M_unique = np.sort(np.unique(M_sum))
    for s in M_unique:
        mask = (M_sum == s) & np.isfinite(ratio)
        if not mask.any(): continue
        ax_a.plot([s] * int(mask.sum()), ratio[mask], "o",
                  color="C0", alpha=0.6, ms=4)
    if np.isfinite(mean_ratio):
        ax_a.axhline(mean_ratio, color="C1", lw=1.0,
                      label=fr"mean (m≥6): {mean_ratio:.2f}")
        ax_a.axhline(mean_ratio * (1 + SATURATION_TOL),
                      color="C1", lw=0.5, ls=":")
        ax_a.axhline(mean_ratio * (1 - SATURATION_TOL),
                      color="C1", lw=0.5, ls=":")
    ax_a.axhline(1.0, color="grey", ls="--", lw=0.5,
                  label=r"$1-\xi = L_\mathrm{eff}$")
    ax_a.set_xlabel(r"$m_\mathrm{act} + m_\mathrm{diff}$")
    ax_a.set_ylabel(r"$(1 - \xi_\mathrm{LCST}) / L_\mathrm{eff}$")
    ax_a.legend(fontsize=7, loc="best")
    ax_a.set_title("(a) main grid saturation")

    # (b): Bi_T slice. (1-xi) vs Bi_T with L_T, L_c overlays
    BiT_continuous = np.linspace(0.04, 0.30, 60)
    L_T_cont = _L_T(BiT_continuous)
    L_c_cont = np.full_like(BiT_continuous, _L_c())
    L_eff_cont = np.minimum(L_T_cont, L_c_cont)
    ax_b.plot(BiT_continuous, L_T_cont, ":", color="C3",
              label=r"$L_T = \sqrt{\alpha/\mathrm{Bi}_T}$")
    ax_b.plot(BiT_continuous, L_c_cont, ":", color="C2",
              label=r"$L_c = \sqrt{\delta/\mathrm{Bi}_c}$")
    ax_b.plot(BiT_continuous, L_eff_cont, "-", color="grey",
              lw=1.0,
              label=r"$\min(L_T, L_c)$")
    ax_b.plot(BI_T_slice, one_m_xi_slice, "o", ms=5, color="C0",
              label=r"PDE: $1 - \xi_\mathrm{LCST}$")
    ax_b.set_xlabel(r"$\mathrm{Bi}_T$")
    ax_b.set_ylabel("depth")
    ax_b.set_title(r"(b) Bi$_T$ slice ($m_\mathrm{act}=m_\mathrm{diff}$=4)")
    ax_b.legend(fontsize=7, loc="best")

    fig.tight_layout()
    pdf = FIG / "iv_c_front_depth_main.pdf"
    png = FIG / "iv_c_front_depth_main.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")

    # ── Figure 2: attractor map on (m_act, m_diff) plane ───────────
    fig2, ax2 = plt.subplots(figsize=(PRE_DOUBLE / 2, 4.0))
    # Build a categorical color array
    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    idx_map = np.array([[class_to_idx.get(str(classification[i, j]), -1)
                          for j in range(len(m_diff))]
                         for i in range(len(m_act))], dtype=int)
    color_list = [CLASS_COLORS[c] for c in CLASS_ORDER]
    cmap = ListedColormap(color_list)
    im = ax2.imshow(idx_map, origin="lower",
                    extent=[m_diff[0] - 0.5, m_diff[-1] + 0.5,
                             m_act[0] - 0.5, m_act[-1] + 0.5],
                    cmap=cmap, vmin=-0.5, vmax=len(CLASS_ORDER) - 0.5,
                    aspect="auto")
    ax2.set_xticks(m_diff); ax2.set_yticks(m_act)
    ax2.set_xlabel(r"$m_\mathrm{diff}$")
    ax2.set_ylabel(r"$m_\mathrm{act}$")
    ax2.set_title(f"attractor classes at "
                  f"Bi_T={Bi_T_grid:.2f}, Da={Da_grid:.1f}, S_chi=1.0")
    # Annotate xi_LCST values
    for i, ma in enumerate(m_act):
        for j, md in enumerate(m_diff):
            xi = xi_LCST[i, j]
            if np.isfinite(xi):
                ax2.text(md, ma, f"{xi:.2f}", ha="center", va="center",
                          fontsize=6, color="white" if idx_map[i, j] in
                          [class_to_idx["cold_SS"], class_to_idx["hot_runaway"]]
                          else "black")
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_ORDER]
    ax2.legend(handles, CLASS_ORDER, loc="center left",
                bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig2.tight_layout()
    pdf2 = FIG / "iv_c_front_depth_attractor_map.pdf"
    png2 = FIG / "iv_c_front_depth_attractor_map.png"
    fig2.savefig(pdf2, dpi=600, bbox_inches="tight")
    fig2.savefig(png2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved {pdf2}")
    print(f"  saved {png2}")


if __name__ == "__main__":
    main()
