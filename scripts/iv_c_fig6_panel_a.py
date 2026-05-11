#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fig6_panel_a.py — Front depth saturation (Phase D).

Shows (1 - xi_LCST) / L_eff vs m_act + m_diff for all cycle cells of
the (m_act, m_diff) main grid. Highlights the saturation crossover
at M_sum ≈ 6, where the ratio plateaus at c ≈ 0.388 ± 0.002.

Source data: data/iv_c/phaseD/phaseD_check.npz
Cached data: data/iv_c/phaseE/fig6_panel_a.npz
Output:      Figure/pub/iv_c_fig6_panel_a.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE  # type: ignore
PANEL_W = PRE_DOUBLE / 2.0
PANEL_H = PANEL_W * 0.78

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = _HERE.parent / "data" / "iv_c" / "phaseE"
CACHE.mkdir(parents=True, exist_ok=True)
FIG_DIR = _HERE.parent / "Figure" / "pub"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_cache():
    src = np.load(_HERE.parent / "data" / "iv_c" / "phaseD" /
                  "phaseD_check.npz", allow_pickle=True)
    m_act = np.array(src["m_act"])
    m_diff = np.array(src["m_diff"])
    classification = np.array(src["classification"])
    one_m_xi = np.array(src["one_m_xi"])
    ratio = np.array(src["ratio"])
    M_sum = np.array(src["M_sum"])
    L_eff_grid = float(src["L_eff_grid"])
    sat_mean = float(src["sat_mean"])
    sat_std = float(src["sat_std"])
    sat_spread = float(src["sat_spread"])

    npz = CACHE / "fig6_panel_a.npz"
    np.savez(npz,
             m_act=m_act, m_diff=m_diff,
             classification=classification,
             one_m_xi=one_m_xi, ratio=ratio, M_sum=M_sum,
             L_eff=L_eff_grid,
             sat_mean=sat_mean, sat_std=sat_std, sat_spread=sat_spread)
    return npz


def plot(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    classification = d["classification"]
    ratio = d["ratio"]
    M_sum = d["M_sum"]
    sat_mean = float(d["sat_mean"])
    sat_spread = float(d["sat_spread"])

    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    # Cycle cells
    cycle_mask = (classification == "cycle")
    cycle_M = M_sum[cycle_mask]
    cycle_r = ratio[cycle_mask]
    # Non-cycle / non-saturated cells
    nonsat_mask = (classification != "cycle") & ~np.isin(classification,
                                                        ["TIMEOUT", "sim_error"])
    nonsat_M = M_sum[nonsat_mask]
    nonsat_r = ratio[nonsat_mask]

    ax.scatter(cycle_M, cycle_r, marker="o", s=30, color="C0",
               edgecolors="white", linewidths=0.5, label="cycle")
    if len(nonsat_M) > 0:
        ax.scatter(nonsat_M, nonsat_r, marker="x", s=24, color="C3",
                   label="non-cycle")

    # Saturation line
    ax.axhline(sat_mean, color="C1", ls="-", lw=1.2,
               label=fr"mean ($M\!\geq\!6$): {sat_mean:.3f}")
    ax.axhspan(sat_mean * (1 - sat_spread),
               sat_mean * (1 + sat_spread),
               color="C1", alpha=0.15)

    # Threshold marker
    ax.axvline(6, color="0.5", ls=":", lw=0.6)
    ax.text(6.1, 0.05, r"$M_c\!\approx\!6$", fontsize=7, color="0.4")

    ax.set_xlabel(r"$m_\mathrm{act}+m_\mathrm{diff}$")
    ax.set_ylabel(r"$(1-\xi_\mathrm{LCST})/L_\mathrm{eff}$")
    ax.set_title("(a) Front-depth saturation")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_ylim(-0.02, 0.55)
    fig.tight_layout()

    pdf = FIG_DIR / "iv_c_fig6_panel_a.pdf"
    png = FIG_DIR / "iv_c_fig6_panel_a.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {pdf}", flush=True)
    print(f"   saved {png}", flush=True)


def main():
    npz = build_cache()
    print(f"   cached {npz}", flush=True)
    plot(npz)


if __name__ == "__main__":
    main()
