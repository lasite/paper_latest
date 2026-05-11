#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_lt_patch_plot.py — L_T patch L_T.3: combined collapse plot.

Combines Phase D's L_c-branch points (saturated cohort: M_sum >= 6, cycle
state) with the L_T-patch points (L_T-limited regime: alpha=0.04, delta=0.50)
into a single (1-xi_LCST) vs min(L_T, L_c) scatter, with the Phase D plateau
ratio (0.388) drawn as a reference line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE  # type: ignore
PRE_SINGLE = PRE_DOUBLE / 2.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = _HERE.parent / "Figure" / "pub"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Phase D — main grid (only cycle cells in the saturated cohort)
    d = np.load(_HERE.parent / "data" / "iv_c" / "phaseD" /
                "phaseD_check.npz", allow_pickle=True)
    m_act = d["m_act"]
    m_diff = d["m_diff"]
    xi_grid = d["xi_LCST"]
    cls_grid = d["classification"]
    M_sum_grid = d["M_sum"]

    # delta and Bi_c defaults for Phase D — L_c is constant
    L_c_default = float(np.sqrt(0.08 / 0.70))   # 0.3381
    alpha_default = 0.20
    # L_T depends on Bi_T which is 0.10 throughout the main grid
    L_T_main = float(np.sqrt(alpha_default / 0.10))  # 1.4142

    # Collect Phase D "cycle" points with M_sum >= 6 (saturated cohort)
    pdpts = []
    for i in range(len(m_act)):
        for j in range(len(m_diff)):
            if cls_grid[i, j] == "cycle" and M_sum_grid[i, j] >= 6:
                pdpts.append({
                    "label": "Phase D main grid",
                    "L_T": L_T_main, "L_c": L_c_default,
                    "xi": float(xi_grid[i, j]),
                    "m_act": int(m_act[i]), "m_diff": int(m_diff[j]),
                })

    # Phase D slice — Bi_T variations at fixed (m_act, m_diff) defaults
    Bi_T_slice = d["Bi_T_slice"]
    L_T_slice = d["L_T_slice"]
    L_c_slice = d["L_c_slice"]
    xi_slice = d["xi_slice"]
    cls_slice = d["cls_slice"]
    pdpts_slice = []
    for k in range(len(Bi_T_slice)):
        if str(cls_slice[k]) == "cycle":
            pdpts_slice.append({
                "label": "Phase D Bi_T slice",
                "L_T": float(L_T_slice[k]),
                "L_c": float(L_c_slice[k]),
                "xi": float(xi_slice[k]),
                "Bi_T": float(Bi_T_slice[k]),
            })

    # L_T patch
    sw = np.load(_HERE.parent / "data" / "iv_c" / "lt_patch" /
                 "sweep_final.npz", allow_pickle=True)
    results = sw["results"]
    ltpts = []
    for r in results:
        rd = r.item() if hasattr(r, "item") else r
        if rd["is_oscillating"] and 0 < rd["xi_LCST"] < 1:
            ltpts.append({
                "label": "L_T patch",
                "L_T": rd["L_T"], "L_c": rd["L_c"],
                "xi": rd["xi_LCST"],
                "Bi_T": rd["Bi_T"],
            })

    # Plot
    set_style()
    fig, ax = plt.subplots(figsize=(PRE_SINGLE, PRE_SINGLE * 0.82))

    # Phase D main grid (L_c-limited, all same L_min)
    if pdpts:
        Lmin = np.array([min(p["L_T"], p["L_c"]) for p in pdpts])
        omx = np.array([1 - p["xi"] for p in pdpts])
        ax.scatter(Lmin, omx, marker="o", s=20, color="C3",
                   edgecolors="white", linewidths=0.4, alpha=0.85,
                   label=fr"Phase D ($L_c$-limited, $\delta=0.08$)")

    # Phase D slice
    if pdpts_slice:
        Lmin = np.array([min(p["L_T"], p["L_c"]) for p in pdpts_slice])
        omx = np.array([1 - p["xi"] for p in pdpts_slice])
        ax.scatter(Lmin, omx, marker="D", s=24, color="C1",
                   edgecolors="white", linewidths=0.4, alpha=0.95,
                   label=r"Phase D Bi$_T$ slice")

    # L_T patch
    if ltpts:
        Lmin = np.array([min(p["L_T"], p["L_c"]) for p in ltpts])
        omx = np.array([1 - p["xi"] for p in ltpts])
        ax.scatter(Lmin, omx, marker="s", s=30, color="C0",
                   edgecolors="white", linewidths=0.4,
                   label=fr"L$_T$ patch ($\alpha=0.04$, $\delta=0.50$)")
        # Annotate with Bi_T for clarity
        for p in ltpts:
            ax.annotate(fr"$\mathrm{{Bi}}_T={p['Bi_T']:.2f}$",
                        (min(p["L_T"], p["L_c"]), 1 - p["xi"]),
                        textcoords="offset points", xytext=(4, -8),
                        fontsize=6, color="C0")

    # Plateau reference line c = 0.388
    Lref = np.linspace(0.0, max(0.9, 1.1 * (max([min(p["L_T"], p["L_c"])
                       for p in (pdpts + pdpts_slice + ltpts)]
                       ) if (pdpts + pdpts_slice + ltpts) else 1.0)), 200)
    ax.plot(Lref, 0.388 * Lref, "--", color="0.4", lw=0.8,
            label=r"$0.388\cdot\min(L_T, L_c)$")

    ax.set_xlabel(r"$\min(L_T,\,L_c)$")
    ax.set_ylabel(r"$1 - \xi_\mathrm{LCST}$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    fig.tight_layout()

    pdf = FIG_DIR / "iv_c_lt_collapse.pdf"
    png = FIG_DIR / "iv_c_lt_collapse.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {pdf}", flush=True)
    print(f"   saved {png}", flush=True)

    # Quick console summary
    print(f"\n   Phase D main-grid points (M_sum>=6, cycle): {len(pdpts)}",
          flush=True)
    print(f"   Phase D Bi_T-slice points (cycle):           "
          f"{len(pdpts_slice)}", flush=True)
    print(f"   L_T patch valid points:                       {len(ltpts)}",
          flush=True)

    all_pts = pdpts + pdpts_slice + ltpts
    ratios = np.array([(1 - p["xi"]) / min(p["L_T"], p["L_c"])
                       for p in all_pts])
    print(f"\n   Combined (1-xi)/min(L_T,L_c):", flush=True)
    print(f"     mean = {ratios.mean():.4f}", flush=True)
    print(f"     std  = {ratios.std():.4f}", flush=True)
    print(f"     spread = {ratios.std()/ratios.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
