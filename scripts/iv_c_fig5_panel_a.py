#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fig5_panel_a.py — Period collapse.

Plots T_PDE * Bi_T vs Bi_T for 5 S_chi values, against the theoretical
asymptote ln(theta_up/theta_lo) (computed in Phase P and reused here).

Source data: data/iv_c/phaseB/phaseB_check.npz
Cached data: data/iv_c/phaseE/fig5_panel_a.npz
Output:      Figure/pub/iv_c_fig5_panel_a.{pdf,png}
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
    """Pull from phaseB_check.npz and the Phase P fold solver."""
    src = np.load(_HERE.parent / "data" / "iv_c" / "phaseB" /
                  "phaseB_check.npz", allow_pickle=True)
    Bi_T = np.array(src["Bi_T"])
    S_chi = np.array(src["S_chi"])
    T_BiT = np.array(src["T_BiT"])    # shape (n_BiT, n_S_chi); NaN = no cycle
    asym = float(np.array(src["asym"]).flatten()[0])
    max_err = float(src["max_err"])
    mean_err = float(src["mean_err"])

    npz = CACHE / "fig5_panel_a.npz"
    np.savez(npz,
             Bi_T=Bi_T, S_chi=S_chi, T_BiT=T_BiT,
             asym=asym, max_err=max_err, mean_err=mean_err)
    return npz


def plot(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Bi_T = d["Bi_T"]
    S_chi = d["S_chi"]
    T_BiT = d["T_BiT"]
    asym = float(d["asym"])

    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    cmap = plt.get_cmap("viridis")
    for j, s in enumerate(S_chi):
        col = cmap(j / max(1, len(S_chi) - 1))
        y = T_BiT[:, j]
        mask = ~np.isnan(y)
        ax.plot(Bi_T[mask], y[mask], "o-", color=col, lw=1.0, ms=4,
                label=fr"$S_\chi={s:.1f}$")

    ax.axhline(asym, ls="--", color="0.4", lw=0.8,
               label=fr"$\ln(\theta_{{up}}/\theta_{{lo}})={asym:.3f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$")
    ax.set_ylabel(r"$T_\mathrm{PDE}\cdot\mathrm{Bi}_T$")
    ax.set_title("(a) Period collapse")
    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.8)
    fig.tight_layout()

    pdf = FIG_DIR / "iv_c_fig5_panel_a.pdf"
    png = FIG_DIR / "iv_c_fig5_panel_a.png"
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
