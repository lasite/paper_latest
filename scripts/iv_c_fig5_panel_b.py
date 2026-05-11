#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fig5_panel_b.py — Amplitude universal constant.

Plots measured h_eff = Delta_theta_surf vs the asymptotic prediction
h_pred = theta_up - theta_lo (Phase P fold value), across all
oscillating points in Phase A (6 points at varying (Bi_T, S_chi)).

Source data: data/iv_c/phaseA/phaseA_check.npz
Cached data: data/iv_c/phaseE/fig5_panel_b.npz
Output:      Figure/pub/iv_c_fig5_panel_b.{pdf,png}
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
    """Pull from phaseA_check.npz."""
    src = np.load(_HERE.parent / "data" / "iv_c" / "phaseA" /
                  "phaseA_check.npz", allow_pickle=True)
    Bi_T = np.array(src["Bi_T"])
    S_chi = np.array(src["S_chi"])
    meas_h = np.array(src["meas_h"])
    rel_err = np.array(src["rel_err"])
    h_pred = float(src["h_pred"])
    max_dev = float(src["max_dev"])
    mean_dev = float(src["mean_dev"])

    npz = CACHE / "fig5_panel_b.npz"
    np.savez(npz,
             Bi_T=Bi_T, S_chi=S_chi, meas_h=meas_h, rel_err=rel_err,
             h_pred=h_pred, max_dev=max_dev, mean_dev=mean_dev)
    return npz


def plot(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Bi_T = d["Bi_T"]
    S_chi = d["S_chi"]
    meas_h = d["meas_h"]
    h_pred = float(d["h_pred"])

    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    # x-axis: arbitrary index labelled by (Bi_T, S_chi)
    n = len(Bi_T)
    idx = np.arange(n)

    # Color by Bi_T
    bi_unique = sorted(set(Bi_T.tolist()))
    cmap = plt.get_cmap("viridis")
    for i in idx:
        c = cmap(bi_unique.index(float(Bi_T[i])) /
                 max(1, len(bi_unique) - 1))
        ax.plot(idx[i], meas_h[i], "o", color=c, ms=8,
                markeredgecolor="white", markeredgewidth=0.5)

    # Reference line at h_pred
    ax.axhline(h_pred, ls="--", color="0.4", lw=0.8,
               label=fr"$\theta_{{up}}-\theta_{{lo}}={h_pred:.3f}$")
    # 20% band
    ax.axhspan(h_pred * 0.8, h_pred * 1.2, color="C0", alpha=0.10,
               label="±20% band")

    labels = [fr"$({b:.2f},{s:.1f})$"
              for b, s in zip(Bi_T, S_chi)]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax.set_xlabel(r"$(\mathrm{Bi}_T,\,S_\chi)$")
    ax.set_ylabel(r"$h_\mathrm{eff}=\Delta\theta_\mathrm{surf}$")
    ax.set_title("(b) Amplitude collapse")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    fig.tight_layout()

    pdf = FIG_DIR / "iv_c_fig5_panel_b.pdf"
    png = FIG_DIR / "iv_c_fig5_panel_b.png"
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
