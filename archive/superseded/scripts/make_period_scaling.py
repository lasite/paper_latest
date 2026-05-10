#!/usr/bin/env python3
"""
make_period_scaling.py — B3: period scaling for the LCST-front cycle.

The relaxation cycle is dominated by the slow cooling phase, which on
the surface obeys
    dtheta/dt ~ - Bi_T * theta
during the quench.  The natural prediction is
    T ~ tau_cool = log(theta_peak / theta_LCST) / Bi_T ~ const / Bi_T
to within an order-unity prefactor that absorbs ignition/collapse times
and the spatial diffusion-barrier release.

This script reads the existing PDE period heatmap from
data/fig4/fig4_grid_Bi_T_S_chi.npz, restricts to oscillating cells,
and plots T_PDE vs 1/Bi_T (and vs S_chi as a secondary axis to expose
the SNIC divergence at the upper-S_chi boundary).

Output: figures_pub/period_scaling.{pdf,png}

Pure post-processing — no PDE re-runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from style_pub import set_style, PRE_DOUBLE, add_panel_label, save
set_style()

DATA_DIR = _HERE.parent / "data" / "fig4"

WP_BI_T  = 0.10
WP_S_CHI = 1.00


def main():
    d = np.load(DATA_DIR / "fig4_grid_Bi_T_S_chi.npz", allow_pickle=True)
    x  = d["x"]                # Bi_T
    y  = d["y"]                # S_chi
    regime = d["regime"]
    T = d["period"]

    osc = (regime == 1) | (regime == 2) | (regime == 3)
    osc &= np.isfinite(T) & (T > 0)

    # Build long arrays of (Bi_T, S_chi, T_PDE) over oscillating cells
    BiT_arr, Sx_arr, T_arr = [], [], []
    for i, BiT in enumerate(x):
        for j, Sx in enumerate(y):
            if osc[i, j]:
                BiT_arr.append(BiT)
                Sx_arr.append(Sx)
                T_arr.append(T[i, j])
    BiT_arr = np.array(BiT_arr)
    Sx_arr  = np.array(Sx_arr)
    T_arr   = np.array(T_arr)

    print(f"  oscillating cells: {len(T_arr)}")
    print(f"  T_PDE range : [{T_arr.min():.2f}, {T_arr.max():.2f}]")
    print(f"  1/Bi_T range: [{1/BiT_arr.max():.2f}, {1/BiT_arr.min():.2f}]")

    # Linear fit on log(T) = a*log(1/Bi_T) + b in the regular regime
    # (exclude the extreme low-S_chi and high-S_chi tails where the cycle
    #  is contaminated by either ignition delay or SNIC divergence).
    main_band = (Sx_arr >= 0.30) & (Sx_arr <= 1.05)
    if main_band.sum() > 5:
        slope, intercept = np.polyfit(
            np.log(1.0 / BiT_arr[main_band]),
            np.log(T_arr[main_band]),
            1,
        )
        print(f"  central-band log-log slope (T vs 1/Bi_T): {slope:.3f}")
        print(f"    prefactor = exp({intercept:.3f}) = {np.exp(intercept):.3f}")
    else:
        slope, intercept = np.nan, np.nan

    # ── Two-panel figure ─────────────────────────────────────────────
    # (a) T_PDE vs 1/Bi_T, points coloured by S_chi
    # (b) T_PDE vs S_chi at the working-point Bi_T row, exposing SNIC tail
    fig = plt.figure(figsize=(0.95 * PRE_DOUBLE, 2.9))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                           left=0.09, right=0.97, top=0.92, bottom=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel (a): scatter T vs 1/Bi_T
    sc = ax_a.scatter(1.0 / BiT_arr, T_arr, c=Sx_arr,
                      cmap="viridis", s=18, edgecolor="k", linewidth=0.3,
                      zorder=4)
    cb = plt.colorbar(sc, ax=ax_a, pad=0.02, fraction=0.045)
    cb.set_label(r"$S_\chi$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # Reference: T = c / Bi_T
    if np.isfinite(slope):
        BiT_grid = np.linspace(BiT_arr.min(), BiT_arr.max(), 100)
        T_pred = np.exp(intercept) * (1.0 / BiT_grid) ** slope
        ax_a.plot(1.0 / BiT_grid, T_pred, "k--", lw=0.9, alpha=0.7,
                  label=fr"fit: $T \propto (1/\mathrm{{Bi}}_T)^{{{slope:.2f}}}$")
        # Also a "T = 1/Bi_T" reference line for comparison
        ax_a.plot(1.0 / BiT_grid, 1.0 / BiT_grid, ":", color="0.4",
                  lw=0.9, label=r"$T = 1/\mathrm{Bi}_T$")
        ax_a.legend(loc="upper left", fontsize=6, framealpha=0.9,
                    handlelength=1.6)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$1/\mathrm{Bi}_T$ (cooling time)")
    ax_a.set_ylabel(r"$T_{\rm PDE}$")
    ax_a.tick_params(direction="out", length=2.5, labelsize=7)
    ax_a.grid(True, which="both", alpha=0.25, lw=0.4)

    # Panel (b): T vs S_chi at the working-point Bi_T row, showing SNIC
    i_wp = int(np.argmin(np.abs(x - WP_BI_T)))
    Sx_row, T_row = [], []
    for j, Sx in enumerate(y):
        if osc[i_wp, j]:
            Sx_row.append(Sx)
            T_row.append(T[i_wp, j])
    Sx_row = np.array(Sx_row)
    T_row  = np.array(T_row)
    ax_b.plot(Sx_row, T_row, "o-", color="#1f77b4", ms=4, lw=1.0,
              mec="k", mew=0.4, zorder=4,
              label=fr"$\mathrm{{Bi}}_T = {x[i_wp]:.3f}$")
    ax_b.axvline(WP_S_CHI, color="0.5", lw=0.6, ls=":")
    ax_b.text(WP_S_CHI, ax_b.get_ylim()[1] * 0.6, "  WP",
              fontsize=6, color="0.3", ha="left", va="top")
    ax_b.set_xlabel(r"$S_\chi$")
    ax_b.set_ylabel(r"$T_{\rm PDE}$")
    ax_b.tick_params(direction="out", length=2.5, labelsize=7)
    ax_b.legend(loc="upper left", fontsize=6, framealpha=0.95,
                handlelength=1.6)
    ax_b.grid(True, alpha=0.25, lw=0.4)

    add_panel_label(ax_a, "a", outside=False, x=0.03, y=0.97)
    add_panel_label(ax_b, "b", outside=False, x=0.03, y=0.97)

    save(fig, "period_scaling")


if __name__ == "__main__":
    main()
