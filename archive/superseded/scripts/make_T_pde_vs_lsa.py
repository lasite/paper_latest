#!/usr/bin/env python3
"""
make_T_pde_vs_lsa.py — RETIRED.  The LSA-vs-PDE period contrast was
dropped from §IV.C: the underlying 25x25 PDE regime grid is too coarse
to draw smooth contours, and the physical conclusion (the linear
period of any homogeneous branch cannot match a relaxation cycle that
shuttles between two branches of a *spatially generated* slow
manifold) is now stated as a direct corollary of the slow-manifold
construction in §IV.B.  Kept only for reproducibility; not part of
the figure pipeline.

Original docstring follows.

make_T_pde_vs_lsa.py — C2: contrast PDE limit-cycle period against the
LSA linear period across the (Bi_T, S_chi) plane.

Reads existing caches:
  data/fig4/fig4_grid_Bi_T_S_chi.npz       (PDE regime, period)
  data/fig4/hopf_boundary_Bi_T_S_chi.npz   (LSA omega for the Hopf eigenpair)

For every grid cell where both:
  * the PDE oscillates (regime in {LCST_FRONT, BULK_HOPF, GLOBAL_COLLAPSE}), and
  * the LSA has a complex eigenpair with positive real part (omega > 0),
compute T_LSA = 2*pi/omega and the ratio T_PDE/T_LSA.

Plot as a single heatmap; values >> 1 indicate the cycle is in the
strongly-nonlinear relaxation-oscillator regime, where the linear
period is a poor predictor of the actual cycle period.

Output: figures_pub/T_pde_vs_lsa.{pdf,png}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from style_pub import set_style, PRE_DOUBLE, add_panel_label, save
set_style()

DATA_DIR = _HERE.parent / "data" / "fig4"

WP_BI_T  = 0.10
WP_S_CHI = 1.00
C_BI_T   = 0.059
C_S_CHI  = 1.80


def main():
    pde = np.load(DATA_DIR / "fig4_grid_Bi_T_S_chi.npz", allow_pickle=True)
    lsa = np.load(DATA_DIR / "hopf_boundary_Bi_T_S_chi.npz", allow_pickle=True)

    Bi_T  = pde["x"]
    S_chi = pde["y"]
    regime = pde["regime"]
    T_pde  = pde["period"]
    omega  = lsa["omega"]
    re_max = lsa["re_max_complex"]
    if not (np.array_equal(Bi_T, lsa["x"]) and np.array_equal(S_chi, lsa["y"])):
        raise RuntimeError("PDE and LSA grids do not match")

    osc = (regime == 1) | (regime == 2) | (regime == 3)
    lsa_hopf = (np.isfinite(omega) & (omega > 1e-6)
                & np.isfinite(re_max) & (re_max > 0))

    # T_LSA only where Hopf is unstable
    T_lsa = np.full_like(omega, np.nan, dtype=float)
    mask_lsa = lsa_hopf
    T_lsa[mask_lsa] = 2 * np.pi / omega[mask_lsa]

    ratio = np.full_like(T_pde, np.nan, dtype=float)
    mask_both = osc & mask_lsa & np.isfinite(T_pde) & (T_pde > 0)
    ratio[mask_both] = T_pde[mask_both] / T_lsa[mask_both]

    # Diagnostic stats
    print(f"  Cells with PDE oscillation: {int(osc.sum())} / {osc.size}")
    print(f"  Cells with LSA Hopf-unstable: {int(lsa_hopf.sum())}")
    print(f"  Cells with BOTH (= ratio defined): {int(mask_both.sum())}")
    if mask_both.any():
        r = ratio[mask_both]
        print(f"  ratio T_PDE/T_LSA: median={np.median(r):.2f}, "
              f"min={r.min():.2f}, max={r.max():.2f}")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(0.55 * PRE_DOUBLE, 3.1))

    # Background: outline PDE oscillation region
    ax.contour(Bi_T, S_chi, osc.astype(float).T, levels=[0.5],
               colors="#1f3a72", linewidths=1.4, zorder=4)
    # Outline LSA Hopf region
    ax.contour(Bi_T, S_chi, lsa_hopf.astype(float).T, levels=[0.5],
               colors="#666666", linewidths=0.8, linestyles="--",
               zorder=3)

    # Heatmap of log10(ratio) on a diverging scale centred at log10(1)=0,
    # so values < 1 (PDE faster than linear) and > 1 (PDE slower) are
    # visually symmetric.
    log_ratio = np.full_like(ratio, np.nan, dtype=float)
    log_ratio[mask_both] = np.log10(ratio[mask_both])
    Z = np.ma.masked_invalid(log_ratio.T)
    if Z.count() > 0:
        v = max(abs(np.nanmin(log_ratio)), abs(np.nanmax(log_ratio)))
        v = max(v, 0.5)  # at least one decade either side
        pcm = ax.pcolormesh(Bi_T, S_chi, Z, cmap="RdBu_r",
                            vmin=-v, vmax=v, shading="auto", zorder=1)
        cb = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.045,
                          extend="both")
        cb.set_label(r"$\log_{10}(T_{\rm PDE}/T_{\rm LSA})$", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        # Tick labels with the actual ratio value at major decades
        cb.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb.set_ticklabels(["0.1", "0.32", "1", "3.2", "10"])
        cb.set_label(r"$T_{\rm PDE}/T_{\rm LSA}$", fontsize=7)

    # Markers
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

    # Manual legend
    handles = [
        plt.Line2D([], [], color="#1f3a72", lw=1.4,
                   label="PDE oscillation"),
        plt.Line2D([], [], color="#666666", lw=0.8, ls="--",
                   label="LSA Hopf-unstable"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              framealpha=0.95, handlelength=1.6, borderpad=0.4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$")
    ax.set_ylabel(r"$S_\chi$")
    ax.set_xlim(Bi_T.min(), Bi_T.max())
    ax.set_ylim(S_chi.min(), S_chi.max())
    ax.tick_params(direction="out", length=2.5)

    save(fig, "T_pde_vs_lsa")


if __name__ == "__main__":
    main()
