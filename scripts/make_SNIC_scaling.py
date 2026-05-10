#!/usr/bin/env python3
"""
make_SNIC_scaling.py — C3 renderer.

Reads scan_SNIC_S_chi.py's cache and (i) plots the period T(S_chi)
near the upper-S_chi boundary at fixed Bi_T, (ii) fits the divergence
against the two canonical scenarios:
  SNIC      : T = a (S_chi_c - S_chi)^{-1/2} + b
  homoclinic: T = a (-log(S_chi_c - S_chi))   + b
fitting S_chi_c jointly.  Reports the residual sum of squares (RSS)
for each model.

Output: figures_pub/SNIC_scaling.{pdf,png}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from style_pub import set_style, PRE_DOUBLE, save
set_style()

DATA_DIR = _HERE.parent / "data" / "fig5"


def fit_snic(S, T):
    """Fit T = a*(Sc-S)^(-1/2) + b with Sc, a, b free."""
    def loss(params):
        Sc, a, b = params
        if Sc <= S.max():
            return 1e10
        pred = a * (Sc - S) ** (-0.5) + b
        if not np.isfinite(pred).all():
            return 1e10
        return float(np.sum((T - pred) ** 2))
    Sc0 = S.max() + 0.1
    res = minimize(loss, x0=[Sc0, 1.0, T.min()],
                   method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
    return res.x, float(res.fun)


def fit_homoclinic(S, T):
    """Fit T = a*(-log(Sc-S)) + b with Sc, a, b free."""
    def loss(params):
        Sc, a, b = params
        if Sc <= S.max():
            return 1e10
        pred = a * (-np.log(Sc - S)) + b
        if not np.isfinite(pred).all():
            return 1e10
        return float(np.sum((T - pred) ** 2))
    Sc0 = S.max() + 0.1
    res = minimize(loss, x0=[Sc0, 1.0, T.min()],
                   method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
    return res.x, float(res.fun)


def main():
    z = np.load(DATA_DIR / "SNIC_scan_S_chi.npz")
    S = z["S_chi_vals"]
    T = z["period"]
    reg = z["regime"]
    BiT = float(z["Bi_T_fixed"])

    # Keep cells where the period was successfully extracted via the
    # fallback peak detector (the regime classifier may report failed
    # for the whole grid, since with N=121 + t_end=200 the
    # `classify_run` heuristic drops some otherwise-clean cycles).
    osc = np.isfinite(T) & (T > 0)
    print(f"  cells with extracted period: {int(osc.sum())} / {len(S)}",
          flush=True)
    if not osc.any():
        print("  no period data — abort", flush=True)
        return

    S_osc = S[osc]
    T_osc = T[osc]

    # Restrict the fit to the upper-S_chi divergent tail (top 50% of points
    # by S_chi value, where the divergence dominates).
    S_max = S_osc.max()
    n_tail = max(int(0.5 * len(S_osc)), 6)
    idx = np.argsort(S_osc)[-n_tail:]
    S_tail = S_osc[idx]
    T_tail = T_osc[idx]
    print(f"  tail fit on top {len(S_tail)} pts; "
          f"S_chi in [{S_tail.min():.3f}, {S_tail.max():.3f}], "
          f"T in [{T_tail.min():.2f}, {T_tail.max():.2f}]", flush=True)

    snic_p, snic_rss = fit_snic(S_tail, T_tail)
    homo_p, homo_rss = fit_homoclinic(S_tail, T_tail)
    Sc_snic, a_snic, b_snic = snic_p
    Sc_homo, a_homo, b_homo = homo_p

    print(f"  SNIC      fit: Sc={Sc_snic:.4f}, a={a_snic:.3f}, b={b_snic:.3f}, "
          f"RSS={snic_rss:.4f}", flush=True)
    print(f"  homoclinic fit: Sc={Sc_homo:.4f}, a={a_homo:.3f}, b={b_homo:.3f}, "
          f"RSS={homo_rss:.4f}", flush=True)
    winner = "SNIC" if snic_rss < homo_rss else "homoclinic"
    print(f"  fit winner: {winner}", flush=True)

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(0.9 * PRE_DOUBLE, 2.9))
    ax_a, ax_b = axes
    fig.subplots_adjust(left=0.10, right=0.97, wspace=0.32,
                        top=0.92, bottom=0.18)

    # Panel (a): T vs S_chi linear axes; fits overlaid
    ax_a.plot(S_osc, T_osc, "o", color="#1f77b4", ms=4,
              mec="k", mew=0.4, zorder=4, label="PDE")

    S_dense = np.linspace(S_osc.min(), max(Sc_snic, Sc_homo) - 1e-3, 400)
    if Sc_snic > S_osc.min():
        T_snic = a_snic * (Sc_snic - S_dense) ** (-0.5) + b_snic
        T_snic = np.clip(T_snic, 0, 200)
        ax_a.plot(S_dense, T_snic, "-", color="#d62728", lw=1.0,
                  label=fr"SNIC: $T\!=\!a(S_\chi^{{c}}\!-\!S_\chi)^{{-1/2}}\!+\!b$"
                        fr", $S_\chi^{{c}}\!=\!{Sc_snic:.3f}$")
    if Sc_homo > S_osc.min():
        T_homo = a_homo * (-np.log(Sc_homo - S_dense)) + b_homo
        T_homo = np.clip(T_homo, 0, 200)
        ax_a.plot(S_dense, T_homo, "--", color="#2ca02c", lw=1.0,
                  label=fr"homo: $T\!=\!-a\log(S_\chi^{{c}}\!-\!S_\chi)\!+\!b$"
                        fr", $S_\chi^{{c}}\!=\!{Sc_homo:.3f}$")
    ax_a.axvline(S_max, color="0.5", ls=":", lw=0.6, zorder=1)
    ax_a.set_xlabel(r"$S_\chi$")
    ax_a.set_ylabel(r"$T_{\rm PDE}$")
    ax_a.set_ylim(0, max(T_tail.max() * 1.3, 30))
    ax_a.tick_params(direction="out", length=2.5, labelsize=7)
    ax_a.legend(loc="upper left", fontsize=5.8, framealpha=0.95,
                handlelength=1.6)
    ax_a.set_title(rf"$\mathrm{{Bi}}_T = {BiT:.3f}$ row", fontsize=8)

    # Panel (b): scaling test in log-log against (Sc-S)
    # Pick the better-fit Sc for the diagnostic
    Sc = Sc_snic if snic_rss < homo_rss else Sc_homo
    delta_S = Sc - S_tail
    # SNIC: T = a*delta^(-1/2) + b → (T-b) ∝ delta^(-1/2)
    # Homo: T = -a*log(delta) + b → (T-b) ∝ -log(delta), linear in log(delta).
    ax_b.loglog(delta_S, T_tail - min(b_snic, b_homo, 0), "o",
                color="#1f77b4", ms=4, mec="k", mew=0.4, zorder=4,
                label="PDE")
    delta_dense = np.geomspace(max(delta_S.min() / 5, 1e-3),
                               delta_S.max() * 1.5, 200)
    # SNIC reference line: slope -1/2 in log-log
    ax_b.plot(delta_dense, a_snic * delta_dense ** (-0.5),
              "-", color="#d62728", lw=1.0,
              label=r"SNIC slope $-1/2$")
    ax_b.plot(delta_dense, -a_homo * np.log(delta_dense),
              "--", color="#2ca02c", lw=1.0,
              label=r"homoclinic $-\log\Delta$")
    ax_b.set_xlabel(r"$\Delta S_\chi = S_\chi^c - S_\chi$")
    ax_b.set_ylabel(r"$T - b$")
    ax_b.tick_params(direction="out", length=2.5, labelsize=7,
                     which="both")
    ax_b.legend(loc="lower left", fontsize=6, framealpha=0.95,
                handlelength=1.6)
    ax_b.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_b.set_title(rf"winner: {winner}", fontsize=8)

    save(fig, "SNIC_scaling")


if __name__ == "__main__":
    main()
