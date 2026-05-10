#!/usr/bin/env python3
"""
make_fig5.py - Fig 5 (paper): linear stability is necessary but not sufficient.

Two-panel composite for IV.C:
  (a) Homogeneous-SS branch existence map on (Bi_T, S_chi) with the
      analytic Whitney-cusp fold overlay (re-uses panel_branches from
      make_homogeneous_ss).
  (b) Period-divergence scaling at the upper-S_chi exit; SNIC vs
      homoclinic fits on linear axes (the log-log diagnostic in the
      standalone SNIC_scaling.pdf is dropped here for a clean 2-panel
      layout that matches the PLAN and the main.tex caption).

Cache-only renderer; cheap to re-run.

Output: Figure/pub/fig5.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from make_homogeneous_ss import load_lsa, load_fold, panel_branches
from make_SNIC_scaling import fit_snic, fit_homoclinic

set_style()

DATA_DIR_FIG5 = _HERE.parent / "data" / "fig5"
OUT_DIR = _HERE.parent / "Figure" / "pub"


def panel_snic(ax):
    """Period-divergence scaling at the upper-S_chi exit (linear axes)."""
    z = np.load(DATA_DIR_FIG5 / "SNIC_scan_S_chi.npz")
    S = z["S_chi_vals"]
    T = z["period"]
    BiT = float(z["Bi_T_fixed"])

    osc = np.isfinite(T) & (T > 0)
    if not osc.any():
        ax.text(0.5, 0.5, "no period data", transform=ax.transAxes,
                ha="center", va="center")
        return None
    S_osc, T_osc = S[osc], T[osc]

    n_tail = max(int(0.5 * len(S_osc)), 6)
    idx = np.argsort(S_osc)[-n_tail:]
    S_tail, T_tail = S_osc[idx], T_osc[idx]

    snic_p, snic_rss = fit_snic(S_tail, T_tail)
    homo_p, homo_rss = fit_homoclinic(S_tail, T_tail)
    Sc_snic, a_snic, b_snic = snic_p
    Sc_homo, a_homo, b_homo = homo_p

    ax.plot(S_osc, T_osc, "o", color="#1f77b4", ms=4,
            mec="k", mew=0.4, zorder=4, label="PDE")

    S_dense = np.linspace(S_osc.min(), max(Sc_snic, Sc_homo) - 1e-3, 400)
    if Sc_snic > S_osc.min():
        T_snic = a_snic * (Sc_snic - S_dense) ** (-0.5) + b_snic
        T_snic = np.clip(T_snic, 0, 200)
        ax.plot(S_dense, T_snic, "-", color="#d62728", lw=1.0,
                label=(rf"SNIC $T\!\propto\!(S_\chi^c\!-\!S_\chi)^{{-1/2}}$"
                       rf", RSS={snic_rss:.3f}"))
    if Sc_homo > S_osc.min():
        T_homo = a_homo * (-np.log(Sc_homo - S_dense)) + b_homo
        T_homo = np.clip(T_homo, 0, 200)
        ax.plot(S_dense, T_homo, "--", color="#2ca02c", lw=1.0,
                label=(rf"homo $T\!\propto\!-\ln(S_\chi^c\!-\!S_\chi)$"
                       rf", RSS={homo_rss:.3f}"))

    ax.axvline(S_osc.max(), color="0.5", ls=":", lw=0.6, zorder=1)
    ax.set_xlabel(r"$S_\chi$")
    ax.set_ylabel(r"$T_{\rm PDE}$")
    ax.set_ylim(0, max(T_tail.max() * 1.3, 30))
    ax.tick_params(direction="out", length=2.5)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.95,
              handlelength=1.6)
    ax.set_title(rf"$\mathrm{{Bi}}_T={BiT:.3f}$ slice, "
                 rf"$S_\chi^c\!\approx\!{Sc_snic:.3f}$",
                 fontsize=8)

    return dict(Sc_snic=Sc_snic, snic_rss=snic_rss,
                Sc_homo=Sc_homo, homo_rss=homo_rss)


def main():
    lsa = load_lsa()
    fold_segs = load_fold()
    if fold_segs:
        n_pts = sum(s.shape[0] for s in fold_segs)
        print(f"  fold-curve overlay: {len(fold_segs)} segs, {n_pts} pts")

    fig = plt.figure(figsize=(PRE_DOUBLE, 3.0))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[0.42, 0.58],
                           wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.90, bottom=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_branches(ax_a, lsa, fold_segs=fold_segs)
    fit_info = panel_snic(ax_b)
    if fit_info:
        print(f"  SNIC fit:      Sc={fit_info['Sc_snic']:.4f}, "
              f"RSS={fit_info['snic_rss']:.4f}")
        print(f"  homoclinic fit: Sc={fit_info['Sc_homo']:.4f}, "
              f"RSS={fit_info['homo_rss']:.4f}")

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig5.pdf"
    png = OUT_DIR / "fig5.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


if __name__ == "__main__":
    main()
