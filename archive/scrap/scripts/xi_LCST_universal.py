#!/usr/bin/env python3
"""
xi_LCST_universal.py — Universal-invariance summary for ξ_LCST.

Combines all five Thiele-axis scans (D0, Da, Bi_c, Bi_T, alpha) from
thiele_collapse.py with the two thermo-axis scans (chi_inf, S_chi)
from thermo_axis_scan.py and the N-convergence test from
xi_LCST_N_convergence.py to make ONE summary figure showing that
ξ_LCST sits at a parameter-robust value ≈ 0.92 across seven independent
controls and as N → ∞.

Outputs:
  Figure/pub/fig3_xi_LCST_universal.{pdf,png}
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT

set_style()

DATA3 = (_HERE.parent / "data" / "fig3").resolve()
OUT_DIR = (_HERE.parent / "Figure" / "pub").resolve()


PARAM_LABEL = {
    "Bi_T":   r"$\mathrm{Bi}_T$",
    "Bi_c":   r"$\mathrm{Bi}_c$",
    "Da":     r"$\mathrm{Da}$",
    "alpha":  r"$\alpha$",
    "D0":     r"$D_0$",
    "chi_inf":r"$\chi_\infty$",
    "S_chi":  r"$S_\chi$",
}
PARAM_MARKER = {"Bi_T":"o", "D0":"s", "alpha":"^", "Da":"v",
                "Bi_c":"D", "chi_inf":"P", "S_chi":"X"}
PARAM_COLOR  = {"Bi_T":"#1f5fa3", "D0":"#a23e1c", "alpha":"#137a73",
                "Da":"#7f3f98", "Bi_c":"#c89a2c",
                "chi_inf":"#3676b8", "S_chi":"#d6604d"}


def load_scan(path):
    if not path.exists():
        return None
    return dict(np.load(path))


def main():
    # Five "kinetic / transport" scans
    kin_scans = {}
    for p in ["D0", "Da", "Bi_c", "Bi_T", "alpha"]:
        z = load_scan(DATA3 / f"thiele_collapse_{p}.npz")
        if z is not None:
            kin_scans[p] = z

    # Two "thermodynamic" scans
    thermo_scans = {}
    for p in ["chi_inf", "S_chi"]:
        z = load_scan(DATA3 / f"thermo_axis_{p}.npz")
        if z is not None:
            thermo_scans[p] = z

    n_convergence = load_scan(DATA3 / "xi_LCST_N_convergence.npz")

    fig, axes = plt.subplots(1, 3, figsize=(PRE_DOUBLE, 2.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.22,
                        wspace=0.42)
    # Common y range for panels (a, b) so the invariance is visible
    Y_LO, Y_HI = 0.86, 0.96

    # ── Panel (a): kinetic / transport invariance ─────────────────────
    ax = axes[0]
    all_xi = []
    for p, d in kin_scans.items():
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        x = d["vals"][m] / WORKING_POINT.get(p, 1.0)
        ax.plot(x, d["xi_LCST"][m],
                marker=PARAM_MARKER.get(p, "o"),
                color=PARAM_COLOR.get(p, "0.3"),
                lw=0.9, ms=4.5, ls="-",
                label=PARAM_LABEL.get(p, p))
        all_xi.extend(d["xi_LCST"][m].tolist())
    ax.axvline(1.0, color="0.4", lw=0.6, ls=":")
    ax.text(1.0, 0.02, " WP", fontsize=6, color="0.3",
            ha="left", va="bottom", transform=ax.get_xaxis_transform())
    if all_xi:
        m_avg = float(np.mean(all_xi))
        s_avg = float(np.std(all_xi))
        ax.axhspan(m_avg - 2*s_avg, m_avg + 2*s_avg,
                   color="0.85", alpha=0.5, lw=0, zorder=0)
        ax.text(0.02, 0.04,
                fr"$\xi_\mathrm{{LCST}}={m_avg:.4f}\pm{s_avg:.4f}$",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.5,
                bbox=dict(facecolor="white", edgecolor="0.5",
                          boxstyle="round,pad=0.18", lw=0.4))
    ax.set_xscale("log")
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel(r"$\beta/\beta_\mathrm{WP}$", fontsize=7.5)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.set_title("kinetic / transport axes", fontsize=7.5)
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9,
              handlelength=1.4, borderpad=0.3, ncol=2,
              columnspacing=0.6, labelspacing=0.25)
    add_panel_label(ax, "a")

    # ── Panel (b): thermodynamic invariance ───────────────────────────
    ax = axes[1]
    all_xi_t = []
    for p, d in thermo_scans.items():
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        x = d["vals"][m] / WORKING_POINT.get(p, 1.0)
        ax.plot(x, d["xi_LCST"][m],
                marker=PARAM_MARKER.get(p, "o"),
                color=PARAM_COLOR.get(p, "0.3"),
                lw=0.9, ms=4.5, ls="-",
                label=PARAM_LABEL.get(p, p))
        all_xi_t.extend(d["xi_LCST"][m].tolist())
    ax.axvline(1.0, color="0.4", lw=0.6, ls=":")
    ax.text(1.0, 0.02, " WP", fontsize=6, color="0.3",
            ha="left", va="bottom", transform=ax.get_xaxis_transform())
    if all_xi_t:
        m_avg = float(np.mean(all_xi_t))
        s_avg = float(np.std(all_xi_t))
        ax.axhspan(m_avg - 2*s_avg, m_avg + 2*s_avg,
                   color="0.85", alpha=0.5, lw=0, zorder=0)
        ax.text(0.02, 0.04,
                fr"$\xi_\mathrm{{LCST}}={m_avg:.4f}\pm{s_avg:.4f}$",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.5,
                bbox=dict(facecolor="white", edgecolor="0.5",
                          boxstyle="round,pad=0.18", lw=0.4))
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel(r"$\beta/\beta_\mathrm{WP}$", fontsize=7.5)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.set_title("thermodynamic axes", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9,
              handlelength=1.4, borderpad=0.3)
    add_panel_label(ax, "b")

    # ── Panel (c): N convergence — front sharpness in dx → 0 limit ──
    # The N convergence data exposes a non-trivial caveat: ξ_peak and
    # ξ_LCST converge to the same value (≈ 0.91) as N → ∞, with the
    # "mechanical halo" width ξ_LCST − ξ_peak shrinking as ~ dx. The
    # halo zone is therefore a *grid-limited* artifact of the working-
    # point N=41; the true sharp-interface limit has a single front at
    # ξ ≈ 0.91.
    ax = axes[2]
    if n_convergence is not None and len(n_convergence["N"]) >= 3:
        Ns  = n_convergence["N"]
        dxs = n_convergence["dx"]
        xi_L = n_convergence["xi_LCST"]
        xi_p = n_convergence["xi_peak"]
        ax.plot(dxs, xi_L, "o-", color="#a23e1c", lw=1.0, ms=5,
                label=r"$\xi_\mathrm{LCST}$")
        ax.plot(dxs, xi_p, "s-", color="#1f5fa3", lw=1.0, ms=4.5,
                label=r"$\xi_\mathrm{peak}$")
        # Halo width (shaded region)
        ax.fill_between(dxs, xi_p, xi_L, color="#fff0c2", alpha=0.6,
                        zorder=1, label="halo width")
        # Annotate N values at the data points
        for dx, xL, N in zip(dxs, xi_L, Ns):
            ax.annotate(f"N={int(N)}", xy=(dx, xL),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=5.0, color="0.20", ha="center")
        ax.set_xscale("log")
        ax.invert_xaxis()  # so dx → 0 (large N) is on the right
        ax.set_xlabel(r"$dx = 1/N$  (sharp limit $\to$)", fontsize=7.5)
        ax.set_ylabel(r"$\xi$", fontsize=8)
        ax.set_ylim(Y_LO, Y_HI)
        ax.tick_params(labelsize=6.5)
        ax.set_title("N convergence", fontsize=7.5)
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9,
                  handlelength=1.4, borderpad=0.3, labelspacing=0.25)
        # Halo width caveat
        halo = xi_L - xi_p
        ax.text(0.02, 0.06,
                f"halo width: {halo[0]:.3f} (N={int(Ns[0])})\n"
                f" → {halo[-1]:.3f} (N={int(Ns[-1])})\n"
                f"halo $\\sim dx$: grid-limited",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.0,
                bbox=dict(facecolor="white", edgecolor="0.5",
                          boxstyle="round,pad=0.2", lw=0.4))
    else:
        ax.text(0.5, 0.5, "N convergence cache missing",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="0.4")
    add_panel_label(ax, "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig3_xi_LCST_universal.pdf"
    png = OUT_DIR / "fig3_xi_LCST_universal.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")

    # Numerical summary (prints both panels)
    print("\n  Universal invariance summary:")
    print(f"    Kinetic/transport axes ({len(kin_scans)} params):")
    for p, d in kin_scans.items():
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        rng = (float(d["xi_LCST"][m].min()), float(d["xi_LCST"][m].max()))
        print(f"      {p:>8}: ξ_LCST ∈ [{rng[0]:.4f}, {rng[1]:.4f}]  "
              f"(span {rng[1]-rng[0]:.4f})")
    print(f"    Thermodynamic axes ({len(thermo_scans)} params):")
    for p, d in thermo_scans.items():
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        rng = (float(d["xi_LCST"][m].min()), float(d["xi_LCST"][m].max()))
        print(f"      {p:>8}: ξ_LCST ∈ [{rng[0]:.4f}, {rng[1]:.4f}]  "
              f"(span {rng[1]-rng[0]:.4f})")
    if n_convergence is not None:
        print("    Grid convergence:")
        for N, xL in zip(n_convergence["N"], n_convergence["xi_LCST"]):
            print(f"      N={int(N):>4}: ξ_LCST = {xL:.5f}")


if __name__ == "__main__":
    main()
