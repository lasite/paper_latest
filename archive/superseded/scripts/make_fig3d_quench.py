#!/usr/bin/env python3
"""
make_fig3d_quench.py — Fig 3(d): rate-factor profiles vs ξ.

Plots the time-averaged value of each multiplicative factor in the local
reaction rate
  R(ξ,τ) = u · (1-φ)^{m_act} · exp[Γ_A θ /(1+ε_T θ)]
across ξ. The factor that sits lowest at any depth is the rate-limiter
there. The solid line is the time-mean; the shaded band is the cycle min
to max, so the swing in each band shows which factor *modulates* the
rate at that depth.

Reading the plot:
  • core (ξ < ξ_c): ⟨u⟩ is the smallest factor by 6+ decades — the rate is
    set entirely by reactant starvation, and the modulation is dominated
    by the Arrhenius factor (the ⟨arrh⟩ band swings most with θ).
  • shell (ξ > ξ_c): ⟨access⟩ collapses to ~0 each cycle while ⟨u⟩ stays
    near unity — accessibility quenching takes over as the dominant
    mechanism that gates the reaction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_pub import set_style, add_panel_label
from fig3_data import panel_data, save_panel
set_style()


C_ACC  = "#a23e1c"   # red — accessibility
C_U    = "#137a73"   # teal — reactant
C_ARRH = "#d6722e"   # orange — Arrhenius


def panel_d(ax, x, access, u, arrh, xi_peak, xi_LCST=None,
            label_fs=9, tick_fs=7, legend_fs=6.5, floor=1e-12):
    """Plot the three rate-factor profiles vs ξ. Returns nothing.

    Vertical guides: dashed at ξ_peak (mechanical halo inner edge);
    dotted at ξ_LCST (collapse-front locus) if provided.
    """
    acc_mean = access.mean(axis=1)
    u_mean = u.mean(axis=1)
    arr_mean = arrh.mean(axis=1)

    acc_lo, acc_hi = access.min(axis=1), access.max(axis=1)
    u_lo,   u_hi   = np.maximum(u.min(axis=1), floor), u.max(axis=1)
    arr_lo, arr_hi = arrh.min(axis=1), arrh.max(axis=1)

    # Bands (cycle min/max)
    ax.fill_between(x, acc_lo, acc_hi, color=C_ACC, alpha=0.18, lw=0)
    ax.fill_between(x, u_lo,   u_hi,   color=C_U,   alpha=0.18, lw=0)
    ax.fill_between(x, arr_lo, arr_hi, color=C_ARRH, alpha=0.18, lw=0)

    # Time-mean lines
    ax.plot(x, acc_mean, color=C_ACC, lw=1.4,
            label=r"$\langle (1-\phi)^{m_{\rm act}}\rangle$")
    ax.plot(x, u_mean, color=C_U, lw=1.4, label=r"$\langle u\rangle$")
    ax.plot(x, arr_mean, color=C_ARRH, lw=1.4,
            label=r"$\langle\exp[\Gamma_A\theta/(1+\varepsilon_T\theta)]\rangle$")

    ax.axvline(xi_peak, color="k", lw=0.7, ls="--", zorder=5)
    if xi_LCST is not None:
        ax.axvline(xi_LCST, color="0.35", lw=0.7, ls=":", zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(floor, max(2e2, arr_hi.max() * 1.3))
    ax.set_xlabel(r"$\xi$", fontsize=label_fs)
    ax.set_ylabel("rate factor", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs, direction="out", length=2.5)
    ax.legend(fontsize=legend_fs, loc="lower right",
              framealpha=0.92, handlelength=1.4, borderpad=0.3,
              labelspacing=0.25)


def main():
    d = panel_data()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.16)
    panel_d(ax, d["x"], d["access"], d["u"], d["arrh"],
            d["xi_peak"], d["xi_LCST"])
    add_panel_label(ax, "d")
    save_panel(fig, "fig3d_quench")
    plt.close(fig)


if __name__ == "__main__":
    main()
