#!/usr/bin/env python3
"""
make_fig3.py — Composite Fig 3: propagating LCST collapse front and its
mechanical halo in the 1D slab.

At the working point of Fig 2 the slab partitions into a passive thermal
core (ξ < ξ_peak), a thin mechanical halo (ξ_peak ≤ ξ ≤ ξ_LCST) where
solvent expelled by the collapsing shell drives a transient over-
swelling without crossing LCST, and an outer collapse-front zone
(ξ > ξ_LCST) that periodically crosses the LCST and undergoes the
accessibility quench. The six panels quantify and explain this:

  Top:    (a) J/J_eq envelope  (b) θ,u envelopes  (c) phase portraits
  Bottom: (d) rate-factor profile  (e) heat source vs reactant
          (f) ξ_peak / ξ_LCST vs scanned parameters (parameter invariance)

J is normalized by the cold-bath equilibrium J_eq(θ=0,μ=μ_b), so
J/J_eq=1 is the gel sitting in equilibrium with the bath.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig3_data import panel_data

from make_fig3a_envelope     import panel_a
from make_fig3b_envelopes_thu import panel_b
from make_fig3c_phase_depth  import panel_c
from make_fig3d_quench       import panel_d
from make_fig3e_heat         import panel_e
from make_fig3f_xi_peak_scan import panel_f as panel_f_scan, load_scans

set_style()


# Composite font sizes (each cell ~2 in wide).
LBL_FS    = 8
TICK_FS   = 6.5
LEG_FS    = 6


def main():
    d = panel_data()

    fig = plt.figure(figsize=(PRE_DOUBLE, 6.4))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.50,
                           left=0.07, right=0.96, top=0.94, bottom=0.08)

    # ── (a) J/J_eq envelope with three zones ──────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, d["x"], d["J_min"], d["J_max"], d["J_eq"],
            d["xi_peak"], d["xi_LCST"],
            label_fs=LBL_FS, tick_fs=TICK_FS, annot_fs=6.5)
    leg = ax_a.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(LEG_FS)
    add_panel_label(ax_a, "a")

    # ── (b) θ / u envelopes ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b, d["x"], d["th_min"], d["th_max"],
            d["u_min"], d["u_max"], d["xi_peak"], d["xi_LCST"],
            label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS - 0.5)
    add_panel_label(ax_b, "b")

    # ── (c) phase portraits at multiple ξ (J normalized by J_eq) ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_c(ax_c, d["x"], d["t"], d["J"], d["theta"], d["J_eq"],
            eq_curve=d["eq_curve"],
            label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS - 0.5)
    add_panel_label(ax_c, "c")

    # ── (d) rate-factor profile ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    panel_d(ax_d, d["x"], d["access"], d["u"], d["arrh"],
            d["xi_peak"], d["xi_LCST"],
            label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS - 0.5)
    add_panel_label(ax_d, "d")

    # ── (e) heat source / reactant penetration ────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    panel_e(ax_e, d["x"], d["u_mean"], d["heat_mean"],
            d["xi_peak"], d["xi_LCST"],
            label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS)
    add_panel_label(ax_e, "e")

    # ── (f) ξ_peak/ξ_LCST under scanned parameters ────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    scans = load_scans()
    panel_f_scan(ax_f, scans,
                 label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS - 0.5)
    add_panel_label(ax_f, "f")

    out_dir = Path(__file__).resolve().parent.parent / "Figure" / "pub"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig3_zones.pdf"
    png = out_dir / "fig3_zones.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
