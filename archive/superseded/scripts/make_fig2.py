#!/usr/bin/env python3
"""
make_fig2.py — Composite Fig 2: oscillation mechanism (2×3 panels).

Layout mirrors make_fig1_stability.py: a single figure built with
GridSpec(2, 3), each panel imported from the per-panel modules so
individual previews and the composite stay in sync.

  Top:    (a) schematic   (b) J/θ/u timeseries   (c) (J,θ) phase portrait
  Bottom: (d) J(ξ,τ) kymo (e) θ(ξ,τ) kymo        (f) u(ξ,τ) kymo
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import load_cache, time_window

from make_fig2a_schematic import draw_schematic
from make_fig2b_timeseries import panel_b
from make_fig2c_phase       import panel_c, one_cycle
from make_fig2d_J_kymo      import panel_d
from make_fig2e_theta_kymo  import panel_e
from make_fig2f_u_kymo      import panel_f, U_FLOOR

set_style()


# Panel-content font sizes for the composite (smaller than per-panel previews
# because each cell is ~2 in wide instead of 3.4 in).
LBL_FS    = 8
TICK_FS   = 6.5
LEG_FS    = 6
CB_LBL_FS = 8
CB_TCK_FS = 6


def main():
    d = load_cache()
    t = d["t"]
    J, u, theta = d["J"], d["u"], d["theta"]

    idx = time_window(t)
    ts = t[idx]
    J_surf, J_ctr   = J[-1, idx],     J[0, idx]
    th_surf, th_ctr = theta[-1, idx], theta[0, idx]
    u_surf, u_ctr   = u[-1, idx],     u[0, idx]

    # Single cycle for panel (c)
    surf_J, surf_T, surf_U = J_surf.copy(), th_surf.copy(), u_surf.copy()
    ctr_J,  ctr_T,  ctr_U  = J_ctr.copy(),  th_ctr.copy(),  u_ctr.copy()
    _, surf_J, surf_T, surf_U = one_cycle(ts, surf_T, surf_J, surf_T, surf_U)
    _, ctr_J,  ctr_T,  ctr_U  = one_cycle(ts, ctr_T,  ctr_J,  ctr_T,  ctr_U)

    # Kymograph windows
    J_kymo  = J[:, idx]
    th_kymo = theta[:, idx]
    u_kymo  = np.clip(u[:, idx], U_FLOOR, 1.0)

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(PRE_DOUBLE, 5.8))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1.25, 1.0],
                           hspace=0.45, wspace=0.45,
                           left=0.07, right=0.96, top=0.94, bottom=0.09)

    # ── (a) schematic ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    # Anchor to the north so the equal-aspect schematic hugs the top
    # of the cell — keeps the (a) label visually next to the content.
    ax_a.set_anchor('N')
    draw_schematic(ax_a, fontsize_scale=0.85)
    add_panel_label(ax_a, 'a')

    # ── (b) J/θ/u stacked timeseries ───────────────────────────────
    sub_b = gs[0, 1].subgridspec(3, 1, hspace=0.12)
    ax_bJ  = fig.add_subplot(sub_b[0])
    ax_bth = fig.add_subplot(sub_b[1], sharex=ax_bJ)
    ax_bu  = fig.add_subplot(sub_b[2], sharex=ax_bJ)
    panel_b(ax_bJ, ax_bth, ax_bu, ts, J_surf, J_ctr, th_surf, th_ctr,
            u_surf, u_ctr,
            label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS)
    # Anchor the legend to the upper-right just above ax_bJ, so the
    # (b) label at upper-left has room and the legend doesn't cover data.
    leg = ax_bJ.get_legend()
    if leg is not None:
        leg.remove()
    ax_bJ.legend(fontsize=5.5, loc='lower right', ncol=2,
                 bbox_to_anchor=(1.0, 1.00),
                 handlelength=1.0, columnspacing=0.6,
                 handletextpad=0.4,
                 framealpha=0.85, borderpad=0.2, labelspacing=0.2)
    add_panel_label(ax_bJ, 'b')

    # ── (c) (J,θ) phase portrait ───────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    lc_c = panel_c(ax_c, ts, surf_J, surf_T, surf_U, ctr_J, ctr_T, ctr_U,
                   label_fs=LBL_FS, tick_fs=TICK_FS,
                   inset_title_fs=6.5, inset_tick_fs=5.5,
                   inset_label_fs=6)
    cb_c = plt.colorbar(lc_c, ax=ax_c, pad=0.02, fraction=0.05)
    cb_c.set_label(r'$u$', fontsize=CB_LBL_FS)
    cb_c.ax.tick_params(labelsize=CB_TCK_FS)
    add_panel_label(ax_c, 'c')

    # ── (d) J kymograph ────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    im_d = panel_d(ax_d, J_kymo, label_fs=LBL_FS, tick_fs=TICK_FS)
    cb_d = plt.colorbar(im_d, ax=ax_d, pad=0.02, fraction=0.05)
    cb_d.set_label(r'$J$', fontsize=CB_LBL_FS)
    cb_d.ax.tick_params(labelsize=CB_TCK_FS)
    add_panel_label(ax_d, 'd')

    # ── (e) θ kymograph ────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    im_e = panel_e(ax_e, th_kymo, label_fs=LBL_FS, tick_fs=TICK_FS)
    cb_e = plt.colorbar(im_e, ax=ax_e, pad=0.02, fraction=0.05)
    cb_e.set_label(r'$\theta$', fontsize=CB_LBL_FS)
    cb_e.ax.tick_params(labelsize=CB_TCK_FS)
    add_panel_label(ax_e, 'e')

    # ── (f) u kymograph ────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    im_f = panel_f(ax_f, u_kymo, label_fs=LBL_FS, tick_fs=TICK_FS)
    cb_f = plt.colorbar(im_f, ax=ax_f, pad=0.02, fraction=0.05)
    cb_f.set_label(r'$u$', fontsize=CB_LBL_FS)
    cb_f.ax.tick_params(labelsize=CB_TCK_FS)
    add_panel_label(ax_f, 'f')

    out_dir = Path(__file__).resolve().parent.parent / "Figure" / "pub"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig2_mechanism.pdf"
    png = out_dir / "fig2_mechanism.png"
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
