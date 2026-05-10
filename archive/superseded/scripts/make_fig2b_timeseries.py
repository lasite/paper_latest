#!/usr/bin/env python3
"""
make_fig2b_timeseries.py — Fig 2(b): stacked J/θ/u time series at surface and
center; bottom panel uses log y-axis to expose the deep core depletion.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from style_pub import set_style, add_panel_label
from fig2_data import load_cache, time_window, save_panel
set_style()


def panel_b(ax_J, ax_th, ax_u, ts, J_surf, J_ctr, th_surf, th_ctr,
            u_surf, u_ctr, label_fs=9, tick_fs=7, legend_fs=6.5):
    """Draw the J/θ/u stacked time series on three pre-created axes."""
    # J subplot
    ax_J.plot(ts, J_surf, '-',  color='#1f77b4', lw=1.0,
              label=r'surface ($\xi=1$)')
    ax_J.plot(ts, J_ctr,  '--', color='#1f77b4', lw=0.9,
              label=r'center ($\xi=0$)')
    ax_J.set_ylabel(r'$J$', fontsize=label_fs)
    ax_J.tick_params(labelbottom=False, labelsize=tick_fs)
    ax_J.legend(fontsize=legend_fs, loc='lower center', ncol=2,
                bbox_to_anchor=(0.5, 1.00),
                handlelength=1.6, columnspacing=0.8, framealpha=0.9)

    # θ subplot
    ax_th.plot(ts, th_surf, '-',  color='#d62728', lw=1.0)
    ax_th.plot(ts, th_ctr,  '--', color='#d62728', lw=0.9)
    ax_th.set_ylabel(r'$\theta$', fontsize=label_fs)
    ax_th.tick_params(labelbottom=False, labelsize=tick_fs)

    # u subplot — linear scale (surface dynamics; center stays ≈ 0)
    ax_u.plot(ts, u_surf, '-',  color='#2ca02c', lw=1.0)
    ax_u.plot(ts, u_ctr,  '--', color='#1a7a1a', lw=1.1,
              dashes=(3, 1.5))
    ax_u.set_ylim(-0.05, 1.10)
    ax_u.set_yticks([0.0, 0.5, 1.0])
    ax_u.set_ylabel(r'$u$', fontsize=label_fs)
    ax_u.set_xlabel(r'$\tau$', fontsize=label_fs)
    ax_u.tick_params(labelsize=tick_fs)

    for ax in (ax_J, ax_th):
        ax.grid(True, which='major', axis='y', alpha=0.25, lw=0.4)
    ax_u.grid(True, which='major', axis='y', alpha=0.30, lw=0.4)

    for ax in (ax_J, ax_th, ax_u):
        ax.yaxis.set_label_coords(-0.13, 0.5)


def main():
    d = load_cache()
    t = d["t"]
    J, u, theta = d["J"], d["u"], d["theta"]

    idx = time_window(t)
    ts = t[idx]
    J_surf, J_ctr   = J[-1, idx],     J[0, idx]
    th_surf, th_ctr = theta[-1, idx], theta[0, idx]
    u_surf, u_ctr   = u[-1, idx],     u[0, idx]

    fig = plt.figure(figsize=(3.4, 3.6))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.10,
                           left=0.16, right=0.97, top=0.90, bottom=0.13)
    ax_J  = fig.add_subplot(gs[0])
    ax_th = fig.add_subplot(gs[1], sharex=ax_J)
    ax_u  = fig.add_subplot(gs[2], sharex=ax_J)

    panel_b(ax_J, ax_th, ax_u, ts, J_surf, J_ctr, th_surf, th_ctr,
            u_surf, u_ctr)
    add_panel_label(ax_J, 'b')

    save_panel(fig, "fig2b_timeseries")
    plt.close(fig)


if __name__ == "__main__":
    main()
