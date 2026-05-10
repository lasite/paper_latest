#!/usr/bin/env python3
"""
make_fig2e_theta_kymo.py — Fig 2(e): θ(ξ,τ) kymograph (inferno, fixed range).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_pub import set_style, add_panel_label
from fig2_data import load_cache, time_window, save_panel, T_START, T_END
set_style()


def panel_e(ax, th_window, label_fs=9, tick_fs=7):
    """Draw the θ(ξ,τ) kymograph on ax. Returns the imshow handle."""
    th_vmin = max(0.0, np.floor(th_window.min() * 10) / 10)
    th_vmax = np.ceil(th_window.max() * 10) / 10
    im = ax.imshow(th_window, origin='lower', aspect='auto',
                   extent=[T_START, T_END, 0, 1],
                   cmap='inferno', vmin=th_vmin, vmax=th_vmax,
                   rasterized=False)
    ax.set_xlabel(r'$\tau$', fontsize=label_fs)
    ax.set_ylabel(r'$\xi$', fontsize=label_fs)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=tick_fs, direction='out', length=2.5)
    return im


def main():
    d = load_cache()
    t = d["t"]
    theta = d["theta"]

    idx = time_window(t)
    th_window = theta[:, idx]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.94, bottom=0.18)

    im = panel_e(ax, th_window)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'$\theta$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    add_panel_label(ax, 'e')

    save_panel(fig, "fig2e_theta_kymo")
    plt.close(fig)


if __name__ == "__main__":
    main()
