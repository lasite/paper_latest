#!/usr/bin/env python3
"""
make_fig2f_u_kymo.py — Fig 2(f): u(ξ,τ) kymograph on log scale (viridis).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from style_pub import set_style, add_panel_label
from fig2_data import load_cache, time_window, save_panel, T_START, T_END
set_style()


U_FLOOR = 1e-6


def panel_f(ax, u_window, label_fs=9, tick_fs=7):
    """Draw the u(ξ,τ) kymograph on ax (log scale). Returns imshow handle."""
    im = ax.imshow(u_window, origin='lower', aspect='auto',
                   extent=[T_START, T_END, 0, 1],
                   cmap='viridis',
                   norm=LogNorm(vmin=U_FLOOR, vmax=1.0),
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
    u = d["u"]

    idx = time_window(t)
    u_window = np.clip(u[:, idx], U_FLOOR, 1.0)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.94, bottom=0.18)

    im = panel_f(ax, u_window)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'$u$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    add_panel_label(ax, 'f')

    save_panel(fig, "fig2f_u_kymo")
    plt.close(fig)


if __name__ == "__main__":
    main()
