#!/usr/bin/env python3
"""
make_fig2d_J_kymo.py — Fig 2(d): J(ξ,τ) kymograph with TwoSlopeNorm centered
at J=1 so blue=collapsed / white=reference / red=swollen.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from style_pub import set_style, add_panel_label
from fig2_data import load_cache, time_window, save_panel, T_START, T_END
set_style()


def panel_d(ax, J_window, label_fs=9, tick_fs=7):
    """Draw the J(ξ,τ) kymograph on ax. Returns the imshow handle."""
    j_vmin = max(0.15, np.floor(J_window.min() * 10) / 10)
    j_vmax = max(1.05, np.ceil(J_window.max()  * 10) / 10)
    j_norm = TwoSlopeNorm(vcenter=1.0,
                          vmin=min(j_vmin, 0.99),
                          vmax=max(j_vmax, 1.01))
    im = ax.imshow(J_window, origin='lower', aspect='auto',
                   extent=[T_START, T_END, 0, 1],
                   cmap='RdBu_r', norm=j_norm, rasterized=False)
    ax.set_xlabel(r'$\tau$', fontsize=label_fs)
    ax.set_ylabel(r'$\xi$', fontsize=label_fs)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=tick_fs, direction='out', length=2.5)
    return im


def main():
    d = load_cache()
    t = d["t"]
    J = d["J"]

    idx = time_window(t)
    J_window = J[:, idx]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.94, bottom=0.18)

    im = panel_d(ax, J_window)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'$J$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    add_panel_label(ax, 'd')

    save_panel(fig, "fig2d_J_kymo")
    plt.close(fig)


if __name__ == "__main__":
    main()
