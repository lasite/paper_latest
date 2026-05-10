"""
make_fig1f.py — Panel (f): period T = 2π/ω and Re(σ_1) vs Da.

Cache: data/fig1/panel_f.npz
Render: Figure/fig1/panel_f.{pdf,png}
"""
import os
import sys
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import DATA_DIR, load_or_compute_Da_period
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_MONO, C_PERIOD,
)

CACHE = os.path.join(DATA_DIR, "panel_f.npz")


def compute():
    raw = load_or_compute_Da_period(n=100)
    np.savez(CACHE, Da=raw['Da'], period=raw['period'], re=raw['re'])
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    Da = d['Da']; T = d['period']; re = d['re']
    ax.plot(Da, T, '-', color=C_PERIOD, lw=1.4)
    ax.set_xlabel(r'$\mathrm{Da}$')
    ax.set_ylabel(r'$T_{\rm LSA}=2\pi/\omega$ $(\tau)$', color=C_PERIOD)
    ax.tick_params(axis='y', colors=C_PERIOD)
    ax.set_xlim(1, 20)
    if np.any(np.isfinite(T)):
        T_fin = T[np.isfinite(T)]
        ax.set_ylim(max(0, T_fin.min() - 1), T_fin.max() + 1)

    ax2 = ax.twinx()
    ax2.plot(Da, re, '--', color=C_MONO, lw=0.9)
    ax2.set_ylabel(r'$\mathrm{Re}(\sigma_1)$', color=C_MONO)
    ax2.tick_params(axis='y', colors=C_MONO)
    ax2.axhline(0, color='k', lw=0.3)

    ax.axvline(4.0, color='k', lw=0.5, ls=':', alpha=0.5)

    handles = [
        Line2D([0], [0], color=C_PERIOD, lw=1.4, label=r'Period $T$'),
        Line2D([0], [0], color=C_MONO, lw=0.9, ls='--', label=r'$\mathrm{Re}(\sigma_1)$'),
    ]
    ax.legend(handles=handles, loc='center right', framealpha=0.9)
    add_panel_label(ax, 'f')
    save_panel(fig, 'panel_f')


if __name__ == '__main__':
    render()
