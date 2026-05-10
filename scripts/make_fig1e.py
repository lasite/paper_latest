"""
make_fig1e.py — Panel (e): S_chi–Bi_T stability classification map.

Cache: data/fig1/panel_e.npz
Render: Figure/fig1/panel_e.{pdf,png}
"""
import os
import sys
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import DATA_DIR, load_or_compute_ScBiT_map
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_HOPF, C_MONO, C_STABLE, C_NO_SS,
)

CACHE = os.path.join(DATA_DIR, "panel_e.npz")


def compute():
    raw = load_or_compute_ScBiT_map(n=60)
    np.savez(CACHE, S_chi=raw['S_chi'], BiT=raw['BiT'], cls=raw['cls'])
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    Sc = d['S_chi']; BiT = d['BiT']; cls = d['cls']
    cmap = ListedColormap([C_NO_SS, C_STABLE, C_HOPF, C_MONO])
    norm = BoundaryNorm([-2.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    Sm, BiTm = np.meshgrid(Sc, BiT)
    ax.pcolormesh(Sm, BiTm, cls, cmap=cmap, norm=norm,
                  shading='nearest', rasterized=True)
    ax.plot(1.0, 0.10, 'w*', ms=8, zorder=5,
            markeredgecolor='k', markeredgewidth=0.5)

    ax.set_xlabel(r'$S_\chi$')
    ax.set_ylabel(r'$\mathrm{Bi}_T$')
    add_panel_label(ax, 'e')
    save_panel(fig, 'panel_e')


if __name__ == '__main__':
    render()
