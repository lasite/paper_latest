"""
make_fig1d.py — Panel (d): Da–Bi_T stability classification map.

Cache: data/fig1/panel_d.npz
Render: Figure/fig1/panel_d.{pdf,png}
"""
import os
import sys
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import DATA_DIR, load_or_compute_DaBiT_map
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_HOPF, C_MONO, C_STABLE, C_NO_SS,
)

CACHE = os.path.join(DATA_DIR, "panel_d.npz")


def compute():
    raw = load_or_compute_DaBiT_map(n=60)
    np.savez(CACHE, Da=raw['Da'], BiT=raw['BiT'], cls=raw['cls'])
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    Da = d['Da']; BiT = d['BiT']; cls = d['cls']
    cmap = ListedColormap([C_NO_SS, C_STABLE, C_HOPF, C_MONO])
    norm = BoundaryNorm([-2.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    Dam, BiTm = np.meshgrid(Da, BiT)
    ax.pcolormesh(Dam, BiTm, cls, cmap=cmap, norm=norm,
                  shading='nearest', rasterized=True)
    ax.plot(4.0, 0.10, 'w*', ms=8, zorder=5,
            markeredgecolor='k', markeredgewidth=0.5)

    ax.set_xlabel(r'$\mathrm{Da}$')
    ax.set_ylabel(r'$\mathrm{Bi}_T$')

    handles = [
        Patch(facecolor=C_HOPF, edgecolor='0.3', lw=0.3, label='Hopf'),
        Patch(facecolor=C_MONO, edgecolor='0.3', lw=0.3, label='Monotone'),
        Patch(facecolor=C_STABLE, edgecolor='0.3', lw=0.3, label='Stable'),
        Patch(facecolor=C_NO_SS, edgecolor='0.3', lw=0.3, label='No SS'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9, handlelength=1.0)
    add_panel_label(ax, 'd')
    save_panel(fig, 'panel_d')


if __name__ == '__main__':
    render()
