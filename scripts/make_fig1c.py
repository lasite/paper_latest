"""
make_fig1c.py — Panel (c): dispersion relation σ(k) at 3 representative Bi_T.

Cache: data/fig1/panel_c.npz
Render: Figure/fig1/panel_c.{pdf,png}
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import DATA_DIR, load_or_compute_dispersion
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_HOPF, C_MONO, C_STABLE,
)

CACHE = os.path.join(DATA_DIR, "panel_c.npz")
BIT_DISP = (0.04, 0.10, 0.40)


def compute():
    raw = load_or_compute_dispersion(BiT_list=BIT_DISP)
    k = raw['k']
    sig = raw['sigmas']
    re_max_per_BiT = np.array([
        np.array([max(s.real) for s in sig[i]]) for i in range(len(BIT_DISP))
    ])
    np.savez(CACHE, k=k, BiT=raw['BiT'], re_max=re_max_per_BiT,
             J0=raw['J0'], theta0=raw['theta0'],
             sig0_re=np.array([sig[i, 0, 0].real for i in range(len(BIT_DISP))]))
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    k = d['k']
    BiT = d['BiT']
    re_max = d['re_max']
    sig0_re = d['sig0_re']

    colors = {0.04: C_MONO, 0.10: C_HOPF, 0.40: C_STABLE}
    ls = {0.04: '--', 0.10: '-', 0.40: '-.'}
    labels = {0.04: r'$\mathrm{Bi}_T\!=\!0.04$',
              0.10: r'$\mathrm{Bi}_T\!=\!0.10$',
              0.40: r'$\mathrm{Bi}_T\!=\!0.40$'}

    for i, BiTv in enumerate(BiT):
        BiTv = float(BiTv)
        ax.plot(k, re_max[i], ls.get(BiTv, '-'),
                color=colors.get(BiTv, 'k'), lw=1.0, label=labels.get(BiTv, f'{BiTv:.2f}'))

    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'max Re$(\sigma)$')
    ax.set_xlim(0, 150)
    # Legend below panel-label, narrow column on the upper-right
    ax.legend(loc='upper right', framealpha=0.9, handlelength=1.5,
              borderpad=0.3, labelspacing=0.3)

    # k→0 inset — placed lower-right, away from legend (upper-left)
    ax_ins = ax.inset_axes([0.50, 0.08, 0.46, 0.40])
    n_zoom = 60
    for i, BiTv in enumerate(BiT):
        BiTv = float(BiTv)
        ax_ins.plot(k[:n_zoom], re_max[i, :n_zoom], ls.get(BiTv, '-'),
                    color=colors.get(BiTv, 'k'), lw=0.7)
    ax_ins.axhline(0, color='k', lw=0.3)
    ax_ins.set_xlim(0, k[n_zoom - 1])
    ax_ins.set_ylim(-2, 10)
    ax_ins.set_xlabel('$k$', fontsize=5, labelpad=0)
    ax_ins.set_ylabel(r'Re$(\sigma)$', fontsize=5, labelpad=0)
    ax_ins.tick_params(labelsize=5)
    ax_ins.set_title(r'$k\!\to\!0$', fontsize=5, pad=2)
    if 0.10 in [float(b) for b in BiT]:
        idx = int(np.where(np.isclose(BiT, 0.10))[0][0])
        ax_ins.plot(0, float(sig0_re[idx]), 'o', color=C_HOPF, ms=3, zorder=5)

    add_panel_label(ax, 'c')
    save_panel(fig, 'panel_c')


if __name__ == '__main__':
    render()
