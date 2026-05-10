"""
make_fig1b.py — Panel (b): leading eigenvalue Re/Im vs Bi_T (lower branch).

Cache: data/fig1/panel_b.npz
Render: Figure/fig1/panel_b.{pdf,png}
"""
import os
import sys
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import DATA_DIR, load_or_compute_BiT_branches
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_HOPF, C_MONO,
)

CACHE = os.path.join(DATA_DIR, "panel_b.npz")
C_IM = '#9467bd'


def compute():
    raw = load_or_compute_BiT_branches()
    BiT = raw['BiT']
    J = raw['J_lower']
    ev = raw['evals_lower']
    mask = ~np.isnan(J)
    bt_c = BiT[mask]
    ev_c = ev[mask]
    re_lead = np.array([max(e.real) for e in ev_c])
    im_lead = np.array([abs(e[np.argmax(e.real)].imag) for e in ev_c])
    np.savez(CACHE, BiT=bt_c, re=re_lead, im=im_lead)
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    bt = d['BiT']; re = d['re']; im = d['im']

    ax.plot(bt, re, '-', color=C_MONO, lw=1.4)
    ax.axhline(0, color='k', lw=0.4)

    ax2 = ax.twinx()
    ax2.plot(bt, im, '--', color=C_IM, lw=1.0)
    ax2.set_ylabel(r'$|\mathrm{Im}(\sigma_1)|$', color=C_IM,
                   rotation=270, labelpad=10)
    ax2.tick_params(axis='y', colors=C_IM)

    hopf_mask = (re > 0) & (im > 0.01)
    mono_mask = (re > 0) & (im < 0.01)
    if np.any(hopf_mask):
        ax.axvspan(bt[hopf_mask].min(), bt[hopf_mask].max(),
                   alpha=0.10, color=C_HOPF, zorder=0)
    if np.any(mono_mask):
        ax.axvspan(bt[mono_mask].min(), bt[mono_mask].max(),
                   alpha=0.10, color=C_MONO, zorder=0)

    y_top = ax.get_ylim()[1] * 0.88
    if np.any(mono_mask):
        ax.text(np.mean(bt[mono_mask]), y_top, 'mono.',
                fontsize=5.5, ha='center', color=C_MONO, weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    if np.any(hopf_mask):
        bh = bt[hopf_mask]
        x_h = bh.min() + 0.40 * (bh.max() - bh.min())
        ax.text(x_h, ax.get_ylim()[1] * 0.78, 'Hopf',
                fontsize=5.5, ha='center', color='#1f8f1f', weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    ax.axvline(0.10, color='k', lw=0.7, ls=':', alpha=0.6)
    ax.text(0.105, ax.get_ylim()[1] * 0.05, r'$\bigstar$', fontsize=8, va='bottom')

    ax.set_xlabel(r'$\mathrm{Bi}_T$')
    ax.set_ylabel(r'$\mathrm{Re}(\sigma_1)$')
    ax.set_xlim(0.02, 0.30)

    handles = [
        Line2D([0], [0], color=C_MONO, lw=1.4, label=r'$\mathrm{Re}(\sigma_1)$'),
        Line2D([0], [0], color=C_IM, lw=1.0, ls='--', label=r'$|\mathrm{Im}(\sigma_1)|$'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'b')
    save_panel(fig, 'panel_b')


if __name__ == '__main__':
    render()
