"""
make_fig1a.py — Panel (a): bifurcation diagram J0(Bi_T) with 3 branches.

Cache: data/fig1/panel_a.npz
Render: Figure/fig1/panel_a.{pdf,png}
"""
import os
import sys
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig1_data import (
    DATA_DIR, classify_point, load_or_compute_BiT_branches,
)
from fig1_style import (
    set_style, new_panel_fig, save_panel, add_panel_label,
    C_HOPF, C_MONO, C_STABLE, C_SADDLE,
)

CACHE = os.path.join(DATA_DIR, "panel_a.npz")


def compute():
    raw = load_or_compute_BiT_branches()
    # working point reference
    BiT = raw['BiT']
    wp_idx = int(np.argmin(np.abs(BiT - 0.10)))
    np.savez(CACHE,
        BiT=BiT,
        J_lower=raw['J_lower'], evals_lower=raw['evals_lower'],
        J_middle=raw['J_middle'], evals_middle=raw['evals_middle'],
        J_upper=raw['J_upper'], evals_upper=raw['evals_upper'],
        wp_BiT=np.array(0.10), wp_J=raw['J_lower'][wp_idx])
    print(f"  Saved cache: {CACHE}")


def _plot_branch_segments(ax, xvals, J, evals, color_override=None,
                          label=None, ls='-', zorder=2):
    runs = []
    i = 0
    while i < len(xvals):
        if np.isnan(J[i]):
            i += 1
            continue
        cl = classify_point(evals[i])
        j = i + 1
        while j < len(xvals) and (not np.isnan(J[j])) and classify_point(evals[j]) == cl:
            j += 1
        runs.append((i, j, cl))
        i = j
    first = True
    for (i0, i1, cl) in runs:
        lo = max(i0 - 1, 0)
        hi = min(i1, len(xvals))
        mask = ~np.isnan(J[lo:hi])
        xx = xvals[lo:hi][mask]
        yy = J[lo:hi][mask]
        if len(xx) < 2:
            continue
        color = color_override or (
            C_HOPF if cl == 'hopf' else (C_MONO if cl == 'mono' else C_STABLE))
        ax.plot(xx, yy, ls, color=color, lw=1.6, zorder=zorder,
                label=label if first else None)
        first = False


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig()

    BiT = d['BiT']
    _plot_branch_segments(ax, BiT, d['J_lower'], d['evals_lower'])
    _plot_branch_segments(ax, BiT, d['J_upper'], d['evals_upper'])
    _plot_branch_segments(ax, BiT, d['J_middle'], d['evals_middle'],
                          color_override=C_SADDLE, ls='--', zorder=3)

    ax.annotate('fold', xy=(0.28, 0.70), fontsize=6, color='0.3',
                ha='center', style='italic', weight='bold')
    ax.plot(float(d['wp_BiT']), float(d['wp_J']), 'k*', ms=8, zorder=5)

    ax.set_xlabel(r'$\mathrm{Bi}_T$')
    ax.set_ylabel(r'$J_0$')
    ax.set_xlim(0, 0.50)
    ax.set_ylim(0, 1.45)

    handles = [
        Line2D([0], [0], color=C_HOPF, lw=1.6, label='Hopf'),
        Line2D([0], [0], color=C_MONO, lw=1.6, label='Monotone'),
        Line2D([0], [0], color=C_STABLE, lw=1.6, label='Stable'),
        Line2D([0], [0], color=C_SADDLE, lw=1.6, ls='--', label='Saddle'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9, handlelength=1.2)
    add_panel_label(ax, 'a')
    save_panel(fig, 'panel_a')


if __name__ == '__main__':
    render()
