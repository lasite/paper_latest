#!/usr/bin/env python3
"""
make_fig2c_phase.py — Fig 2(c): single-cycle (J,θ) phase portrait.

Both the surface trajectory and the magnified center inset are drawn as
LineCollection segments colored by the local u value on a shared linear
Normalize(0, 1) colorbar (the physical bath fraction). One full limit
cycle is selected peak-to-peak from the cached n_save grid; direction-
of-flow arrows are added uniformly along the arc length.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks

from style_pub import set_style, add_panel_label
from fig2_data import load_cache, time_window, save_panel
set_style()


U_NORM = Normalize(vmin=0.0, vmax=1.0)


def one_cycle(ts, Ts, *arrays):
    """Return (ts, *arrays) restricted to one full limit cycle (peak-to-peak)."""
    peaks, _ = find_peaks(Ts, prominence=0.5, distance=30)
    if len(peaks) < 2:
        return (ts,) + tuple(arrays)
    i0, i1 = peaks[0], peaks[1] + 1
    return (ts[i0:i1],) + tuple(a[i0:i1] for a in arrays)


def colored_path(ax, x, y, c, **kw):
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='viridis', norm=U_NORM, **kw)
    lc.set_array(np.clip(c[:-1], 0.0, 1.0))
    ax.add_collection(lc)
    return lc


def add_direction_arrows(ax, x, y, n=4, scale=11, lw=1.0, arrow_frac=0.025):
    """Place n direction-of-flow arrows uniformly along the trajectory arc.

    Each arrow has a fixed visual length set by arrow_frac (fraction of the
    total arc length, scaled by axis aspect). This avoids both clustering in
    slow phases and the variable-length appearance from index-based arrows.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax_aspect = (y1 - y0) / max(x1 - x0, 1e-12)
    dx = np.diff(x)
    dy = np.diff(y) / max(ax_aspect, 1e-12)
    seg = np.sqrt(dx ** 2 + dy ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    half_arrow = arrow_frac * total / 2.0
    targets = np.linspace(total / (2 * n), total * (1 - 1 / (2 * n)), n)
    for tgt in targets:
        i_start = int(np.searchsorted(arc, max(0.0,        tgt - half_arrow)))
        i_end   = int(np.searchsorted(arc, min(total - 1e-9, tgt + half_arrow)))
        i_start = max(0, min(i_start, len(x) - 2))
        i_end   = max(i_start + 1, min(i_end, len(x) - 1))
        ax.annotate('', xy=(x[i_end], y[i_end]),
                    xytext=(x[i_start], y[i_start]),
                    arrowprops=dict(arrowstyle='-|>', color='k',
                                    lw=lw, mutation_scale=scale,
                                    alpha=0.95),
                    zorder=7)


def panel_c(ax_c, ts, surf_J, surf_T, surf_U, ctr_J, ctr_T, ctr_U,
            label_fs=9, tick_fs=7, inset_title_fs=7.5,
            inset_tick_fs=6.5, inset_label_fs=7):
    """Draw the (J,θ) phase portrait + center inset on ax_c.

    Returns the LineCollection (lc_main) for the caller to attach a colorbar.
    """
    # Surface trajectory
    lc_main = colored_path(ax_c, surf_J, surf_T, surf_U,
                           linewidth=1.5, alpha=0.95)

    j_lo = max(0.0, surf_J.min() - 0.05)
    j_hi = surf_J.max() + 0.10
    t_lo = max(0.0, surf_T.min() - 0.20)
    t_hi = surf_T.max() + 0.20
    ax_c.set_xlim(j_lo, j_hi)
    ax_c.set_ylim(t_lo, t_hi)

    add_direction_arrows(ax_c, surf_J, surf_T, n=4,
                         scale=11, lw=1.0, arrow_frac=0.025)

    ax_c.set_xlabel(r'$J$', fontsize=label_fs)
    ax_c.set_ylabel(r'$\theta$', fontsize=label_fs)
    ax_c.tick_params(labelsize=tick_fs)

    # Magnified center inset (same colormap/norm as surface).
    ax_ci = ax_c.inset_axes([0.30, 0.30, 0.42, 0.36])
    colored_path(ax_ci, ctr_J, ctr_T, ctr_U, linewidth=1.2, alpha=0.95)

    jc_pad = max(0.005, 0.10 * (ctr_J.max() - ctr_J.min()))
    tc_pad = max(0.005, 0.10 * (ctr_T.max() - ctr_T.min()))
    ax_ci.set_xlim(ctr_J.min() - jc_pad, ctr_J.max() + jc_pad)
    ax_ci.set_ylim(ctr_T.min() - tc_pad, ctr_T.max() + tc_pad)
    ax_ci.xaxis.set_major_locator(MaxNLocator(3))
    ax_ci.yaxis.set_major_locator(MaxNLocator(3))

    ax_ci.set_title(r'center ($\xi=0$)', fontsize=inset_title_fs,
                    pad=2, color='#333')
    ax_ci.tick_params(labelsize=inset_tick_fs, length=2, pad=1)
    ax_ci.set_xlabel(r'$J$', fontsize=inset_label_fs, labelpad=1)
    ax_ci.set_ylabel(r'$\theta$', fontsize=inset_label_fs, labelpad=1)
    for spine in ax_ci.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color('#444')
    ax_ci.set_facecolor('#f0f0f0')
    return lc_main


def main():
    d = load_cache()
    t = d["t"]
    J, u, theta = d["J"], d["u"], d["theta"]

    idx = time_window(t)
    ts = t[idx]
    surf_J = J[-1, idx]
    surf_T = theta[-1, idx]
    surf_U = u[-1, idx]
    ctr_J  = J[0, idx]
    ctr_T  = theta[0, idx]
    ctr_U  = u[0, idx]

    ts_s, surf_J, surf_T, surf_U = one_cycle(ts, surf_T, surf_J, surf_T, surf_U)
    ts_c, ctr_J,  ctr_T,  ctr_U  = one_cycle(ts, ctr_T,  ctr_J,  ctr_T,  ctr_U)

    fig, ax_c = plt.subplots(figsize=(3.4, 3.0))
    fig.subplots_adjust(left=0.16, right=0.84, top=0.95, bottom=0.16)

    lc_main = panel_c(ax_c, ts_s, surf_J, surf_T, surf_U,
                      ctr_J, ctr_T, ctr_U)
    cb = plt.colorbar(lc_main, ax=ax_c, pad=0.02, fraction=0.05)
    cb.set_label(r'$u$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    add_panel_label(ax_c, 'c')

    save_panel(fig, "fig2c_phase")
    plt.close(fig)


if __name__ == "__main__":
    main()
