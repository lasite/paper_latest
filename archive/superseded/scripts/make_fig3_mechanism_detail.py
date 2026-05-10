#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig3_mechanism_detail.py — Detailed spatial evidence for the 1D slab mechanism.

Layout: double-column, 3 rows.
  (a) J surface/center time series
  (b) theta surface/center time series
  (c) u surface/center time series
  (d) J kymograph
  (e) accessibility kymograph
  (f) u(x) profiles at four representative phases
  (g) accessibility profiles at the same phases
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from style_pub import set_style, PRE_DOUBLE, C, add_panel_label, save
from scan_optimized import Params, simulate

set_style()

_CACHE = os.path.join(os.path.dirname(__file__), "figures_pub/fig3_mechanism_detail_cache.npz")
_CACHE_VERSION = 2

PHASE_COLORS = ["#4a90e2", "#a23db7", "#f39c12", "#7e57c2"]
PHASE_LABELS = ["(1) swollen", "(2) ignition", "(3) collapse", "(4) cooling"]


def run_case():
    from fig2_data import WORKING_POINT
    # Use the canonical working point (N=301 spatial resolution).
    # Override t_end → 120 for the shorter mechanism cycle window;
    # n_save reduced from 10000 → 4500 for figure-time density.
    p_dict = dict(WORKING_POINT)
    p_dict["t_end"]  = 120.0
    p_dict["n_save"] = 4500
    p = Params(**p_dict)
    print("  Running detailed mechanism case ...")
    data = simulate(p)
    print(f"  Done. nfev={data['nfev']}")
    return p, data


def select_window_and_phases(t, J_surf):
    i0 = int(0.35 * len(t))
    tail = J_surf[i0:]
    prominence = max(0.02, 0.10 * np.ptp(tail))
    peaks, _ = find_peaks(tail, prominence=prominence, distance=35)

    if len(peaks) >= 3:
        peaks = peaks[-3:] + i0
        start = peaks[0]
        stop = peaks[2]
        cycle_start = peaks[0]
        cycle_stop = peaks[1]
    else:
        start = i0
        stop = len(t) - 1
        cycle_start = start
        cycle_stop = start + max(8, (stop - start) // 2)

    phase_frac = np.array([0.04, 0.28, 0.42, 0.74])
    phase_idx = cycle_start + np.rint(phase_frac * (cycle_stop - cycle_start)).astype(int)
    phase_idx = np.clip(phase_idx, start, stop - 1)
    return start, stop, phase_idx


def build_cache():
    p, data = run_case()
    t = np.asarray(data["t"])
    x = np.asarray(data["x"])
    J = np.asarray(data["J"])
    u = np.asarray(data["u"])
    theta = np.asarray(data["theta"])
    phi = np.asarray(data["phi"])
    access = np.maximum(1.0 - phi, 1.0e-12) ** p.m_act

    start, stop, phase_idx = select_window_and_phases(t, J[-1])
    sl = slice(start, stop)

    cache = {
        "cache_version": np.array([_CACHE_VERSION], dtype=int),
        "t": t,
        "x": x,
        "J": J,
        "u": u,
        "theta": theta,
        "phi": phi,
        "access": access,
        "window_start": np.array([start], dtype=int),
        "window_stop": np.array([stop], dtype=int),
        "phase_idx": phase_idx.astype(int),
        "t_plot": t[sl] - t[start],
        "J_surf_plot": J[-1, sl],
        "J_ctr_plot": J[0, sl],
        "theta_surf_plot": theta[-1, sl],
        "theta_ctr_plot": theta[0, sl],
        "u_surf_plot": u[-1, sl],
        "u_ctr_plot": u[0, sl],
        "J_plot": J[:, sl],
        "access_plot": access[:, sl],
        "u_phase_profiles": u[:, phase_idx],
        "access_phase_profiles": access[:, phase_idx],
    }
    np.savez_compressed(_CACHE, **cache)
    return cache


def load_or_build_cache():
    required = {
        "cache_version", "t", "x", "J", "u", "theta", "phi", "access",
        "window_start", "window_stop", "phase_idx", "t_plot", "J_surf_plot",
        "J_ctr_plot", "theta_surf_plot", "theta_ctr_plot", "u_surf_plot",
        "u_ctr_plot", "J_plot", "access_plot", "u_phase_profiles",
        "access_phase_profiles",
    }
    if os.path.exists(_CACHE):
        print("  Loading mechanism-detail cache ...")
        d = np.load(_CACHE)
        if required.issubset(d.files) and int(np.atleast_1d(d["cache_version"])[0]) == _CACHE_VERSION:
            return {k: d[k] for k in d.files}
        print("  Cache invalid; rebuilding ...")
    return build_cache()


def add_phase_guides(ax, t_phase):
    for tp, color in zip(t_phase, PHASE_COLORS):
        ax.axvline(tp, color=color, lw=0.8, ls=(0, (2, 2)), alpha=0.9)


def panel_time_series(ax, t_plot, y_surf, y_ctr, ylabel, surf_color, ctr_color=None, legend_loc="upper right"):
    ctr_color = ctr_color or surf_color
    l1, = ax.plot(t_plot, y_surf, color=surf_color, lw=1.2, label="surface")
    l2, = ax.plot(t_plot, y_ctr, color=ctr_color, lw=1.2, ls="--", label="center")
    ax.set_xlabel(r"$\tau - \tau_0$")
    ax.set_ylabel(ylabel)
    ax.legend(handles=[l1, l2], fontsize=6, loc=legend_loc)


def panel_kymograph(ax, x, t_plot, field, cmap, cbar_label, norm=None):
    extent = [t_plot[0], t_plot[-1], x[0], x[-1]]
    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.set_xlabel(r"$\tau - \tau_0$")
    ax.set_ylabel(r"$x/H_0$")
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(cbar_label, fontsize=7)
    return im


def main():
    set_style()
    cache = load_or_build_cache()

    x = cache["x"]
    t_plot = cache["t_plot"]
    phase_idx = cache["phase_idx"].astype(int)
    t_phase = cache["t"][phase_idx] - cache["t"][int(cache["window_start"][0])]

    fig = plt.figure(figsize=(PRE_DOUBLE, 6.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.1],
        hspace=0.42,
        wspace=0.40,
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_time_series(ax_a, t_plot, cache["J_surf_plot"], cache["J_ctr_plot"], r"$J$", C[0])
    add_phase_guides(ax_a, t_phase)

    panel_time_series(ax_b, t_plot, cache["theta_surf_plot"], cache["theta_ctr_plot"], r"$\theta$", C[1])
    add_phase_guides(ax_b, t_phase)

    panel_time_series(ax_c, t_plot, cache["u_surf_plot"], cache["u_ctr_plot"], r"$u$", "#148f87", ctr_color="#5c6bc0")
    add_phase_guides(ax_c, t_phase)
    collapse_local = phase_idx[2] - int(cache["window_start"][0])
    ax_c.plot(
        t_plot[collapse_local],
        cache["u_surf_plot"][collapse_local],
        "o",
        ms=5,
        color=C[1],
        zorder=5,
    )
    phase_handles = [Line2D([0], [0], color=c, lw=1.0, ls=(0, (2, 2)), label=l) for c, l in zip(PHASE_COLORS, PHASE_LABELS)]
    ax_c.legend(handles=ax_c.get_legend_handles_labels()[0] + phase_handles,
                labels=ax_c.get_legend_handles_labels()[1] + PHASE_LABELS,
                fontsize=5.5, loc="upper right", ncol=2)

    panel_kymograph(
        ax_d,
        x,
        t_plot,
        cache["J_plot"],
        cmap="RdBu_r",
        cbar_label=r"$J$",
        norm=Normalize(
            vmin=np.nanpercentile(cache["J_plot"], 2),
            vmax=np.nanpercentile(cache["J_plot"], 98),
        ),
    )
    add_phase_guides(ax_d, t_phase)

    # Zoom x-axis on the reactive shell (ξ ∈ [0.65, 1]) — at N=301 the
    # LCST front sits at ξ ≈ 0.9 and all action lives in this band; a
    # full [0,1] axis crushes the structure into 1-pixel-wide spikes.
    x_show = (x >= 0.65)
    for profile, color, label in zip(cache["u_phase_profiles"].T, PHASE_COLORS, PHASE_LABELS):
        ax_e.semilogy(x[x_show], np.maximum(profile[x_show], 1e-12),
                      color=color, lw=1.2, label=label)
    ax_e.set_xlabel(r"$x/H_0$")
    ax_e.set_ylabel(r"$u(x)$")
    ax_e.set_xlim(0.65, 1.0)
    ax_e.set_ylim(1e-8, 2.0)
    ax_e.legend(fontsize=6, loc="upper left", ncol=2)

    for profile, color, label in zip(cache["access_phase_profiles"].T, PHASE_COLORS, PHASE_LABELS):
        ax_f.plot(x[x_show], profile[x_show], color=color, lw=1.2, label=label)
    ax_f.set_xlabel(r"$x/H_0$")
    ax_f.set_ylabel(r"$(1-\phi)^{m_\mathrm{act}}$")
    ax_f.set_xlim(0.65, 1.0)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("abcdef")):
        add_panel_label(ax, label, x=0.01, y=0.98)

    save(fig, "fig3_mechanism_detail")
    print("Fig.3 detail done.")


if __name__ == "__main__":
    main()
