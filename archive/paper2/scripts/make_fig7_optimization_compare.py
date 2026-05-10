#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig7_optimization_compare.py — Original vs optimized penetration dynamics.

Layout: double-column, 3 rows × 4 columns.
  (a,b) J surface/center time series
  (c,d) u kymographs
  (e,f,g,h) u_peak(x) and theta time series for the two parameter sets
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

from style_pub import set_style, PRE_DOUBLE, C, add_panel_label, save
from scan_optimized import Params, simulate

set_style()

_CACHE = os.path.join(os.path.dirname(__file__), "figures_pub/fig7_optimization_compare_cache.npz")
_CACHE_VERSION = 2


def make_params(original):
    common = dict(
        S_chi=1.0,
        Bi_T=0.10,
        Gamma_A=1.5,
        N=40,
        t_end=600.0,
        n_save=5000,
    )
    if original:
        return Params(Da=14.0, D0=1.0, Bi_c=0.35, **common)
    return Params(Da=4.0, D0=2.0, Bi_c=0.70, **common)


def run_case(label, p):
    print(f"  Running {label} optimization case ...")
    data = simulate(p)
    print(f"  Done. nfev={data['nfev']}")
    return data


def select_window(t, J_surf, frac_start=0.60):
    i0 = int(frac_start * len(t))
    tail = J_surf[i0:]
    prominence = max(0.02, 0.08 * np.ptp(tail))
    peaks, _ = find_peaks(tail, prominence=prominence, distance=35)
    if len(peaks) >= 4:
        peaks = peaks[-4:] + i0
        start = peaks[0]
        stop = peaks[-1]
    else:
        start = i0
        stop = len(t) - 1
    return slice(start, stop)


def summarize_case(data):
    t = np.asarray(data["t"])
    x = np.asarray(data["x"])
    J = np.asarray(data["J"])
    u = np.asarray(data["u"])
    theta = np.asarray(data["theta"])

    sl = select_window(t, J[-1])
    t_plot = t[sl] - t[sl][0]
    mid_idx = int(np.argmin(np.abs(x - 0.5)))

    return {
        "t": t,
        "x": x,
        "J": J,
        "u": u,
        "theta": theta,
        "window_start": np.array([sl.start], dtype=int),
        "window_stop": np.array([sl.stop], dtype=int),
        "t_plot": t_plot,
        "J_surf_plot": J[-1, sl],
        "J_ctr_plot": J[0, sl],
        "u_plot": u[:, sl],
        "theta_surf_plot": theta[-1, sl],
        "theta_ctr_plot": theta[0, sl],
        "u_peak_profile": np.max(u[:, sl], axis=1),
        "u_mid_peak": np.array([np.max(u[mid_idx, sl])]),
    }


def build_cache():
    original = summarize_case(run_case("original", make_params(True)))
    optimized = summarize_case(run_case("optimized", make_params(False)))
    cache = {"cache_version": np.array([_CACHE_VERSION], dtype=int)}
    for prefix, summary in [("orig", original), ("opt", optimized)]:
        for key, value in summary.items():
            cache[f"{prefix}_{key}"] = value
    np.savez_compressed(_CACHE, **cache)
    return cache


def load_or_build_cache():
    required = {
        "cache_version",
        "orig_t", "orig_x", "orig_J", "orig_u", "orig_theta", "orig_t_plot",
        "orig_J_surf_plot", "orig_J_ctr_plot", "orig_u_plot",
        "orig_theta_surf_plot", "orig_theta_ctr_plot", "orig_u_peak_profile",
        "orig_u_mid_peak",
        "opt_t", "opt_x", "opt_J", "opt_u", "opt_theta", "opt_t_plot",
        "opt_J_surf_plot", "opt_J_ctr_plot", "opt_u_plot",
        "opt_theta_surf_plot", "opt_theta_ctr_plot", "opt_u_peak_profile",
        "opt_u_mid_peak",
    }
    if os.path.exists(_CACHE):
        print("  Loading optimization-comparison cache ...")
        d = np.load(_CACHE)
        if required.issubset(d.files) and int(np.atleast_1d(d["cache_version"])[0]) == _CACHE_VERSION:
            return {k: d[k] for k in d.files}
        print("  Cache invalid; rebuilding ...")
    return build_cache()


def plot_timeseries(ax, t_plot, surf, ctr, ylabel, color):
    ax.plot(t_plot, surf, color=color, lw=1.2, label="surface")
    ax.plot(t_plot, ctr, color=color, lw=1.2, ls="--", label="center")
    ax.set_xlabel(r"$\tau - \tau_0$")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, loc="upper right")


def plot_u_kymograph(ax, x, t_plot, u_plot):
    im = ax.imshow(
        u_plot,
        origin="lower",
        aspect="auto",
        extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
        cmap="inferno",
        norm=LogNorm(vmin=1e-12, vmax=1.0),
        rasterized=True,
    )
    ax.set_xlabel(r"$\tau - \tau_0$")
    ax.set_ylabel(r"$x/H_0$")
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(r"$u$", fontsize=7)


def plot_u_profile(ax, x, u_peak, color):
    ax.semilogy(x, u_peak, color=color, lw=1.4)
    ax.axhline(1e-6, color="0.5", lw=0.9, ls=(0, (1.5, 2.0)))
    ax.set_xlabel(r"$x/H_0$")
    ax.set_ylabel(r"$u_\mathrm{peak}(x)$")


def main():
    set_style()
    cache = load_or_build_cache()

    fig = plt.figure(figsize=(PRE_DOUBLE, 5.2))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.1],
        hspace=0.42,
        wspace=0.35,
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.09,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    fig.text(0.30, 0.985, "Original", ha="center", va="top", fontsize=9,
             fontstyle="italic")
    fig.text(0.76, 0.985, "Optimized", ha="center", va="top", fontsize=9,
             fontstyle="italic")

    plot_timeseries(ax_a, cache["orig_t_plot"], cache["orig_J_surf_plot"], cache["orig_J_ctr_plot"], r"$J$", C[0])
    plot_timeseries(ax_b, cache["opt_t_plot"], cache["opt_J_surf_plot"], cache["opt_J_ctr_plot"], r"$J$", C[1])

    plot_u_kymograph(ax_c, cache["orig_x"], cache["orig_t_plot"], cache["orig_u_plot"])
    plot_u_kymograph(ax_d, cache["opt_x"], cache["opt_t_plot"], cache["opt_u_plot"])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], list("abcd")):
        add_panel_label(ax, label, x=0.01, y=0.98)

    save(fig, "fig7_optimization_compare")
    print("Fig.7 compare done.")


if __name__ == "__main__":
    main()
