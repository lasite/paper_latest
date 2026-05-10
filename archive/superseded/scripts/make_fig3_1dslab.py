#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig3_1dslab.py — Fig.3: Two oscillatory regimes in the 1D slab.

Layout: double-column (6.875 in), 3 rows × 2 columns.
  Left column  = Regime II: S_chi=1.0, Da=9.5 ("thermal breathing")
  Right column = Regime I:  S_chi=0.7, Da=4.0 ("volume pulse")

  Row 1 (a,b): Time series ⟨J⟩(τ) and ⟨θ⟩(τ) — dual y-axis
  Row 2 (c,d): Kymograph J(x,τ) — last 3 periods, rasterized imshow
  Row 3 (e,f): Spatial profiles J(x), θ(x), u(x) at peak and trough

Parameters: Bi_c=0.70, Bi_T=0.10, Gamma_A=1.5, N=40, t_end=350.
Regime I: arrh_exp_cap=30, max_step=0.25 to avoid Jacobian singularity.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, kymo_show, save)
from scan_optimized import Params, simulate

set_style()

# ─── Simulation parameters ────────────────────────────────────────
COMMON = dict(Bi_c=0.70, Bi_T=0.10, Gamma_A=1.5, N=40, n_save=6000)

PARAMS_II = Params(Da=9.5, S_chi=1.0, t_end=350, **COMMON)
PARAMS_I  = Params(Da=4.0, S_chi=0.7, t_end=500,
                   arrh_exp_cap=30.0, max_step=0.25, **COMMON)


def run_both():
    print("  Running Regime II (Da=9.5, S_chi=1.0) ...")
    d2 = simulate(PARAMS_II)
    print(f"  Done. nfev={d2['nfev']}")
    print("  Running Regime I  (Da=4.0, S_chi=0.7) ...")
    d1 = simulate(PARAMS_I)
    print(f"  Done. nfev={d1['nfev']}")
    return d2, d1


# ── Helpers ──────────────────────────────────────────────────────

def tail_3cycles(t, signal, frac_start=0.65):
    """Return slice covering last ~3 oscillation cycles."""
    i0 = int(frac_start * len(t))
    t_sub  = t[i0:]
    sig_sub = signal[i0:] if signal.ndim == 1 else signal[:, i0:]

    # find peaks in mean J
    y = sig_sub if sig_sub.ndim == 1 else np.mean(sig_sub, axis=0)
    pks, _ = find_peaks(y, prominence=0.005, distance=10)
    if len(pks) >= 4:
        i_end = pks[3] + 5
        t_sub  = t_sub[:i_end]
        sig_sub = sig_sub[:i_end] if sig_sub.ndim == 1 else sig_sub[:, :i_end]
    return t_sub, sig_sub


def plot_timeseries(ax, data, label_left, label_right, title=""):
    t = data["t"]
    J_mean = np.mean(data["J"], axis=0)
    th_mean = np.mean(data["theta"], axis=0)

    # Use J_mean to define the tail window; apply same indices to θ
    t_tail, J_tail = tail_3cycles(t, J_mean)
    # Apply the same absolute time window to θ
    t0_abs = t_tail[0] + t[int(0.65 * len(t))]   # start in original t
    i_start = np.searchsorted(t, t_tail[0] + t[int(0.65 * len(t))])
    # simpler: recompute indices directly
    n_tail = len(t_tail)
    i0 = int(0.65 * len(t))
    th_tail = th_mean[i0: i0 + n_tail]
    # ensure same length
    min_len = min(len(t_tail), len(th_tail))
    t_tail  = t_tail[:min_len] - t_tail[0]
    J_tail  = J_tail[:min_len]
    th_tail = th_tail[:min_len]

    ax2 = ax.twinx()
    lJ, = ax.plot(t_tail, J_tail, color=C[0], lw=1.0,
                  label=r"$\langle J \rangle$")
    lT, = ax2.plot(t_tail, th_tail, color=C[1], lw=0.9, ls="--",
                   label=r"$\langle \theta \rangle$")

    ax.set_ylabel(label_left, color=C[0], fontsize=8)
    ax2.set_ylabel(label_right, color=C[1], fontsize=8)
    ax.tick_params(axis="y", labelcolor=C[0])
    ax2.tick_params(axis="y", labelcolor=C[1])
    ax.set_xlabel(r"$\tau - \tau_0$")
    if title:
        ax.set_title(title, fontsize=8)
    ax.legend([lJ, lT], [lJ.get_label(), lT.get_label()],
              loc="upper right", fontsize=6.5)


def plot_kymograph(ax, data, label=""):
    t = data["t"]
    x = data["x"]
    J = data["J"]   # shape (N, n_t)

    t_tail, J_tail = tail_3cycles(t, J, frac_start=0.65)
    t_tail = t_tail - t_tail[0]

    vmin = np.nanpercentile(J_tail, 2)
    vmax = np.nanpercentile(J_tail, 98)

    kymo_show(ax, J_tail, x, t_tail, cmap="RdBu_r",
              label=r"$J(x,\tau)$", vmin=vmin, vmax=vmax)

    # overlay u=0.5 isocontour (reaction front)
    u = data["u"]
    _, u_tail = tail_3cycles(t, u, frac_start=0.65)
    try:
        X, T = np.meshgrid(t_tail, x)
        ax.contour(T, X, u_tail, levels=[0.5], colors="yellow",
                   linewidths=0.6, linestyles="--", alpha=0.8)
        # label it once
        ax.text(t_tail[-1] * 0.02, x[-1] * 0.9,
                r"$u\!=\!0.5$", fontsize=5.5, color="yellow")
    except Exception:
        pass

    ax.set_xlabel(r"$\tau - \tau_0$")
    ax.set_ylabel(r"$x / H_0$")
    if label:
        ax.set_title(label, fontsize=8)


def plot_profiles(ax, data, label=""):
    t = data["t"]
    x = data["x"]
    J = data["J"]         # (N, n_t)
    theta = data["theta"]  # (N, n_t)
    u = data["u"]          # (N, n_t)

    J_mean = np.mean(J, axis=0)

    # find peak and trough in tail
    i0 = int(0.65 * len(t))
    J_tail = J_mean[i0:]
    pks, _  = find_peaks( J_tail, prominence=0.005, distance=10)
    trgs, _ = find_peaks(-J_tail, prominence=0.005, distance=10)

    if len(pks) == 0 or len(trgs) == 0:
        i_pk  = i0 + np.argmax(J_tail)
        i_trg = i0 + np.argmin(J_tail)
    else:
        i_pk  = i0 + pks[0]
        i_trg = i0 + trgs[0]

    ax2 = ax.twinx()
    # J profiles
    ax.plot(x, J[:, i_pk],  color=C[0], lw=1.0, label=r"$J$ (peak)")
    ax.plot(x, J[:, i_trg], color=C[0], lw=1.0, ls="--", label=r"$J$ (trough)")
    ax.set_ylabel(r"$J(x)$", color=C[0], fontsize=8)
    ax.tick_params(axis="y", labelcolor=C[0])

    # u profiles
    ax2.plot(x, u[:, i_pk],  color=C[2], lw=0.9, label=r"$u$ (peak)")
    ax2.plot(x, u[:, i_trg], color=C[2], lw=0.9, ls="--", label=r"$u$ (trough)")
    ax2.set_ylabel(r"$u(x)$", color=C[2], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=C[2])

    ax.set_xlabel(r"$x / H_0$")
    if label:
        ax.set_title(label, fontsize=8)

    # combined legend
    lines = (ax.get_lines() + ax2.get_lines())
    ax.legend(lines, [l.get_label() for l in lines],
              fontsize=5.5, loc="upper left", ncol=2)


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.3
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()
    data_II, data_I = run_both()

    fig = plt.figure(figsize=(PRE_DOUBLE, 7.8))
    gs  = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.48,
                           left=0.10, right=0.97, top=0.97, bottom=0.07)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    # Row 1: time series
    plot_timeseries(axes[0][0], data_II,
                    r"$\langle J \rangle$", r"$\langle \theta \rangle$",
                    title=r"Regime II: $S_\chi=1.0$, Da$=9.5$ (thermal breathing)")
    plot_timeseries(axes[0][1], data_I,
                    r"$\langle J \rangle$", r"$\langle \theta \rangle$",
                    title=r"Regime I: $S_\chi=0.7$, Da$=4.0$ (volume pulse)")

    # Row 2: kymographs
    plot_kymograph(axes[1][0], data_II, label=r"$J(x,\tau)$ — Regime II")
    plot_kymograph(axes[1][1], data_I,  label=r"$J(x,\tau)$ — Regime I")

    # Row 3: spatial profiles
    plot_profiles(axes[2][0], data_II, label="Spatial profiles — Regime II")
    plot_profiles(axes[2][1], data_I,  label="Spatial profiles — Regime I")

    # Panel labels
    labels = ["a","b","c","d","e","f"]
    for row in range(3):
        for col in range(2):
            add_panel_label(axes[row][col], labels[row*2 + col])

    save(fig, "fig3_1dslab")
    print("Fig.3 done.")


if __name__ == "__main__":
    main()
