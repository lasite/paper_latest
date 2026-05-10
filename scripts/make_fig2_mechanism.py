#!/usr/bin/env python3
"""
make_fig2_mechanism.py — Fig 4 (paper): anatomy of the LCST-front
relaxation cycle. 4-panel APS-style figure, 2×2 equal-size subplots,
minimal annotation.

  (a) Slow manifold geometry: μ(J,θ) − μ_b heatmap with the μ = μ_b
      contour (the slow manifold) and fold points. No cycle.
  (b) PDE limit cycles in (θ, J) at ξ = 1 (surface, full S-curve
      traversal) and ξ = 0 (centre, small loop confined to the
      swollen branch — the inset). Speed-coloured. The contrast
      between the two cycles is the §IV.B "spatially generated"
      argument made visible.
  (c) Reactant kymograph u(ξ, τ) on a log scale with ξ_LCST line.
  (d) Cycle period T_PDE vs 1/Bi_T scatter, coloured by S_χ.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks

from style_pub import set_style, PRE_DOUBLE, add_panel_label, OUT, C
set_style()

# Reuse helpers (no re-implementation of Mu/branches/markers logic)
from make_slow_manifold import (
    trace_branches,
    limit_cycle_segments,
    detect_phase_markers,
)
from scan_optimized import Params, finalize_params, local_chem_pot
from fig2_data import WORKING_POINT, load_cache

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FIG2 = _DATA_DIR / "fig2_alpha003_cache.npz"
PERIOD_CACHE = _DATA_DIR / "fig4" / "fig4_grid_Bi_T_S_chi.npz"


# ──────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────
def run_or_load():
    """Run α=0.03 simulation or load existing cache."""
    if CACHE_FIG2.exists():
        print(f"  Using cache: {CACHE_FIG2}")
        return np.load(CACHE_FIG2)
    print("  Running simulation (α=0.03) ...")
    from scan_optimized import simulate
    p_dict = dict(WORKING_POINT)
    p_dict["n_save"] = 2500
    p = Params(**p_dict)
    result = simulate(p)
    t, J, u, theta = result["t"], result["J"], result["u"], result["theta"]
    phi = p.phi_p0 / J
    access = np.power(np.clip(1.0 - phi, 1e-12, 1.0), p.m_act)
    np.savez(CACHE_FIG2, x=result["x"], t=t, J=J, u=u, theta=theta,
             access=access)
    return np.load(CACHE_FIG2)


# ──────────────────────────────────────────────────────────────────
# (a) Slow manifold geometry: μ heatmap + μ=μ_b contour (no cycle)
# ──────────────────────────────────────────────────────────────────
def panel_geometry(ax, fig, p, bran):
    """μ(J,θ) − μ_b heatmap with the μ = μ_b contour overlay (the slow
    manifold) and fold points. Geometry only, no PDE cycle."""
    th_min, th_max = -0.3, 5.0
    J_min,  J_max  = 0.10, 1.85

    theta_g = np.linspace(th_min, th_max, 220)
    J_g     = np.linspace(J_min,  J_max,  220)
    TH, JJ = np.meshgrid(theta_g, J_g)
    MU = (local_chem_pot(JJ.ravel(), TH.ravel(), p)
          - p.m_b).reshape(JJ.shape)
    sat = 0.6
    pcm = ax.pcolormesh(TH, JJ, MU, cmap="RdBu_r",
                        vmin=-sat, vmax=sat, shading="auto",
                        rasterized=True, zorder=1)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.045,
                     extend="both")
    cb.set_label(r"$\mu(J,\theta) - \mu_b$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.contour(TH, JJ, MU, levels=[0.0], colors="black",
               linewidths=1.2, zorder=3)
    for k, (th_f, J_f) in bran["folds"].items():
        ax.plot(th_f, J_f, "o", mfc="white", mec="k", mew=0.8,
                ms=3.5, zorder=5)

    ax.set_xlabel(r"$\theta$", fontsize=9)
    ax.set_ylabel(r"$J$", fontsize=9)
    ax.set_xlim(th_min, th_max)
    ax.set_ylim(J_min, J_max)
    ax.tick_params(labelsize=7, direction="out", length=2.5)


# ──────────────────────────────────────────────────────────────────
# (b) PDE cycles: surface ξ=1 (full traversal) + centre ξ=0 (inset)
#     Demonstrates the spatial structure: the slow manifold is felt
#     at the surface but not in the bulk.
# ──────────────────────────────────────────────────────────────────
def panel_cycles(ax, fig, bran, t_w, J_surf, th_surf, J_ctr, th_ctr):
    """Surface (ξ=1) limit cycle and centre (ξ=0) limit cycle in (θ,J).
    Surface cycle traverses the full S-curve between the folds; centre
    cycle is a tiny loop confined to the swollen branch's interior. The
    centre cycle is shown in a zoomed inset because its size is two
    orders of magnitude smaller than the surface cycle."""
    th_min, th_max = -0.3, 5.0
    J_min,  J_max  = 0.10, 1.85

    # Faint S-curve reference for the surface cycle (no heatmap to
    # avoid duplicating panel (a))
    upper, middle, lower = bran["upper"], bran["middle"], bran["lower"]
    if len(upper):
        ax.plot(upper[:, 0], upper[:, 1], "-", color="0.55", lw=0.9,
                zorder=2)
    if len(lower):
        ax.plot(lower[:, 0], lower[:, 1], "-", color="0.55", lw=0.9,
                zorder=2)
    if len(middle):
        order = np.argsort(middle[:, 0])
        ax.plot(middle[order, 0], middle[order, 1], "--", color="0.65",
                lw=0.8, zorder=2)
    for k, (th_f, J_f) in bran["folds"].items():
        ax.plot(th_f, J_f, "o", mfc="white", mec="0.4", mew=0.7,
                ms=3.0, zorder=3)

    # Surface cycle, speed-coloured
    segs, speed = limit_cycle_segments(th_surf, J_surf, t_w)
    norm = LogNorm(vmin=max(speed.min(), 1e-3), vmax=speed.max())
    lc = LineCollection(segs, cmap="viridis", norm=norm, lw=1.6,
                        zorder=6, alpha=0.95)
    lc.set_array(speed)
    ax.add_collection(lc)

    ax.set_xlabel(r"$\theta$", fontsize=9)
    ax.set_ylabel(r"$J$", fontsize=9)
    ax.set_xlim(th_min, th_max)
    ax.set_ylim(J_min, J_max)
    ax.tick_params(labelsize=7, direction="out", length=2.5)

    # Inset: centre cycle (zoomed). Size + position chosen to sit in
    # the empty upper-left quadrant of the surface cycle plot.
    axins = inset_axes(ax, width="44%", height="38%",
                      loc="upper left",
                      bbox_to_anchor=(0.07, -0.05, 1, 1),
                      bbox_transform=ax.transAxes,
                      borderpad=0.0)
    th_ctr_pad = 0.02 * (th_ctr.max() - th_ctr.min())
    J_ctr_pad  = 0.02 * (J_ctr.max() - J_ctr.min())
    axins.plot(th_ctr, J_ctr, "-", color="0.15", lw=1.0)
    axins.set_xlim(th_ctr.min() - th_ctr_pad,
                   th_ctr.max() + th_ctr_pad)
    axins.set_ylim(J_ctr.min() - J_ctr_pad,
                   J_ctr.max() + J_ctr_pad)
    axins.tick_params(labelsize=6, direction="out", length=2.0)
    axins.set_title(r"centre $\xi\!=\!0$", fontsize=7, pad=2)
    axins.set_facecolor("#f7f7f7")
    return lc


# ──────────────────────────────────────────────────────────────────
# (b) time-series stack
# ──────────────────────────────────────────────────────────────────
def draw_timeseries(fig, gs_b, t, J_surf, J_ctr, th_surf, th_ctr,
                    u_surf, u_ctr, idx, ts):
    sub = gs_b.subgridspec(3, 1, hspace=0.10)
    ax_J  = fig.add_subplot(sub[0])
    ax_th = fig.add_subplot(sub[1], sharex=ax_J)
    ax_u  = fig.add_subplot(sub[2], sharex=ax_J)

    ax_J.plot(ts, J_surf[idx],  '-',  color='#1f77b4', lw=1.0,
              label=r'surface ($\xi=1$)')
    ax_J.plot(ts, J_ctr[idx],   '--', color='#1f77b4', lw=0.8,
              label=r'center ($\xi=0$)')
    ax_J.set_ylabel(r'$J$', fontsize=8)
    ax_J.legend(fontsize=6, loc='lower center', ncol=2, framealpha=0.85,
                handlelength=1.6, columnspacing=0.8,
                bbox_to_anchor=(0.5, 1.00))
    ax_J.tick_params(labelbottom=False)

    ax_th.plot(ts, th_surf[idx], '-',  color='#d62728', lw=1.0)
    ax_th.plot(ts, th_ctr[idx],  '--', color='#d62728', lw=0.8)
    ax_th.set_ylabel(r'$\theta$', fontsize=8)
    ax_th.tick_params(labelbottom=False)

    u_surf_safe = np.clip(u_surf[idx], 1e-6, None)
    u_ctr_safe  = np.clip(u_ctr[idx],  1e-6, None)
    ax_u.semilogy(ts, u_surf_safe, '-',  color='#2ca02c', lw=1.0)
    ax_u.semilogy(ts, u_ctr_safe,  '--', color='#2ca02c', lw=0.8)
    ax_u.set_ylim(1e-6, 2.0)
    ax_u.set_ylabel(r'$u$', fontsize=8)
    ax_u.set_xlabel(r'$\tau$', fontsize=8)
    ax_u.set_yticks([1e-5, 1e-3, 1e-1])
    ax_u.grid(True, which='major', axis='y', alpha=0.25, lw=0.4)

    for ax in [ax_J, ax_th, ax_u]:
        ax.tick_params(labelsize=7)
    return ax_J




# ──────────────────────────────────────────────────────────────────
# (c) Reactant kymograph u(xi, tau) with xi_LCST line
# ──────────────────────────────────────────────────────────────────
def panel_u_kymograph(ax, fig, t, x, J, u, idx, t_start, t_end,
                      phi_p0=0.15):
    """Reactant kymograph u(ξ,τ) on log scale. Single annotation: the
    horizontal white dashed line at ξ_LCST."""
    u_window = np.clip(u[:, idx], 1e-6, 2.0)
    norm_u = LogNorm(vmin=1e-6, vmax=1.0)
    im = ax.imshow(u_window, origin="lower", aspect="auto",
                   extent=[t_start, t_end, 0, 1],
                   cmap="cividis", norm=norm_u, rasterized=False)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045,
                      extend="min")
    cb.set_label(r"$u(\xi,\tau)$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ξ_LCST: innermost xi where max_t φ ever crosses 0.5 (same window)
    phi_window = phi_p0 / J[:, idx]
    over = phi_window.max(axis=1) >= 0.5
    if over.any():
        xi_lcst = float(x[int(np.where(over)[0].min())])
    else:
        xi_lcst = 0.885
    ax.axhline(xi_lcst, color="white", lw=1.0, ls="--", alpha=0.85,
               zorder=4)

    ax.set_xlabel(r"$\tau$", fontsize=9)
    ax.set_ylabel(r"$\xi$", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=7, direction="out", length=2.5)


# ──────────────────────────────────────────────────────────────────
# (e) period scaling
# ──────────────────────────────────────────────────────────────────
def panel_period_scaling(ax, fig):
    if not PERIOD_CACHE.exists():
        ax.text(0.5, 0.5, "period cache absent", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    d = np.load(PERIOD_CACHE, allow_pickle=True)
    x = d["x"]; y = d["y"]
    regime = d["regime"]
    T = d["period"]

    osc = (regime == 1) | (regime == 2) | (regime == 3)
    osc &= np.isfinite(T) & (T > 0)

    BiT_arr, Sx_arr, T_arr = [], [], []
    for i, BiT in enumerate(x):
        for j, Sx in enumerate(y):
            if osc[i, j]:
                BiT_arr.append(BiT); Sx_arr.append(Sx); T_arr.append(T[i, j])
    BiT_arr = np.array(BiT_arr)
    Sx_arr  = np.array(Sx_arr)
    T_arr   = np.array(T_arr)

    sc = ax.scatter(1.0 / BiT_arr, T_arr, c=Sx_arr,
                    cmap="viridis", s=14, edgecolor="k", linewidth=0.3,
                    zorder=4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(r"$S_\chi$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    main_band = (Sx_arr >= 0.30) & (Sx_arr <= 1.05)
    if main_band.sum() > 5:
        slope, intercept = np.polyfit(
            np.log(1.0 / BiT_arr[main_band]),
            np.log(T_arr[main_band]),
            1,
        )
        BiT_grid = np.linspace(BiT_arr.min(), BiT_arr.max(), 100)
        T_pred = np.exp(intercept) * (1.0 / BiT_grid) ** slope
        ax.plot(1.0 / BiT_grid, T_pred, "k--", lw=0.9, alpha=0.75)
        ax.plot(1.0 / BiT_grid, 1.0 / BiT_grid, ":", color="0.4",
                lw=0.9)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/\mathrm{Bi}_T$", fontsize=9)
    ax.set_ylabel(r"$T_{\rm PDE}$", fontsize=9)
    ax.tick_params(labelsize=7, direction="out", length=2.5)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    d = run_or_load()
    t, x = d["t"], d["x"]
    J, u, theta = d["J"], d["u"], d["theta"]

    J_surf, J_ctr = J[-1, :], J[0, :]
    th_surf, th_ctr = theta[-1, :], theta[0, :]

    t_start, t_end = 180.0, 240.0
    idx = (t >= t_start) & (t <= t_end)
    ts = t[idx]

    # 4-panel mechanism figure, 2×2 equal-size subplots:
    #  (a) slow-manifold geometry (μ heatmap + S-curve)
    #  (b) PDE cycles at surface (ξ=1) and centre (ξ=0, inset)
    #  (c) reactant kymograph u(ξ,τ)
    #  (d) period scaling T_PDE vs 1/Bi_T
    fig = plt.figure(figsize=(PRE_DOUBLE, PRE_DOUBLE * 0.85))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.35, wspace=0.40,
        left=0.08, right=0.96, top=0.95, bottom=0.10,
    )

    p = finalize_params(Params(**WORKING_POINT))
    bran = trace_branches(p)
    t_w = ts.copy()
    J_w  = J_surf[idx]; th_w  = th_surf[idx]
    Jc_w = J_ctr[idx];  thc_w = th_ctr[idx]

    # (a) geometry: μ heatmap + S-curve contour, no cycle
    ax_a = fig.add_subplot(gs[0, 0])
    panel_geometry(ax_a, fig, p, bran)
    add_panel_label(ax_a, "a")

    # (b) cycles: surface (full S-curve traversal) + centre (inset)
    ax_b = fig.add_subplot(gs[0, 1])
    lc = panel_cycles(ax_b, fig, bran, t_w, J_w, th_w, Jc_w, thc_w)
    cb_speed = plt.colorbar(lc, ax=ax_b, pad=0.02, fraction=0.045)
    cb_speed.set_label(r"speed", fontsize=8)
    cb_speed.ax.tick_params(labelsize=7)
    add_panel_label(ax_b, "b")

    # (c) u kymograph
    ax_c = fig.add_subplot(gs[1, 0])
    panel_u_kymograph(ax_c, fig, t, x, J, u, idx, t_start, t_end,
                     phi_p0=WORKING_POINT["phi_p0"])
    add_panel_label(ax_c, "c")

    # (d) period scaling
    ax_d = fig.add_subplot(gs[1, 1])
    panel_period_scaling(ax_d, fig)
    add_panel_label(ax_d, "d")

    # ── Save ────────────────────────────────────────────────────
    targets = [
        Path(OUT),
        Path(__file__).resolve().parent.parent / "Figure" / "pub",
    ]
    for tgt in targets:
        tgt.mkdir(parents=True, exist_ok=True)
        out = tgt / "fig4"
        fig.savefig(str(out) + ".pdf", dpi=600, bbox_inches="tight")
        fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
