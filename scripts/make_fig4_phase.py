#!/usr/bin/env python3
"""
make_fig4_phase.py — Fig 2 (paper): nonlinear regime map.

Four panels drawn from the parallel grid scans of ``fig4_data.py``:

  (a) Regime classification on (Bi_T, S_chi) plane with WP marker.
  (b) Regime classification on (Bi_T, Da) at S_chi = WP — confirms the
      LCST-front regime persists when reactivity is varied.
  (c) Period heatmap on (Bi_T, S_chi) (oscillating cells only).
  (d) max φ heatmap on the same grid (LCST-crossing depth).

Earlier versions also carried a J-amplitude heatmap and a J(ξ)
profile panel; both were retired during the §IV restructure (the
amplitude map duplicates the regime map's information; the J(ξ)
profiles are now shown explicitly per attractor in fig3).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import (ListedColormap, BoundaryNorm,
                               LogNorm, Normalize)

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT
from fig4_data import (build_grid, representative_runs, REGIME_NAMES,
                       BI_T_VALS, S_CHI_VALS, DA_VALS, T_ANA, SIM_OVERRIDES,
                       REG_FAILED, REG_STEADY_COLD, REG_BULK_HOPF,
                       REG_LCST_FRONT, REG_GLOBAL_COLLAPSE,
                       REG_STEADY_COLLAPSED, REG_STEADY_FRONT)
from hopf_boundary import hopf_grid

set_style()

LBL_FS  = 9
TICK_FS = 7
LEG_FS  = 7


# ── Regime palette ────────────────────────────────────────────────────
REG_COLORS = {
    REG_FAILED:           "#cccccc",  # light grey – solver failed
    REG_STEADY_COLD:      "#cfe7ff",
    REG_BULK_HOPF:        "#7fbf7b",
    REG_LCST_FRONT:       "#d6604d",
    REG_GLOBAL_COLLAPSE:  "#762a83",
    REG_STEADY_COLLAPSED: "#3a1f73",
    REG_STEADY_FRONT:     "#fed98e",
}
REG_LABELS = {
    REG_STEADY_COLD:      "steady cold",
    REG_BULK_HOPF:        "bulk Hopf",
    REG_LCST_FRONT:       "LCST front",
    REG_STEADY_FRONT:     "frozen front",
    REG_GLOBAL_COLLAPSE:  "global collapse",
    REG_STEADY_COLLAPSED: "uniform collapsed",
    REG_FAILED:           "failed (SNIC)",
}


def _regime_pcm(ax, x, y, reg, x_log=False, y_log=False):
    """Pcolormesh for a categorical regime map."""
    codes = sorted(REG_COLORS.keys())
    cmap = ListedColormap([REG_COLORS[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    pcm = ax.pcolormesh(x, y, reg, cmap=cmap, norm=norm, shading="nearest",
                        rasterized=True)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    return pcm


def _heatmap(ax, x, y, Z, cmap="viridis", norm=None,
             x_log=False, y_log=False, **kw):
    pcm = ax.pcolormesh(x, y, Z, cmap=cmap, norm=norm, shading="nearest",
                        rasterized=True, **kw)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    return pcm


def _wp_marker(ax, xv, yv, label="WP"):
    ax.scatter([xv], [yv], s=52, marker="*", color="white",
               edgecolor="k", linewidth=0.8, zorder=8)
    ax.annotate(label, (xv, yv), xytext=(5, 4),
                textcoords="offset points", fontsize=6.5, color="0.10")


# ─────────────────────────────────────────────────────────────────────
def _overlay_hopf(ax, h, label_pos=None):
    """Outline the analytical 0D Hopf-unstable region.

    The Hopf-unstable indicator is ``Re σ_complex > 0`` (a complex pair
    in the right half-plane). In the stable region the complex pair
    sometimes collides on the real axis and disappears, leaving
    ``re_max_complex = -inf``; the standard ``contour(level=0)`` then
    fails to draw an edge. So we instead draw the ``0.5`` contour of
    the binary indicator ``Re σ_complex > 0``, which is robust.
    """
    Z = h["re_max_complex"]
    mask = np.isfinite(Z) & (Z > 0.0)
    if not mask.any():
        return
    indicator = mask.astype(float)
    # dark backing pass for legibility on any fill colour
    ax.contour(h["x"], h["y"], indicator, levels=[0.5],
               colors="0.10", linewidths=2.6, linestyles="-",
               alpha=0.55, zorder=5.5)
    # bright dashed line on top
    ax.contour(h["x"], h["y"], indicator, levels=[0.5],
               colors="white", linewidths=1.4, linestyles="--",
               zorder=6)
    if label_pos is not None:
        ax.text(*label_pos, "0D Hopf onset",
                color="white", fontsize=5.5,
                ha="center", va="center", zorder=7,
                bbox=dict(facecolor="0.20", edgecolor="white",
                          boxstyle="round,pad=0.18", lw=0.4))


def panel_a(ax, g, h=None, wp_x="Bi_T", wp_y="S_chi"):
    """Regime classification on (Bi_T, S_chi). 0D LSA overlay is
    intentionally OMITTED here: the lcst_front and steady_front regimes
    are intrinsically nonlinear (hard-excitation / subcritical Hopf
    around branches the linearization does not see), and overlaying the
    swollen-branch LSA boundary misled readers into thinking the regime
    edges align with linear instability. See `compare_lsa_vs_nonlinear.py`
    and the supplementary LSA-vs-NL panel for the explicit comparison.
    """
    pcm = _regime_pcm(ax, g["x"], g["y"], g["regime"], x_log=True)
    _wp_marker(ax, WORKING_POINT[wp_x], WORKING_POINT[wp_y])
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)


def panel_b(ax, g, fig=None):
    """Surface J amplitude. Non-oscillating cells (J_amp < AMP_THRESH)
    are rendered as a soft grey rather than masked white, so the panel
    background reads as 'low oscillation' instead of 'no data'."""
    Z = g["J_amp_max"].copy()
    Z = np.where(np.isfinite(Z), np.maximum(Z, 1e-3), np.nan)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_under("#e8e8e8")  # cells below vmin → light grey
    cmap.set_bad("#e8e8e8")    # NaN (failed) → same grey
    vmax = max(3.0, np.nanmax(Z) * 1.05) if np.isfinite(Z).any() else 3.0
    norm = LogNorm(vmin=1e-2, vmax=vmax)
    pcm = _heatmap(ax, g["x"], g["y"], Z, cmap=cmap,
                   norm=norm, x_log=True)
    _wp_marker(ax, WORKING_POINT["Bi_T"], WORKING_POINT["S_chi"])
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    if fig is not None:
        cb = fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=TICK_FS - 1)
        cb.set_label(r"$\max_\xi\,\Delta J$", fontsize=LBL_FS - 1)


def panel_c(ax, g, fig=None):
    Z = g["phi_max"]
    norm = Normalize(vmin=0.10, vmax=1.0)
    pcm = _heatmap(ax, g["x"], g["y"], Z, cmap="cividis",
                   norm=norm, x_log=True)
    # 0.5 LCST contour
    ax.contour(g["x"], g["y"], Z, levels=[0.5],
               colors="white", linewidths=1.0, zorder=5)
    _wp_marker(ax, WORKING_POINT["Bi_T"], WORKING_POINT["S_chi"])
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    if fig is not None:
        cb = fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=TICK_FS - 1)
        cb.set_label(r"$\max\,\varphi$", fontsize=LBL_FS - 1)


def panel_d(ax, g, h=None):
    """Same omission of LSA overlay as panel_a — see panel_a docstring."""
    pcm = _regime_pcm(ax, g["x"], g["y"], g["regime"], x_log=True, y_log=True)
    _wp_marker(ax, WORKING_POINT["Bi_T"], WORKING_POINT["Da"])
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$\mathrm{Da}$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    ax.text(0.02, 0.98,
            rf"$S_\chi = {WORKING_POINT['S_chi']:.2f}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=LEG_FS,
            bbox=dict(facecolor="white", edgecolor="0.5",
                      boxstyle="round,pad=0.18", lw=0.4))


def panel_e(ax, g, fig=None):
    P = g["period"]
    osc = (g["regime"] == REG_LCST_FRONT) | (g["regime"] == REG_BULK_HOPF) \
          | (g["regime"] == REG_GLOBAL_COLLAPSE)
    Z = np.where(osc & np.isfinite(P) & (P > 0), P, np.nan)
    if np.all(~np.isfinite(Z)):
        ax.text(0.5, 0.5, "no oscillating cells", ha="center", va="center",
                transform=ax.transAxes, fontsize=LBL_FS)
        ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
        ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
        return
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    # LogNorm exposes the SNIC period divergence near the upper-S_chi
    # boundary that linear scaling visually compresses.
    cmap = plt.get_cmap("plasma_r").copy()
    cmap.set_bad("#ececec")  # non-oscillating cells -> light grey
    pcm = _heatmap(ax, g["x"], g["y"], Z, cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax), x_log=True)
    _wp_marker(ax, WORKING_POINT["Bi_T"], WORKING_POINT["S_chi"])
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=LBL_FS)
    ax.set_ylabel(r"$S_\chi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    if fig is not None:
        cb = fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=TICK_FS - 1)
        cb.set_label(r"period $T$ (log)", fontsize=LBL_FS - 1)


def panel_f(ax, repr_runs, t_window=T_ANA):
    """Spatial J(ξ) profiles — time-mean line, with min/max envelope band
    for the oscillating regime. Splits the three regimes cleanly:
    steady cold = flat near J_eq, LCST front = envelope band, frozen
    front = static step from swollen core to collapsed shell.
    """
    ordering = [
        ("steady_cold",   "steady cold",     "#3676b8"),
        ("lcst_front_WP", "LCST front (WP)", "#d6604d"),
        ("steady_front",  "frozen front",    "#c89a2c"),
    ]
    for lbl, title, c in ordering:
        if lbl not in repr_runs:
            continue
        r = repr_runs[lbl]
        t = r["t"]; J = r["J"]; x = r["x"]
        idx = (t >= t_window[0]) & (t <= t_window[1])
        Jw = J[:, idx]
        J_mean = Jw.mean(axis=1)
        J_min  = Jw.min(axis=1)
        J_max  = Jw.max(axis=1)
        if (J_max - J_min).max() > 0.05:
            ax.fill_between(x, J_min, J_max, color=c, alpha=0.25, lw=0)
        ax.plot(x, J_mean, color=c, lw=1.4, label=title)

    ax.set_xlabel(r"$\xi$", fontsize=LBL_FS)
    ax.set_ylabel(r"$J(\xi)$", fontsize=LBL_FS)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    ax.legend(fontsize=LEG_FS, loc="center left",
              framealpha=0.9, handlelength=1.2,
              borderpad=0.25, labelspacing=0.25)


# ── Composite ─────────────────────────────────────────────────────────
def main():
    g_main = build_grid(param_x="Bi_T", x_vals=BI_T_VALS,
                        param_y="S_chi", y_vals=S_CHI_VALS)
    g_da   = build_grid(param_x="Bi_T", x_vals=BI_T_VALS,
                        param_y="Da",   y_vals=DA_VALS)
    # LSA grids no longer overlaid on (a)/(d); see panel_a docstring.
    # `hopf_grid` is still computed and cached for the supplementary
    # LSA-vs-NL disagreement panel.

    repr_pts = [
        ("steady_cold",   dict(WORKING_POINT, S_chi=0.20)),
        ("lcst_front_WP", dict(WORKING_POINT)),
        ("steady_front",  dict(WORKING_POINT, Bi_T=0.18, S_chi=1.50)),
    ]
    repr_runs = representative_runs(repr_pts)

    fig = plt.figure(figsize=(PRE_DOUBLE, 5.7))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.55,
                           left=0.07, right=0.97, top=0.94, bottom=0.13)

    # New layout (PLAN): 2x2.
    #  (a) regime (Bi_T, S_chi)   (b) regime (Bi_T, Da)
    #  (c) period                 (d) max phi
    ax_a = fig.add_subplot(gs[0, 0]); panel_a(ax_a, g_main)
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1]); panel_d(ax_b, g_da)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0]); panel_e(ax_c, g_main, fig=fig)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1]); panel_c(ax_d, g_main, fig=fig)
    add_panel_label(ax_d, "d")

    # Common Bi_T x-axis range across all 4 panels (data span is the
    # same BI_T_VALS grid; without this, NaN-masking visually shrinks
    # the displayed range of (c) vs (a)/(d)).
    bi_t_lo, bi_t_hi = float(BI_T_VALS.min()), float(BI_T_VALS.max())
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlim(bi_t_lo, bi_t_hi)

    # Common S_chi y-axis range for the three S_chi panels.
    s_chi_lo, s_chi_hi = float(S_CHI_VALS.min()), float(S_CHI_VALS.max())
    for ax in (ax_a, ax_c, ax_d):
        ax.set_ylim(s_chi_lo, s_chi_hi)

    # Global regime legend (one entry per regime appearing anywhere,
    # including REG_FAILED so the grey solver-failure cells are
    # explicitly identified).
    present = sorted(set(g_main["regime"].flatten().tolist())
                     | set(g_da["regime"].flatten().tolist()))
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=REG_COLORS[int(c)],
                             edgecolor="0.4",
                             label=REG_LABELS[int(c)])
               for c in present]
    # Auxiliary entry for the masked-NaN background of panel (c).
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#ececec",
                                 edgecolor="0.4",
                                 label="non-oscillating (in c)"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=len(handles),
               fontsize=LEG_FS, framealpha=0.9, handlelength=1.1,
               borderpad=0.35, columnspacing=1.1)

    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            "..", "Figure", "pub"))
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "fig2.pdf")
    png = os.path.join(out_dir, "fig2.png")
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
