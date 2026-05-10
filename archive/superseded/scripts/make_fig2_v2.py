#!/usr/bin/env python3
"""
make_fig2_mechanism_v2.py — Fig.2: Oscillation mechanism and spatial heterogeneity.

Layout: 2×3 panels (double column, ~6.875in × 5.0in)
  Top row: time dynamics (both surface and center traces)
    (a) slab schematic with three surface fluxes
    (b) time series J(1,τ), J(0,τ), θ(1,τ), θ(0,τ)
    (c) phase portrait: surface limit cycle and center limit cycle
  Bottom row: space-time heterogeneity (kymographs)
    (d) J(ξ,τ)           — swelling field
    (e) u(ξ,τ)           — reactant depletion (inner core)
    (f) (1-φ)^m_act(ξ,τ) — accessibility closure (outer shell)
  Together: (e)+(f) = "spatially heterogeneous reaction quenching"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from dataclasses import replace

from style_pub import set_style, PRE_DOUBLE, C, COLORS, add_panel_label, save, OUT
from scan_optimized import Params, simulate

set_style()

_CACHE = os.path.join(os.path.dirname(__file__),
                      "figures_pub/fig2_v2_cache.npz")
_CACHE_VERSION = 1


# ──────────────────────────────────────────────────────────────────
# 1. Run simulation (or load cache)
# ──────────────────────────────────────────────────────────────────

def run_or_load():
    # Reuse Fig 3 cache which already has full spatial data
    cache_path = os.path.join(os.path.dirname(__file__),
                              "figures_pub/fig3_mechanism_detail_cache.npz")
    if not os.path.exists(cache_path):
        raise RuntimeError("Cache not found; run Fig 3 first.")
    d = np.load(cache_path, allow_pickle=True)
    print(f"  Using cache: {cache_path}")
    # Time window (subsampled plot arrays)
    ws = int(d["window_start"][0])
    we = int(d["window_stop"][0])
    # Extract spatial arrays on window (subsample same as _plot arrays)
    t_full = d["t"][ws:we]
    u_full = d["u"][:, ws:we]
    theta_full = d["theta"][:, ws:we]
    # Note: _plot arrays are already windowed but may have different length
    # Use full (windowed) arrays for consistency
    return {
        "x":     d["x"],
        "t":     d["t_plot"],          # already windowed, shape matches J_plot
        "J":     d["J_plot"],           # (40, 2924)
        "access": d["access_plot"],     # (40, 2924)
        # Recompute u, theta on plot window — they come from cached plot data
        "u_full": u_full,               # (40, we-ws)
        "theta_full": theta_full,       # (40, we-ws)
        "t_full":  t_full,
        "J_surf": d["J_surf_plot"],
        "J_ctr":  d["J_ctr_plot"],
        "theta_surf": d["theta_surf_plot"],
        "theta_ctr":  d["theta_ctr_plot"],
        "u_surf": d["u_surf_plot"],
        "u_ctr":  d["u_ctr_plot"],
    }


# ──────────────────────────────────────────────────────────────────
# 2. Panel (a): slab schematic
# ──────────────────────────────────────────────────────────────────

def draw_schematic(ax):
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.4, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # gel slab
    gel = FancyBboxPatch((0, 0), 1.6, 1.8,
                         boxstyle="round,pad=0.04",
                         facecolor="#c8e6fa", edgecolor="#1f77b4", linewidth=1.2)
    ax.add_patch(gel)
    ax.text(0.8, 1.1, "gel", ha="center", va="center",
            fontsize=10, color="#1f77b4", weight="bold")
    ax.text(0.8, 0.65, r"catalyst + reactant",
            ha="center", va="center", fontsize=6.5,
            color="#1f77b4", style="italic")

    # bath
    bath = FancyBboxPatch((1.82, 0), 0.88, 1.8,
                          boxstyle="round,pad=0.04",
                          facecolor="#eaf5e9", edgecolor="#2ca02c",
                          linewidth=1.0, linestyle="--")
    ax.add_patch(bath)
    ax.text(2.26, 1.60, "bath", ha="center", va="center",
            fontsize=8, color="#2ca02c")

    # symmetry center
    ax.plot([0, 0], [0, 1.8], color="#555", lw=1.0, ls="--")
    ax.text(0, -0.22, r"$\xi=0$", ha="center", va="top", fontsize=7)
    ax.text(0, -0.45, "symmetry", ha="center", va="top", fontsize=6,
            color="#555")

    # free surface
    ax.text(1.6, -0.22, r"$\xi=1$", ha="center", va="top", fontsize=7)
    ax.text(1.6, -0.45, "free surface", ha="center", va="top", fontsize=6,
            color="#555")

    # three boundary fluxes — arrows span gel/bath boundary with labels in bath area
    # Bi_mu: solvent (gel ↔ bath, drawn outward)
    ax.annotate("", xy=(2.00, 1.20), xytext=(1.50, 1.20),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4))
    ax.text(2.35, 1.20, r"$\mathrm{Bi}_\mu$", ha="left", va="center",
            fontsize=8, color="#1f77b4")
    # Bi_c: reactant (bath → gel)
    ax.annotate("", xy=(1.50, 0.90), xytext=(2.00, 0.90),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4))
    ax.text(2.35, 0.90, r"$\mathrm{Bi}_c$", ha="left", va="center",
            fontsize=8, color="#2ca02c")
    # Bi_T: heat loss (gel → bath)
    ax.annotate("", xy=(2.00, 0.40), xytext=(1.50, 0.40),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))
    ax.text(2.35, 0.40, r"$\mathrm{Bi}_T$", ha="left", va="center",
            fontsize=8, color="#d62728")


# ──────────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────────

def main():
    data = run_or_load()
    x = data["x"]
    t_plot = data["t"]              # already windowed
    J = data["J"]                   # (40, 2924)
    access = data["access"]         # (40, 2924)
    t_full = data["t_full"]         # (we-ws,)
    u_full = data["u_full"]         # (40, we-ws)
    theta_full = data["theta_full"] # (40, we-ws)

    J_surf = data["J_surf"]
    J_ctr  = data["J_ctr"]
    th_surf = data["theta_surf"]
    th_ctr  = data["theta_ctr"]

    # Time window for display (focus on 3 cycles from mid-range)
    # t_plot spans 0 to ~78
    t_start, t_end = 10.0, 78.0
    idx_plot = (t_plot >= t_start) & (t_plot <= t_end)
    t_show = t_plot[idx_plot]
    J_surf_s = J_surf[idx_plot]
    J_ctr_s  = J_ctr[idx_plot]
    th_surf_s = th_surf[idx_plot]
    th_ctr_s  = th_ctr[idx_plot]
    J_kymo = J[:, idx_plot]
    access_kymo = access[:, idx_plot]

    # For u kymograph, interpolate u_full onto t_plot window
    # (u_full has its own time axis t_full)
    # Match t_plot to nearest t_full index
    idx_full_in_plot = []
    for tp in t_show:
        idx_full_in_plot.append(np.argmin(np.abs(t_full - (tp + t_full[0] - t_plot[0]))))
    # Simpler: t_plot is mapped window of t_full, so linear interp
    # t_plot[0] corresponds to t_full[0] (since both are from same ws:we)
    # Actually t_plot is a subsample of t_full. Just align by index.
    # t_plot length 2924, t_full length (we-ws). If they differ, interpolate.
    if len(t_plot) == len(t_full):
        u_kymo = u_full[:, idx_plot]
        theta_kymo = theta_full[:, idx_plot]
    else:
        # Interpolate u_full onto t_show
        u_kymo = np.zeros((len(x), int(idx_plot.sum())))
        for i in range(len(x)):
            u_kymo[i, :] = np.interp(t_show, t_full - t_full[0] + t_plot[0], u_full[i, :])
        theta_kymo = np.zeros((len(x), int(idx_plot.sum())))
        for i in range(len(x)):
            theta_kymo[i, :] = np.interp(t_show, t_full - t_full[0] + t_plot[0], theta_full[i, :])

    # ── FIGURE ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(PRE_DOUBLE, 5.2))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.48,
                           left=0.07, right=0.97, top=0.95, bottom=0.09)

    # (a) schematic
    ax_a = fig.add_subplot(gs[0, 0])
    draw_schematic(ax_a)
    add_panel_label(ax_a, 'a')

    # (b) time series
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b2 = ax_b.twinx()
    ax_b.plot(t_show, J_surf_s, '-', color='#1f77b4', lw=1.1,
              label=r'$J(1)$')
    ax_b.plot(t_show, J_ctr_s,  ':', color='#1f77b4', lw=1.0,
              label=r'$J(0)$')
    ax_b2.plot(t_show, th_surf_s, '-', color='#d62728', lw=1.1,
               label=r'$\theta(1)$')
    ax_b2.plot(t_show, th_ctr_s,  ':', color='#d62728', lw=1.0,
               label=r'$\theta(0)$')
    ax_b.set_xlabel(r'$\tau$')
    ax_b.set_ylabel(r'$J$', color='#1f77b4')
    ax_b2.set_ylabel(r'$\theta$', color='#d62728', rotation=270, labelpad=10)
    ax_b.tick_params(axis='y', colors='#1f77b4')
    ax_b2.tick_params(axis='y', colors='#d62728')
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, fontsize=5.5, loc='upper right',
                framealpha=0.9, ncol=2)
    add_panel_label(ax_b, 'b')

    # (c) phase portrait
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(J_surf_s, th_surf_s, '-', color='#1f77b4', lw=0.9,
              alpha=0.85, label=r'surface')
    ax_c.plot(J_ctr_s, th_ctr_s, '-', color='#ff7f0e', lw=0.9,
              alpha=0.85, label=r'center')
    # Mark 4 phases on surface cycle
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(th_surf_s, prominence=0.5, distance=50)
    troughs, _ = find_peaks(-th_surf_s, prominence=0.5, distance=50)
    if len(peaks) >= 2 and len(troughs) >= 2:
        i_peak = peaks[0]
        trough_after = troughs[troughs > i_peak]
        if len(trough_after) > 0:
            i_trough = trough_after[0]
            i_peak2 = peaks[1] if len(peaks) > 1 else len(t_show) - 1
            i_ignite  = max(0, i_peak - 15)
            i_collapse = (i_peak + i_trough) // 2
            i_cool = i_trough
            i_swell = (i_trough + i_peak2) // 2
            pts = [
                ('① swell',    i_swell,    ( 5,  -8)),
                ('② ignite',   i_ignite,   ( 5,   8)),
                ('③ collapse', i_collapse, (-50,  0)),
                ('④ cool',     i_cool,     ( 5,  -8)),
            ]
            for lab, ipt, off in pts:
                ax_c.plot(J_surf_s[ipt], th_surf_s[ipt], 'ko', ms=4, zorder=5)
                ax_c.annotate(lab, (J_surf_s[ipt], th_surf_s[ipt]),
                              fontsize=5.5, ha='left', va='center',
                              xytext=off, textcoords='offset points')
    ax_c.set_xlabel(r'$J$')
    ax_c.set_ylabel(r'$\theta$')
    ax_c.legend(fontsize=6, loc='lower right', framealpha=0.9)
    add_panel_label(ax_c, 'c')

    # (d) J kymograph
    ax_d = fig.add_subplot(gs[1, 0])
    im = ax_d.imshow(J_kymo, origin='lower', aspect='auto',
                     extent=[t_start, t_end, 0, 1],
                     cmap='RdBu_r', vmin=0.2, vmax=2.5, rasterized=True)
    plt.colorbar(im, ax=ax_d, pad=0.02, fraction=0.045, label=r'$J$')
    ax_d.set_xlabel(r'$\tau$')
    ax_d.set_ylabel(r'$\xi$')
    add_panel_label(ax_d, 'd')

    # (e) u kymograph — reactant depletion in inner core (log scale!)
    ax_e = fig.add_subplot(gs[1, 1])
    from matplotlib.colors import LogNorm
    u_kymo_safe = np.clip(u_kymo, 1e-6, 1.0)
    im = ax_e.imshow(u_kymo_safe, origin='lower', aspect='auto',
                     extent=[t_start, t_end, 0, 1],
                     cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=1),
                     rasterized=True)
    plt.colorbar(im, ax=ax_e, pad=0.02, fraction=0.045, label=r'$u$')
    ax_e.set_xlabel(r'$\tau$')
    ax_e.set_ylabel(r'$\xi$')
    add_panel_label(ax_e, 'e')

    # (f) accessibility kymograph — closure in outer shell
    ax_f = fig.add_subplot(gs[1, 2])
    im = ax_f.imshow(access_kymo, origin='lower', aspect='auto',
                     extent=[t_start, t_end, 0, 1],
                     cmap='magma', vmin=0, vmax=1, rasterized=True)
    plt.colorbar(im, ax=ax_f, pad=0.02, fraction=0.045,
                 label=r'$(1-\phi)^{m_{\rm act}}$')
    ax_f.set_xlabel(r'$\tau$')
    ax_f.set_ylabel(r'$\xi$')
    add_panel_label(ax_f, 'f')

    out_pdf = os.path.join(OUT, "fig2_mechanism.pdf")
    out_png = os.path.join(OUT, "fig2_mechanism.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf}")


if __name__ == "__main__":
    main()
