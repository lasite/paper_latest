#!/usr/bin/env python3
"""
make_fig6_manifold.py — Fig 6 (paper): attractor manifold portrait.

PLAN-aligned 4-panel layout (2×2):
  (a) Schematic state-space topology of the four PDE attractors.
  (b) Late-time spatial profile J(ξ) for each attractor.
  (c) D1 — refined basin of attraction at the C-cell (11×11 IC grid in
      (J_0, θ_0), classified by long-time attractor).
  (d) D2 — IC-induced hysteresis: terminal ⟨J⟩ vs initial thermal
      pulse θ_0 for cold and pre-collapsed seeds.

The earlier 5×5 streamline-ensemble panel is replaced by the denser
basin map (panel c), and the hysteresis curve is folded in from the
formerly standalone hysteresis_C_theta0 figure.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label

set_style()

DATA_DIR = _HERE.parent / "data" / "fig4"
DATA_DIR_FIG6 = _HERE.parent / "data" / "fig6"
OUT_DIR  = _HERE.parent / "Figure" / "pub"

PRE_DOUBLE = 6.875

LBL_FS  = 8
TICK_FS = 6.5
LEG_FS  = 6.5
ANNO_FS = 6.0

COLOR = {
    "cold":         "#3676b8",
    "lcst_front":   "#c0392b",
    "frozen_front": "#d49a3f",
    "hot_runaway":  "#7c3aa8",
}
LABEL = {
    "cold":         "cold-swollen",
    "lcst_front":   "LCST-front cycle",
    "frozen_front": "frozen front",
    "hot_runaway":  "hot-runaway",
}


def load_repr(name):
    z = np.load(DATA_DIR / "fig4_repr.npz", allow_pickle=True)
    return dict(
        t=z[f"{name}__t"], x=z[f"{name}__x"],
        J=z[f"{name}__J"], theta=z[f"{name}__theta"],
    )


def load_demo():
    z = np.load(DATA_DIR / "hard_excitation_demo.npz", allow_pickle=True)
    cold = dict(t=z["t_A"], x=z["x_A"], J=z["J_A"], theta=z["theta_A"])
    hot  = dict(t=z["t_B"], x=z["x_B"], J=z["J_B"], theta=z["theta_B"])
    return cold, hot, dict(Bi_T=float(z["Bi_T"]), S_chi=float(z["S_chi"]))


# ── Panel (a): schematic topology ─────────────────────────────────────
def panel_schematic(ax):
    """Clean abstract-state-space schematic. Markers + minimal labels +
    one connecting arrow + one basin separator. No backdrop colors."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.75")
        s.set_linewidth(0.5)

    # Marker positions
    pos = {
        "cold":         (0.18, 0.22),
        "lcst_front":   (0.50, 0.50),
        "frozen_front": (0.18, 0.78),
        "hot_runaway":  (0.82, 0.78),
    }

    # Cold
    ax.plot(*pos["cold"], "o", color=COLOR["cold"], ms=11,
            mec="k", mew=0.6, zorder=5)
    ax.text(pos["cold"][0], pos["cold"][1] - 0.08,
            "cold-swollen", ha="center", va="top",
            fontsize=ANNO_FS, fontweight="bold", color=COLOR["cold"])
    ax.text(pos["cold"][0], pos["cold"][1] - 0.16,
            r"low $S_\chi$",
            ha="center", va="top", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")

    # LCST-front cycle
    th = np.linspace(0, 2*np.pi, 100)
    cyc_a, cyc_b = 0.10, 0.12
    cx, cy = pos["lcst_front"]
    ax.plot(cx + cyc_a*np.cos(th), cy + cyc_b*np.sin(th),
            "-", color=COLOR["lcst_front"], lw=1.7, zorder=4)
    ax.annotate("", xy=(cx + cyc_a*np.cos(np.pi/2 + 0.18),
                        cy + cyc_b*np.sin(np.pi/2 + 0.18)),
                xytext=(cx + cyc_a*np.cos(np.pi/2 - 0.04),
                        cy + cyc_b*np.sin(np.pi/2 - 0.04)),
                arrowprops=dict(arrowstyle="->", color=COLOR["lcst_front"],
                                lw=1.4))
    ax.text(cx, cy - cyc_b - 0.04, "LCST-front cycle",
            ha="center", va="top", fontsize=ANNO_FS,
            fontweight="bold", color=COLOR["lcst_front"])
    ax.text(cx, cy - cyc_b - 0.11, r"central regime",
            ha="center", va="top", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")

    # Frozen front
    ax.plot(*pos["frozen_front"], "s", color=COLOR["frozen_front"],
            ms=11, mec="k", mew=0.6, zorder=5)
    ax.text(pos["frozen_front"][0], pos["frozen_front"][1] + 0.08,
            "frozen front", ha="center", va="bottom",
            fontsize=ANNO_FS, fontweight="bold",
            color=COLOR["frozen_front"])
    ax.text(pos["frozen_front"][0], pos["frozen_front"][1] + 0.15,
            r"high $\mathrm{Bi}_T$",
            ha="center", va="bottom", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")

    # Hot-runaway
    ax.plot(*pos["hot_runaway"], "*", color=COLOR["hot_runaway"],
            ms=15, mec="k", mew=0.6, zorder=5)
    ax.text(pos["hot_runaway"][0], pos["hot_runaway"][1] + 0.08,
            "hot-runaway", ha="center", va="bottom",
            fontsize=ANNO_FS, fontweight="bold",
            color=COLOR["hot_runaway"])
    ax.text(pos["hot_runaway"][0], pos["hot_runaway"][1] + 0.15,
            r"IC-dependent",
            ha="center", va="bottom", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")

    # Connection: cycle → frozen front
    arr1 = FancyArrowPatch(
        (cx - cyc_a - 0.02, cy + 0.02),
        (pos["frozen_front"][0] + 0.06, pos["frozen_front"][1] - 0.07),
        arrowstyle="->", color="0.45",
        connectionstyle="arc3,rad=-0.25", lw=0.9, zorder=3,
    )
    ax.add_patch(arr1)
    ax.text(0.29, 0.62, r"$\mathrm{Bi}_T\!\uparrow$",
            fontsize=ANNO_FS, color="0.40", style="italic")

    # Basin separator: cycle ↔ hot-runaway
    sep_x = np.linspace(0.65, 0.75, 60)
    sep_y = 0.55 + 0.20 * (sep_x - 0.65) / 0.10
    ax.plot(sep_x, sep_y, "--", color="0.50", lw=1.2, zorder=2)
    ax.text(0.71, 0.50, "basin",
            ha="center", va="top", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")
    ax.text(0.71, 0.45, "separator",
            ha="center", va="top", fontsize=ANNO_FS - 0.3,
            color="0.40", style="italic")


# ── Panel (b): late-time spatial profile J(ξ) ─────────────────────────
def panel_spatial_profile(ax, runs):
    """Plot J(ξ) at late time for each attractor. For LCST-front cycle,
    show the J_min(ξ) and J_max(ξ) envelope across the late window."""
    for key in ["cold", "lcst_front", "frozen_front", "hot_runaway"]:
        d = runs[key]
        x = d["x"]; J = d["J"]; t = d["t"]
        c = COLOR[key]

        if key == "lcst_front":
            # Envelope across last 40% of trajectory
            i0 = int(0.60 * len(t))
            J_late = J[:, i0:]
            J_min = J_late.min(axis=1)
            J_max = J_late.max(axis=1)
            ax.fill_between(x, J_min, J_max, color=c, alpha=0.25, zorder=2)
            ax.plot(x, J_min, "-", color=c, lw=0.8, alpha=0.7, zorder=3)
            ax.plot(x, J_max, "-", color=c, lw=0.8, alpha=0.7, zorder=3)
            # Mid-cycle profile
            ax.plot(x, J_late.mean(axis=1), "-", color=c, lw=1.4,
                    label=LABEL[key], zorder=4)
        else:
            # Use the late-time profile (last time step)
            ax.plot(x, J[:, -1], "-", color=c, lw=1.4,
                    label=LABEL[key], zorder=4)

    # LCST contour line
    phi_p0 = 0.15
    LCST = 0.5
    J_LCST = phi_p0 / LCST  # = 0.3
    ax.axhline(J_LCST, color="0.45", lw=0.7, ls=":", zorder=1)
    ax.text(0.02, J_LCST * 1.15, r"$\varphi=0.5$ (LCST)",
            fontsize=ANNO_FS, color="0.40", va="bottom")

    ax.set_xlabel(r"$\xi$", fontsize=LBL_FS)
    ax.set_ylabel(r"$J(\xi)$", fontsize=LBL_FS)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.10, 6.5)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    ax.legend(loc="lower left", fontsize=LEG_FS, framealpha=0.9,
              handlelength=1.4, borderpad=0.4, ncol=1)
    ax.grid(True, alpha=0.25, lw=0.4)


# ── Panel (c): refined basin (D1) at the C-cell ───────────────────────
def panel_basin_dense(ax):
    """Refined basin map at the C-cell (11×11 IC grid)."""
    cache = DATA_DIR_FIG6 / "basin_C_dense.npz"
    z = np.load(cache)
    J0_vals     = z["J0_vals"]
    theta0_vals = z["theta0_vals"]
    Jm_term     = z["Jm_term"]
    BiT  = float(z["cell_Bi_T"])
    Sx   = float(z["cell_S_chi"])

    cls = np.full_like(Jm_term, -1, dtype=np.int8)
    cls[(Jm_term < 1.0) & np.isfinite(Jm_term)] = 0
    cls[(Jm_term >= 1.0) & (Jm_term < 3.0)] = 1
    cls[Jm_term >= 3.0] = 2

    C_FROZEN = COLOR["frozen_front"]
    C_CYCLE  = COLOR["lcst_front"]
    C_HOT    = COLOR["hot_runaway"]
    cmap = ListedColormap(["#dddddd", C_FROZEN, C_CYCLE, C_HOT])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(J0_vals, theta0_vals, cls, cmap=cmap, norm=norm,
                  shading="auto", zorder=1)

    # D2 seed markers
    ax.plot(1.30, 0.0, "*", color="white", mec="k", mew=0.7,
            ms=10, zorder=10)
    ax.plot(0.20, 0.0, "^", color="white", mec="k", mew=0.7,
            ms=7, zorder=10)
    ax.text(1.30, 0.4, "cold seed", fontsize=5.5, color="k",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5,
                      alpha=0.85))
    ax.text(0.20, 0.4, "pre-collapsed seed", fontsize=5.5, color="k",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5,
                      alpha=0.85))

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_FROZEN, ec="0.4",
                      label="frozen / partial"),
        plt.Rectangle((0, 0), 1, 1, fc=C_CYCLE, ec="0.4",
                      label="LCST-front cycle"),
        plt.Rectangle((0, 0), 1, 1, fc=C_HOT, ec="0.4",
                      label="hot-runaway"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.8,
              framealpha=0.95, handlelength=1.2, borderpad=0.3)

    ax.set_xlabel(r"$J_0$ (initial swelling)", fontsize=LBL_FS)
    ax.set_ylabel(r"$\theta_0$ (initial thermal pulse)", fontsize=LBL_FS)
    ax.set_xlim(J0_vals.min(), J0_vals.max())
    ax.set_ylim(theta0_vals.min(), theta0_vals.max())
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.5)
    ax.text(0.02, 0.98,
            rf"C-cell: $\mathrm{{Bi}}_T={BiT:.3f}$, $S_\chi={Sx:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=ANNO_FS,
            bbox=dict(facecolor="white", edgecolor="0.5",
                      boxstyle="round,pad=0.2", lw=0.4))


# ── Panel (d): IC-induced hysteresis (D2) ─────────────────────────────
def panel_hysteresis(ax):
    """θ_0 sweep at the C-cell: terminal ⟨J⟩ for cold vs pre-collapsed
    seeds; a sharp jump at the cold-seed basin threshold is the
    hysteresis signature."""
    cache = DATA_DIR_FIG6 / "hysteresis_C_theta0.npz"
    z = np.load(cache)
    theta0 = z["axis_values"]
    Jm_cold = z["Jm_term_cold"]
    Jm_hot  = z["Jm_term_hot"]
    s_cold = z["success_cold"]
    s_hot  = z["success_hot"]
    BiT = float(z["cell_Bi_T"])
    Sx  = float(z["cell_S_chi"])
    J0_cold = float(z["J0_cold"])
    J0_hot  = float(z["J0_hot"])

    msk_c = s_cold.astype(bool)
    msk_h = s_hot.astype(bool)

    ax.plot(theta0[msk_c], Jm_cold[msk_c], "o-",
            color=COLOR["lcst_front"], ms=4.5, lw=1.0,
            mec="k", mew=0.4, zorder=4,
            label=rf"cold seed ($J_0={J0_cold:.2f}$)")
    ax.plot(theta0[msk_h], Jm_hot[msk_h], "s-",
            color=COLOR["hot_runaway"], ms=4.5, lw=1.0,
            mec="k", mew=0.4, zorder=4,
            label=rf"pre-collapsed seed ($J_0={J0_hot:.2f}$)")

    # Threshold markers: for the cold seed find the largest theta0 jump
    if msk_c.sum() >= 4:
        Jc = Jm_cold[msk_c]
        th_c = theta0[msk_c]
        d = np.diff(Jc)
        i_jump = int(np.argmax(np.abs(d)))
        th_star = 0.5 * (th_c[i_jump] + th_c[i_jump + 1])
        ax.axvline(th_star, color="0.4", lw=0.7, ls=":")
        ax.text(th_star, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 6.0,
                rf"  $\theta_0^*\!\approx\!{th_star:.2f}$",
                fontsize=ANNO_FS, color="0.3",
                ha="left", va="top")

    ax.set_xlabel(r"$\theta_0$ (initial thermal pulse)", fontsize=LBL_FS)
    ax.set_ylabel(r"$\langle J\rangle_{\rm late}$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.5)
    ax.legend(loc="center right", fontsize=LEG_FS, framealpha=0.95,
              handlelength=1.4, borderpad=0.3)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.text(0.02, 0.98,
            rf"C-cell ($\mathrm{{Bi}}_T={BiT:.3f}$, $S_\chi={Sx:.2f}$)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=ANNO_FS,
            bbox=dict(facecolor="white", edgecolor="0.5",
                      boxstyle="round,pad=0.2", lw=0.4))


# ── (legacy, kept for backwards compatibility callers) ───────────────
def panel_bistability(ax, cold_demo, hot_demo, c_meta):
    """Streamline-style ensemble of trajectories from a 5×5 IC grid at
    the C-cell, projected to (⟨J⟩, ⟨θ⟩). Falls back to the 2-IC
    visualization if the streamline cache is absent."""
    cache = DATA_DIR / "ic_streamlines_C.npz"
    if not cache.exists():
        # Fallback: original 2-IC visualization
        return _panel_c_two_ic_fallback(ax, cold_demo, hot_demo, c_meta)

    z = np.load(cache, allow_pickle=False)
    Jm_grid    = z["Jm_grid"]
    thm_grid   = z["thm_grid"]
    Jm_term    = z["Jm_term"]
    thm_term   = z["thm_term"]
    success    = z["success"]
    NT, NJ = success.shape

    # Classify each terminal: LCST-front cycle (low Jm_term) vs hot-runaway
    # (high Jm_term). Boundary at Jm_term ~ 3 separates them empirically.
    cls_cycle  = (Jm_term < 3.0)
    cls_hot    = (Jm_term >= 3.0)
    color_cycle = COLOR["lcst_front"]
    color_hot   = COLOR["hot_runaway"]

    # Plot all 25 trajectories
    for j in range(NT):
        for i in range(NJ):
            if not success[j, i]:
                continue
            Jm  = Jm_grid[j, i]
            thm = thm_grid[j, i]
            if not np.isfinite(Jm).all():
                continue
            c = color_cycle if cls_cycle[j, i] else color_hot
            ax.plot(Jm, thm, "-", color=c, lw=0.5, alpha=0.55,
                    zorder=2)
            # IC start marker
            ax.plot(Jm[0], thm[0], "x", color=c, ms=4,
                    mew=0.9, alpha=0.65, zorder=3)

    # Overlay terminal-state markers
    if cls_cycle.any():
        ax.scatter(Jm_term[cls_cycle], thm_term[cls_cycle],
                   marker="o", s=28, facecolor=color_cycle,
                   edgecolor="k", linewidth=0.5, zorder=6)
    if cls_hot.any():
        ax.scatter(Jm_term[cls_hot], thm_term[cls_hot],
                   marker="o", s=28, facecolor=color_hot,
                   edgecolor="k", linewidth=0.5, zorder=6)

    # Legend handles (one per attractor)
    ax.plot([], [], "-", color=color_cycle, lw=1.0,
            label="→ LCST-front cycle")
    ax.plot([], [], "-", color=color_hot, lw=1.0,
            label="→ hot-runaway")

    ax.set_xlabel(r"$\langle J\rangle$", fontsize=LBL_FS)
    ax.set_ylabel(r"$\langle\theta\rangle$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    # Parameter chip in mid-right (free space between cycle and hot-runaway)
    ax.text(0.98, 0.50,
            rf"C-cell"
            "\n"
            rf"$\mathrm{{Bi}}_T={c_meta['Bi_T']:.3f}$"
            "\n"
            rf"$S_\chi={c_meta['S_chi']:.2f}$",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=ANNO_FS,
            bbox=dict(facecolor="white", edgecolor="0.5",
                      boxstyle="round,pad=0.25", lw=0.4))
    ax.legend(loc="lower right", fontsize=LEG_FS, framealpha=0.9,
              handlelength=1.4, borderpad=0.4)
    ax.grid(True, alpha=0.25, lw=0.4)


def _panel_c_two_ic_fallback(ax, cold_demo, hot_demo, c_meta):
    """Fallback 2-IC visualization if the streamline cache is absent."""
    for d, lbl, c in [(cold_demo, "cold IC",       "#3676b8"),
                      (hot_demo,  "collapsed IC",  "#7c3aa8")]:
        Jm = d["J"].mean(axis=0)
        thm = d["theta"].mean(axis=0)
        ax.plot(Jm, thm, "-", color=c, lw=0.9, alpha=0.85, label=lbl)
        ax.plot(Jm[-1], thm[-1], "o", color=c, ms=8,
                mec="k", mew=0.6, zorder=6)
        ax.plot(Jm[0], thm[0], "x", color=c, ms=6,
                mew=1.0, alpha=0.7, zorder=5)
    ax.annotate("→ LCST-front\ncycle",
                xy=(cold_demo["J"].mean(axis=0)[-1],
                    cold_demo["theta"].mean(axis=0)[-1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=ANNO_FS, color="#c0392b",
                ha="left", va="bottom")
    ax.annotate("→ hot-runaway",
                xy=(hot_demo["J"].mean(axis=0)[-1],
                    hot_demo["theta"].mean(axis=0)[-1]),
                xytext=(-12, -16), textcoords="offset points",
                fontsize=ANNO_FS, color=COLOR["hot_runaway"],
                ha="right", va="top")
    ax.set_xlabel(r"$\langle J\rangle$", fontsize=LBL_FS)
    ax.set_ylabel(r"$\langle\theta\rangle$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    ax.legend(loc="lower right", fontsize=LEG_FS, framealpha=0.9,
              handlelength=1.4, borderpad=0.4)
    ax.set_title(rf"C-cell: $\mathrm{{Bi}}_T={c_meta['Bi_T']:.3f}$, "
                 rf"$S_\chi={c_meta['S_chi']:.2f}$",
                 fontsize=LBL_FS)
    ax.grid(True, alpha=0.25, lw=0.4)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    runs = {}
    runs["cold"]         = load_repr("steady_cold")
    runs["lcst_front"]   = load_repr("lcst_front_WP")
    runs["frozen_front"] = load_repr("steady_front")
    cold_demo, hot_demo, c_meta = load_demo()
    runs["hot_runaway"]  = hot_demo

    fig = plt.figure(figsize=(PRE_DOUBLE, 5.0))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.32, hspace=0.40,
        left=0.07, right=0.98, top=0.95, bottom=0.10,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    panel_schematic(ax_a)
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_spatial_profile(ax_b, runs)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    panel_basin_dense(ax_c)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    panel_hysteresis(ax_d)
    add_panel_label(ax_d, "d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig6.pdf"
    png = OUT_DIR / "fig6.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


if __name__ == "__main__":
    main()
