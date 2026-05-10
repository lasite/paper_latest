#!/usr/bin/env python3
"""
explore_bulk_hopf.py — Stage-0 exploratory scan for the Fig 4 plan.

The default fig4 grid (chi_inf=0.60, Bi_T ∈ [0.035, 0.40], S_χ ∈ [0, 2.1])
contains no `bulk_hopf` cells: every oscillating cell crosses the LCST.
That can mean either (a) we just haven't scanned the right corner, or
(b) the model has no bulk-Hopf sub-region at all (Hopf onset always
recruits the LCST collapse).

This script runs two targeted scans designed to find bulk Hopf if it
exists by *moving the LCST fold further away* via a smaller chi_inf:

  Scan A: (Bi_T, S_chi) with chi_inf=0.40
          — raises the temperature needed to reach chi=0.5 trigger.
  Scan B: (S_chi, Da) with chi_inf=0.40, Bi_T=0.06
          — explicit (S_chi, Da) sweep at the low-Bi_T column where
            oscillation is strongest in fig4(d).

Outputs:
  data/fig4/explore_bulk_hopf_<axes>.npz
  Figure/pub/explore_bulk_hopf.png

The PNG combines regime maps with max-φ heatmaps so the two diagnostics
can be eyeballed together.
"""
import os
import sys
from pathlib import Path

# Pin BLAS *before* numpy import, mirror fig4_data.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT
from fig4_data import (classify_point, REGIME_NAMES,
                        REG_FAILED, REG_STEADY_COLD, REG_BULK_HOPF,
                        REG_LCST_FRONT, REG_GLOBAL_COLLAPSE,
                        REG_STEADY_COLLAPSED, REG_STEADY_FRONT,
                        N_WORKERS_DEFAULT)

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig4").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


REG_COLORS = {
    REG_FAILED:           "#888888",
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
    REG_FAILED:           "failed",
}


def _worker(task):
    """Picklable worker for one grid point."""
    j, i, p_dict = task
    r = classify_point(p_dict)
    return j, i, r


def run_grid(name, base_overrides, param_x, x_vals, param_y, y_vals,
             n_workers=None, force=False):
    """Run a 2D grid scan and cache to a uniquely-named npz."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"explore_bulk_hopf_{name}.npz"
    if cache.exists() and not force:
        print(f"  Using cache: {cache}")
        z = np.load(cache, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        out.update(dict(param_x=param_x, param_y=param_y))
        return out

    NX, NY = len(x_vals), len(y_vals)
    regime      = np.full((NY, NX), REG_FAILED, dtype=int)
    J_amp_max   = np.full((NY, NX), np.nan)
    surf_amp    = np.full((NY, NX), np.nan)
    phi_max     = np.full((NY, NX), np.nan)
    phi_max_min = np.full((NY, NX), np.nan)
    period      = np.full((NY, NX), np.nan)

    base = dict(WORKING_POINT)
    base.update(base_overrides or {})

    tasks = []
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            p = dict(base)
            p[param_x] = float(xv)
            p[param_y] = float(yv)
            tasks.append((j, i, p))

    if n_workers is None:
        n_workers = N_WORKERS_DEFAULT
    n_workers = max(1, min(int(n_workers), len(tasks)))

    print(f"  [{name}] grid {param_x}×{param_y} = {NX}×{NY} = {len(tasks)} pts; "
          f"overrides={base_overrides}; workers={n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            j, i, r = f.result()
            regime[j, i]      = r["regime"]
            J_amp_max[j, i]   = r["J_amp_max"]
            surf_amp[j, i]    = r["surf_amp"]
            phi_max[j, i]     = r["phi_max"]
            phi_max_min[j, i] = r["phi_max_min"]
            period[j, i]      = r["period"]
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"    [{done:>3}/{len(tasks)}]")

    np.savez_compressed(cache, x=x_vals, y=y_vals,
                        regime=regime, J_amp_max=J_amp_max,
                        surf_amp=surf_amp, phi_max=phi_max,
                        phi_max_min=phi_max_min, period=period)
    print(f"  Saved: {cache}")
    out = dict(x=x_vals, y=y_vals, regime=regime,
               J_amp_max=J_amp_max, surf_amp=surf_amp,
               phi_max=phi_max, phi_max_min=phi_max_min, period=period,
               param_x=param_x, param_y=param_y)
    return out


def _regime_pcm(ax, x, y, reg, x_log=False, y_log=False):
    codes = sorted(REG_COLORS.keys())
    cmap = ListedColormap([REG_COLORS[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    ax.pcolormesh(x, y, reg, cmap=cmap, norm=norm, shading="nearest",
                  rasterized=True)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")


def _phi_heat(ax, x, y, phi, x_log=False, y_log=False):
    pcm = ax.pcolormesh(x, y, phi, cmap="cividis",
                        norm=Normalize(vmin=0.10, vmax=1.0),
                        shading="nearest", rasterized=True)
    ax.contour(x, y, phi, levels=[0.5],
               colors="white", linewidths=1.0, zorder=5)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    return pcm


def main():
    # ── Scan A: (Bi_T, S_chi) at chi_inf=0.40 ─────────────────────────
    BI_T_VALS_A  = np.geomspace(0.035, 0.40, 12)
    S_CHI_VALS_A = np.linspace(0.0, 2.1, 12)
    grid_A = run_grid(
        "A_BiT_Schi_chiinf040",
        base_overrides=dict(chi_inf=0.40),
        param_x="Bi_T", x_vals=BI_T_VALS_A,
        param_y="S_chi", y_vals=S_CHI_VALS_A,
    )

    # ── Scan B: (S_chi, Da) at Bi_T=0.06, chi_inf=0.40 ────────────────
    S_CHI_VALS_B = np.linspace(0.0, 2.1, 10)
    DA_VALS_B    = np.geomspace(0.5, 30.0, 10)
    grid_B = run_grid(
        "B_Schi_Da_chiinf040",
        base_overrides=dict(chi_inf=0.40, Bi_T=0.06),
        param_x="S_chi", x_vals=S_CHI_VALS_B,
        param_y="Da", y_vals=DA_VALS_B,
    )

    # ── Compose figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(PRE_DOUBLE, 4.6))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.30,
                           left=0.07, right=0.95, top=0.92, bottom=0.13)

    # Row 1: regime maps
    ax_a1 = fig.add_subplot(gs[0, 0])
    _regime_pcm(ax_a1, grid_A["x"], grid_A["y"], grid_A["regime"], x_log=True)
    ax_a1.scatter([WORKING_POINT["Bi_T"]], [WORKING_POINT["S_chi"]],
                  s=46, marker="*", color="white", edgecolor="k",
                  linewidth=0.8, zorder=8)
    ax_a1.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=8)
    ax_a1.set_ylabel(r"$S_\chi$", fontsize=8)
    ax_a1.tick_params(labelsize=6.5)
    ax_a1.text(0.02, 0.98, r"$\chi_\infty=0.40$", transform=ax_a1.transAxes,
               ha="left", va="top", fontsize=6.5,
               bbox=dict(facecolor="white", edgecolor="0.5",
                         boxstyle="round,pad=0.15", lw=0.4))
    add_panel_label(ax_a1, "a")

    ax_a2 = fig.add_subplot(gs[0, 1])
    _regime_pcm(ax_a2, grid_B["x"], grid_B["y"], grid_B["regime"], y_log=True)
    ax_a2.scatter([WORKING_POINT["S_chi"]], [WORKING_POINT["Da"]],
                  s=46, marker="*", color="white", edgecolor="k",
                  linewidth=0.8, zorder=8)
    ax_a2.set_xlabel(r"$S_\chi$", fontsize=8)
    ax_a2.set_ylabel(r"$\mathrm{Da}$", fontsize=8)
    ax_a2.tick_params(labelsize=6.5)
    ax_a2.text(0.02, 0.98, r"$\chi_\infty=0.40,\ \mathrm{Bi}_T=0.06$",
               transform=ax_a2.transAxes, ha="left", va="top", fontsize=6.5,
               bbox=dict(facecolor="white", edgecolor="0.5",
                         boxstyle="round,pad=0.15", lw=0.4))
    add_panel_label(ax_a2, "b")

    # Row 2: phi_max with LCST contour
    ax_b1 = fig.add_subplot(gs[1, 0])
    pcm1 = _phi_heat(ax_b1, grid_A["x"], grid_A["y"], grid_A["phi_max"],
                     x_log=True)
    ax_b1.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=8)
    ax_b1.set_ylabel(r"$S_\chi$", fontsize=8)
    ax_b1.tick_params(labelsize=6.5)
    cb1 = fig.colorbar(pcm1, ax=ax_b1, fraction=0.045, pad=0.02)
    cb1.set_label(r"$\max\,\varphi$", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    add_panel_label(ax_b1, "c")

    ax_b2 = fig.add_subplot(gs[1, 1])
    pcm2 = _phi_heat(ax_b2, grid_B["x"], grid_B["y"], grid_B["phi_max"],
                     y_log=True)
    ax_b2.set_xlabel(r"$S_\chi$", fontsize=8)
    ax_b2.set_ylabel(r"$\mathrm{Da}$", fontsize=8)
    ax_b2.tick_params(labelsize=6.5)
    cb2 = fig.colorbar(pcm2, ax=ax_b2, fraction=0.045, pad=0.02)
    cb2.set_label(r"$\max\,\varphi$", fontsize=7)
    cb2.ax.tick_params(labelsize=6)
    add_panel_label(ax_b2, "d")

    # Legend
    present = sorted(set(grid_A["regime"].flatten().tolist())
                     | set(grid_B["regime"].flatten().tolist()))
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=REG_COLORS[int(c)],
                             edgecolor="0.4",
                             label=REG_LABELS[int(c)])
               for c in present if int(c) != REG_FAILED]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.00), ncol=len(handles),
               fontsize=6, framealpha=0.9, handlelength=1.0,
               borderpad=0.3, columnspacing=1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "explore_bulk_hopf.pdf"
    png = OUT_DIR / "explore_bulk_hopf.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")

    # ── Numerical summary ─────────────────────────────────────────────
    def summarize(name, g):
        codes, counts = np.unique(g["regime"], return_counts=True)
        rows = [f"{REGIME_NAMES.get(int(c), '?'):>20} : {int(n):>3}"
                for c, n in zip(codes, counts)]
        print(f"\n  {name}:")
        for r in rows:
            print(f"    {r}")
    summarize("Scan A (Bi_T × S_chi @ chi_inf=0.40)", grid_A)
    summarize("Scan B (S_chi × Da @ Bi_T=0.06, chi_inf=0.40)", grid_B)


if __name__ == "__main__":
    main()
