#!/usr/bin/env python3
"""
make_hysteresis.py — D2 renderer.

Reads scan_hysteresis_C.py's cache and produces the IC-induced
bistability panel for §IV.D: late-time mean swelling ratio vs the
swept initial condition (theta_0 by default), with one curve per
seed (cold-IC vs pre-collapsed-IC).  The split between the two
curves over the bistable interval is the falsifiable prediction.

Cache-only renderer; cheap to re-run.
Output: figures_pub/hysteresis_C_<axis>.{pdf,png}

Usage:
  python make_hysteresis.py                # default theta_0 axis
  python make_hysteresis.py --axis u0      # if a u0 sweep cache exists
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from style_pub import set_style, PRE_DOUBLE, add_panel_label, save, C
set_style()

DATA_DIR = _HERE.parent / "data" / "fig6"

C_CYCLE   = "#c0392b"   # LCST-front cycle color (matches Fig.6)
C_RUNAWAY = "#7c3aa8"   # hot-runaway color (matches Fig.6)


def load_cache(axis: str):
    path = DATA_DIR / f"hysteresis_C_{axis}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run scan_hysteresis_C.py --axis {axis} first."
        )
    return np.load(path, allow_pickle=True), path


def split_branches(Jm_term):
    """Return boolean masks (cycle, runaway) using the Jm_term ~ 3 split."""
    cycle = Jm_term < 3.0
    runaway = ~cycle & np.isfinite(Jm_term)
    return cycle, runaway


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["theta0", "u0"], default="theta0")
    args = ap.parse_args()

    z, src = load_cache(args.axis)
    axis_vals  = z["axis_values"]
    Jc_term    = z["Jm_term_cold"]
    thc_term   = z["thm_term_cold"]
    Jh_term    = z["Jm_term_hot"]
    thh_term   = z["thm_term_hot"]
    succ_c     = z["success_cold"]
    succ_h     = z["success_hot"]
    Bi_T       = float(z["cell_Bi_T"])
    S_chi      = float(z["cell_S_chi"])
    J0_cold    = float(z["J0_cold"])
    J0_hot     = float(z["J0_hot"])
    u0_fixed   = float(z["u0"])

    cyc_c, run_c = split_branches(Jc_term)
    cyc_h, run_h = split_branches(Jh_term)

    print(f"  Cache: {src}")
    print(f"  axis = {args.axis} from {axis_vals[0]:.3f} to {axis_vals[-1]:.3f}")
    print(f"  cold-seed  cycle/runaway counts: {cyc_c.sum()} / {run_c.sum()}")
    print(f"  hot-seed   cycle/runaway counts: {cyc_h.sum()} / {run_h.sum()}")

    # Locate the cold-seed transition (jump from cycle → runaway)
    th_jump = None
    for k in range(len(axis_vals) - 1):
        if cyc_c[k] and run_c[k + 1]:
            th_jump = 0.5 * (axis_vals[k] + axis_vals[k + 1])
            break

    # ── Two-panel figure ─────────────────────────────────────────────
    # (a) <J>_late vs axis — the hysteresis curve
    # (b) <theta>_late vs axis — secondary projection
    fig = plt.figure(figsize=(0.95 * PRE_DOUBLE, 2.9))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                            left=0.09, right=0.97, top=0.92, bottom=0.18)
    ax_J = fig.add_subplot(gs[0, 0])
    ax_T = fig.add_subplot(gs[0, 1])

    axis_label = (r"initial pulse $\theta_0$" if args.axis == "theta0"
                  else r"initial reactant $u_0$")

    def draw_panel(ax, y_cold, y_hot, ylabel,
                   y_cycle_ref, y_runaway_ref):
        # Two-curve plot
        ax.plot(axis_vals[succ_c], y_cold[succ_c], "o-", color=C_CYCLE,
                ms=3.5, lw=1.0, mec="k", mew=0.4, zorder=4,
                label=fr"cold seed ($J_0={J0_cold:.2f}$)")
        ax.plot(axis_vals[succ_h], y_hot[succ_h], "s--", color=C_RUNAWAY,
                ms=3.0, lw=1.0, mec="k", mew=0.4, zorder=4,
                label=fr"pre-collapsed seed ($J_0={J0_hot:.2f}$)")

        # Reference horizontal lines for the two attractors
        ax.axhline(y_runaway_ref, ls=":", color=C_RUNAWAY, lw=0.6, zorder=1)
        ax.axhline(y_cycle_ref,   ls=":", color=C_CYCLE,   lw=0.6, zorder=1)

        # Mark the cold-seed jump (if any)
        if th_jump is not None:
            ax.axvline(th_jump, ls="-.", color="0.4", lw=0.7, zorder=2)
            ax.text(th_jump, ax.get_ylim()[1] * 0.95,
                    rf"  $\theta_0^*\approx {th_jump:.2f}$",
                    fontsize=6, color="0.25", va="top", ha="left")

        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="out", length=2.5, labelsize=7)
        ax.legend(loc="best", fontsize=6, framealpha=0.95,
                  handlelength=1.6)

    # Reference y values from terminal mean over the obviously-converged samples
    cycle_J_ref   = float(np.nanmean(Jc_term[cyc_c])) if cyc_c.any() else np.nan
    runaway_J_ref = float(np.nanmean(np.r_[Jc_term[run_c], Jh_term[run_h]]))
    cycle_T_ref   = float(np.nanmean(thc_term[cyc_c])) if cyc_c.any() else np.nan
    runaway_T_ref = float(np.nanmean(np.r_[thc_term[run_c], thh_term[run_h]]))

    draw_panel(ax_J, Jc_term, Jh_term, r"$\langle J\rangle_{\rm late}$",
               cycle_J_ref, runaway_J_ref)
    draw_panel(ax_T, thc_term, thh_term, r"$\langle\theta\rangle_{\rm late}$",
               cycle_T_ref, runaway_T_ref)

    # Annotations for the two attractor levels in panel (a)
    ax_J.text(axis_vals[0], cycle_J_ref + 0.12, "cycle",
              fontsize=6, color=C_CYCLE, va="bottom")
    ax_J.text(axis_vals[0], runaway_J_ref + 0.12, "hot-runaway",
              fontsize=6, color=C_RUNAWAY, va="bottom")

    add_panel_label(ax_J, "a", outside=False, x=0.02, y=0.97)
    add_panel_label(ax_T, "b", outside=False, x=0.02, y=0.97)

    # Title strip with cell parameters
    fig.suptitle(rf"C-cell IC-induced hysteresis: "
                 rf"$\mathrm{{Bi}}_T={Bi_T:.3f}$, $S_\chi={S_chi:.2f}$, "
                 rf"$u_0={u0_fixed:.2f}$",
                 fontsize=8, y=0.99)

    save(fig, f"hysteresis_C_{args.axis}")


if __name__ == "__main__":
    main()
