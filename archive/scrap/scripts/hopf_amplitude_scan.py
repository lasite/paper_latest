#!/usr/bin/env python3
"""
hopf_amplitude_scan.py — Stage-1.3 Hopf-onset amplitude scaling.

Sweep Bi_T finely along the working-point row (S_χ = 1.0) to test
whether the J amplitude obeys the supercritical Hopf law

    A_J  ∝  √( Bi_T_c − Bi_T )    just inside the unstable side,
    A_J  =  0                     on the stable side.

Two boundaries are recorded for comparison:
  * 0D analytical Hopf onset Bi_T_c^{0D}: from `re_max_complex = 0` of
    the volume-averaged Jacobian (see hopf_boundary.py).
  * PDE oscillation onset Bi_T_c^{PDE}: the largest Bi_T at which the
    surface J amplitude is above the AMP_THRESH used in fig4_data.

If the two onsets coincide and A_J ~ √(d) with the right exponent on
the unstable side, the Hopf is supercritical and 0D-equivalent.

Outputs:
  data/fig4/hopf_amplitude_scan_Schi_1p00.npz
  Figure/pub/fig4_aux_hopf_scaling.png
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

# Pin BLAS *before* numpy import (mirrors fig4_data)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT
from fig4_data import classify_point, AMP_THRESH, N_WORKERS_DEFAULT
from linear_stability_1d import LSAParams, find_uniform_ss, build_A0
from scipy.linalg import eigvals

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig4").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


def _classify_one(p_dict):
    return classify_point(p_dict)


def _hopf_indicator(p_dict):
    """0D Hopf indicator at one parameter point (swollen-branch SS)."""
    lsa_keys = {f.name for f in LSAParams.__dataclass_fields__.values()}
    base_lsa = {k: v for k, v in p_dict.items() if k in lsa_keys}
    p = LSAParams(**base_lsa)
    ss_list = find_uniform_ss(p)
    if not ss_list:
        return float("nan"), float("nan")
    J0, u0, theta0 = ss_list[-1]   # swollen branch
    A = build_A0(J0, u0, theta0, p)
    evs = eigvals(A)
    imag_nz = np.abs(evs.imag) > 1e-8
    if not imag_nz.any():
        return float("-inf"), float("nan")
    re_c = float(np.max(evs.real[imag_nz]))
    idx = int(np.argmax(np.where(imag_nz, evs.real, -np.inf)))
    omega = float(np.abs(evs[idx].imag))
    return re_c, omega


def _worker(task):
    i, p_dict = task
    r = _classify_one(p_dict)
    re_c, omega = _hopf_indicator(p_dict)
    return i, r, re_c, omega


def run_scan(s_chi=1.0, n_pts=30, force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"hopf_amplitude_scan_Schi_{s_chi:.2f}".replace(".", "p") + ".npz"
    cache = DATA_DIR / fname
    if cache.exists() and not force:
        print(f"  Using cache: {cache}")
        return dict(np.load(cache))

    # Wider range than the fig4 main grid, to bracket both 0D Hopf and PDE onsets
    Bi_T_vals = np.geomspace(0.025, 0.50, n_pts)
    surf_amp  = np.full(n_pts, np.nan)
    J_amp_max = np.full(n_pts, np.nan)
    period    = np.full(n_pts, np.nan)
    re_c      = np.full(n_pts, np.nan)
    omega     = np.full(n_pts, np.nan)
    phi_max   = np.full(n_pts, np.nan)

    base = dict(WORKING_POINT)
    base["S_chi"] = float(s_chi)

    tasks = []
    for i, bv in enumerate(Bi_T_vals):
        p = dict(base); p["Bi_T"] = float(bv)
        tasks.append((i, p))

    print(f"  Scanning Bi_T ∈ [{Bi_T_vals.min():.3f}, {Bi_T_vals.max():.3f}], "
          f"{n_pts} pts, S_chi={s_chi}; workers={N_WORKERS_DEFAULT}")
    with ProcessPoolExecutor(max_workers=N_WORKERS_DEFAULT) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            i, r, rc, om = f.result()
            surf_amp[i]  = r["surf_amp"]
            J_amp_max[i] = r["J_amp_max"]
            period[i]    = r["period"]
            phi_max[i]   = r["phi_max"]
            re_c[i]      = rc
            omega[i]     = om
            done += 1
            if done % 5 == 0 or done == n_pts:
                print(f"    [{done:>3}/{n_pts}]")

    np.savez_compressed(cache, Bi_T=Bi_T_vals, surf_amp=surf_amp,
                        J_amp_max=J_amp_max, period=period,
                        re_c=re_c, omega=omega, phi_max=phi_max,
                        S_chi=np.array(s_chi))
    print(f"  Saved: {cache}")
    return dict(np.load(cache))


def hopf_zero_crossing(Bi_T, re_c):
    """Find the upper Hopf boundary on the swollen branch.

    Treat -inf (no complex pair) as a stable side and look for the
    largest Bi_T at which re_c is finite & positive. Returns (lower, upper)
    Bi_T values bracketing the Hopf-unstable interval (NaN if absent).
    """
    re_c = np.asarray(re_c)
    Bi_T = np.asarray(Bi_T)
    pos_mask = np.isfinite(re_c) & (re_c > 0)
    if not pos_mask.any():
        return float("nan"), float("nan")
    idx = np.where(pos_mask)[0]
    return float(Bi_T[idx[0]]), float(Bi_T[idx[-1]])


def pde_oscillating_interval(Bi_T, surf_amp, thresh=AMP_THRESH):
    """Largest contiguous interval of Bi_T where surf_amp > thresh."""
    mask = np.isfinite(surf_amp) & (surf_amp > thresh)
    if not mask.any():
        return float("nan"), float("nan")
    idx = np.where(mask)[0]
    return float(Bi_T[idx[0]]), float(Bi_T[idx[-1]])


def main():
    d = run_scan(s_chi=1.0, n_pts=30)

    Bi_T = d["Bi_T"]; surf = d["surf_amp"]
    J_amp = d["J_amp_max"]; rec = d["re_c"]
    period = d["period"]; phi_max = d["phi_max"]

    # Onsets — interval form, since the unstable region is bounded on
    # both sides (low Bi_T frozen-front; high Bi_T steady-cold).
    Bi_T_0D_lo, Bi_T_0D_hi = hopf_zero_crossing(Bi_T, rec)
    Bi_T_PDE_lo, Bi_T_PDE_hi = pde_oscillating_interval(Bi_T, surf,
                                                         thresh=AMP_THRESH)

    print(f"\n  0D Hopf-unstable interval  (swollen branch):  "
          f"Bi_T ∈ [{Bi_T_0D_lo:.4f}, {Bi_T_0D_hi:.4f}]")
    print(f"  PDE oscillating interval   (surf_amp>{AMP_THRESH:.2f}): "
          f"Bi_T ∈ [{Bi_T_PDE_lo:.4f}, {Bi_T_PDE_hi:.4f}]")
    print("\n  Two key observations:")
    print(f"    1. PDE onsets at lower Bi_T than 0D ({Bi_T_PDE_lo:.4f} vs "
          f"{Bi_T_0D_lo:.4f}) — PDE has an extra unstable mode the 0D")
    print(f"       analysis on the swollen branch does NOT see.")
    print(f"    2. PDE shuts off earlier on the high-Bi_T side "
          f"({Bi_T_PDE_hi:.4f} vs {Bi_T_0D_hi:.4f}) — 0D predicts Hopf")
    print(f"       but PDE picks the frozen-front branch instead "
          f"(spatial pinning).")

    # ── Figure: 1×3 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(PRE_DOUBLE, 2.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.22,
                        wspace=0.42)

    def shade_intervals(ax):
        # 0D Hopf-unstable shading (light yellow)
        if np.isfinite(Bi_T_0D_lo) and np.isfinite(Bi_T_0D_hi):
            ax.axvspan(Bi_T_0D_lo, Bi_T_0D_hi,
                       color="#fff0c2", alpha=0.55, lw=0, zorder=0)
        # PDE oscillating shading (light red)
        if np.isfinite(Bi_T_PDE_lo) and np.isfinite(Bi_T_PDE_hi):
            ax.axvspan(Bi_T_PDE_lo, Bi_T_PDE_hi,
                       color="#fde0d3", alpha=0.85, lw=0, zorder=0.5)

    # Panel (a): J amplitudes vs Bi_T
    ax = axes[0]
    shade_intervals(ax)
    ax.plot(Bi_T, surf, "o-", color="#1f5fa3", lw=1.0, ms=4,
            label=r"surface $\Delta J$", zorder=3)
    ax.plot(Bi_T, J_amp, "s--", color="#a23e1c", lw=0.9, ms=3.5,
            label=r"$\max_\xi\,\Delta J$", zorder=3)
    ax.axhline(AMP_THRESH, color="0.4", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=8)
    ax.set_ylabel(r"J amplitude", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.9,
              handlelength=1.4, borderpad=0.25)
    add_panel_label(ax, "a")
    # Small legend tags for the shadings
    ax.text(0.99, 0.02,
            "yellow: 0D Hopf-unstable\nred: PDE oscillating",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.0, color="0.20",
            bbox=dict(facecolor="white", edgecolor="0.6",
                      boxstyle="round,pad=0.15", lw=0.3))

    # Panel (b): 0D Hopf indicator + max φ to expose the branches.
    ax = axes[1]
    shade_intervals(ax)
    rec_pos = np.where(np.isfinite(rec) & (rec >= 0), rec, np.nan)
    rec_neg = np.where(np.isfinite(rec) & (rec < 0), rec, np.nan)
    rec_minf = ~np.isfinite(rec)
    ax.plot(Bi_T, rec_pos, "o-", color="#1f5fa3", lw=1.0, ms=4,
            label=r"$\max\,\Re(\sigma_c)$ (0D)", zorder=3)
    ax.plot(Bi_T, rec_neg, "o-", color="#1f5fa3", lw=1.0, ms=4,
            zorder=3)
    if rec_minf.any():
        ax.scatter(Bi_T[rec_minf], np.zeros(int(rec_minf.sum())) - 0.05,
                   marker="v", color="0.4", s=18, zorder=4,
                   label="no complex pair")
    ax.axhline(0.0, color="0.3", lw=0.6, ls=":")
    ax2 = ax.twinx()
    ax2.plot(Bi_T, phi_max, "x", color="#137a73", ms=4, lw=0.6,
             alpha=0.7, label=r"$\max\,\varphi$ (PDE)")
    ax2.axhline(0.5, color="#137a73", lw=0.5, ls=":", alpha=0.6)
    ax2.set_ylabel(r"$\max\,\varphi$", color="#137a73", fontsize=7.5)
    ax2.tick_params(axis="y", labelcolor="#137a73", labelsize=6)
    ax2.set_ylim(0.0, 1.05)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$", fontsize=8)
    ax.set_ylabel(r"$\Re(\sigma_c)$ (0D, swollen)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9,
              handlelength=1.4, borderpad=0.25)
    add_panel_label(ax, "b")

    # Panel (c): supercritical scaling test: amp² vs (Bi_T_PDE_hi − Bi_T)/Bi_T_PDE_hi
    # We test the high-Bi_T side, which is the "Hopf-like" boundary
    # between LCST front oscillation and steady-cold/frozen front.
    ax = axes[2]
    if np.isfinite(Bi_T_PDE_hi):
        d_hi = (Bi_T_PDE_hi - Bi_T) / Bi_T_PDE_hi
        unstable_side = (d_hi > 0) & (Bi_T > Bi_T_PDE_lo) \
                        & np.isfinite(surf) & (surf > AMP_THRESH)
        if unstable_side.sum() >= 3:
            xs = d_hi[unstable_side]
            ys = surf[unstable_side]
            ax.plot(xs, ys**2, "o", color="#1f5fa3", ms=4,
                    label=r"PDE high-Bi$_T$ side")
            lx = np.log(xs); ly = np.log(ys**2)
            slope, intercept = np.polyfit(lx, ly, 1)
            xp = np.linspace(xs.min() * 0.9, xs.max() * 1.05, 50)
            ax.plot(xp, np.exp(slope * np.log(xp) + intercept),
                    color="#1f5fa3", lw=0.9, alpha=0.7,
                    label=fr"$A_J^{{2}} \propto d^{{{slope:.2f}}}$")
            ax.text(0.02, 0.98,
                    f"supercritical: slope = 1\nfit slope = {slope:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=5.5,
                    bbox=dict(facecolor="white", edgecolor="0.5",
                              boxstyle="round,pad=0.2", lw=0.4))
            print(f"\n  High-Bi_T side fit:  ln A^2 = {slope:.3f}*ln d + c"
                  f"   ->  A ~ d^{slope/2:.3f}")
            print(f"    slope ~ 1 means supercritical;  slope ~ 0 means hard onset")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$d = (\mathrm{Bi}_T^{\mathrm{PDE,hi}}-\mathrm{Bi}_T)/\mathrm{Bi}_T^{\mathrm{PDE,hi}}$",
                          fontsize=6.5)
            ax.set_ylabel(r"$A_J^{\,2}$", fontsize=8)
            ax.tick_params(labelsize=6.5)
            ax.legend(fontsize=5.5, loc="lower right",
                      framealpha=0.85, handlelength=1.4)
    add_panel_label(ax, "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig4_aux_hopf_scaling.pdf"
    png = OUT_DIR / "fig4_aux_hopf_scaling.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
