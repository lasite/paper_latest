#!/usr/bin/env python3
"""
make_slow_manifold.py — §IV.B signature panel: slow manifold + limit cycle.

Renders the (theta, J) phase plane with:
  * the locus mu(J, theta) = mu_b (= bath equilibrium chemical potential)
    decomposed into stable swollen / unstable middle / stable collapsed
    branches by the sign of d(mu)/dJ;
  * upper and lower fold points;
  * the PDE surface limit cycle at the working point, colored by the
    speed |d(theta,J)/dt| on a log scale to expose slow-manifold drift
    vs. fast mechanical jumps;
  * the small centre loop, plotted alongside.

Output:  figures_pub/slow_manifold.{pdf,png}
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
from scipy.signal import find_peaks

from style_pub import set_style, PRE_DOUBLE, add_panel_label, save, C
from scan_optimized import Params, local_chem_pot, finalize_params
from fig2_data import WORKING_POINT, load_cache

set_style()


# ─────────────────────────────────────────────────────────────────────
# Slow manifold extraction
# ─────────────────────────────────────────────────────────────────────

def mu_minus_mb_at(J, theta, p):
    return float(local_chem_pot(np.array([J]), np.array([theta]), p)[0]) - p.m_b


def find_roots_at_theta(theta, p, J_lo, J_hi, n_sample=4000):
    """Return all J roots of mu(J, theta)=mu_b in [J_lo, J_hi], sorted."""
    Js = np.linspace(J_lo, J_hi, n_sample)
    mus = local_chem_pot(Js, np.full_like(Js, theta), p) - p.m_b
    sgn = np.sign(mus)
    flips = np.where(np.diff(sgn) != 0)[0]
    roots = []
    for k in flips:
        try:
            r = brentq(mu_minus_mb_at, Js[k], Js[k + 1], args=(theta, p),
                       xtol=1e-9, maxiter=200)
            roots.append(r)
        except (ValueError, RuntimeError):
            pass
    return sorted(roots)


def dmu_dJ(J, theta, p, dJ=1e-5):
    """Central difference for d(mu)/dJ at (J, theta)."""
    a = local_chem_pot(np.array([J + dJ]), np.array([theta]), p)[0]
    b = local_chem_pot(np.array([J - dJ]), np.array([theta]), p)[0]
    return (a - b) / (2.0 * dJ)


def trace_branches(p, theta_min=-0.3, theta_max=4.5, n_theta=900,
                   J_lo=None, J_hi=3.0):
    """Sweep theta, collect roots, separate into upper/middle/lower branches.

    Returns dict with arrays for each branch and located fold points.
    """
    if J_lo is None:
        J_lo = p.phi_p0 / 0.995 + 1e-6  # just above the hard ceiling
    thetas = np.linspace(theta_min, theta_max, n_theta)

    # Per-theta root buckets, with stability flag (∂μ/∂J > 0 = stable)
    upper = []   # stable, large J
    middle = []  # unstable, intermediate J
    lower = []   # stable, small J (collapsed)

    for theta in thetas:
        roots = find_roots_at_theta(theta, p, J_lo, J_hi)
        if not roots:
            continue
        if len(roots) == 1:
            r = roots[0]
            slope = dmu_dJ(r, theta, p)
            # Single-root regime: above lower fold but past upper fold (only collapsed),
            # or below lower fold (only swollen). Use J value to attribute.
            if slope > 0 and r > 0.5:
                upper.append((theta, r))
            elif slope > 0 and r <= 0.5:
                lower.append((theta, r))
            else:
                middle.append((theta, r))
        elif len(roots) == 2:
            # Tangent / fold; assign by J-ordering with stability check
            for r in roots:
                slope = dmu_dJ(r, theta, p)
                if slope > 0 and r > 0.5:
                    upper.append((theta, r))
                elif slope > 0:
                    lower.append((theta, r))
                else:
                    middle.append((theta, r))
        else:  # 3 roots → collapsed, middle, swollen
            r_lo, r_md, r_hi = roots[0], roots[1], roots[-1]
            lower.append((theta, r_lo))
            middle.append((theta, r_md))
            upper.append((theta, r_hi))

    upper = np.array(upper)
    middle = np.array(middle)
    lower = np.array(lower)

    # Locate folds: maximum theta on the upper branch (upper fold),
    #               minimum theta on the lower branch (lower fold).
    folds = {}
    if len(upper):
        i = np.argmax(upper[:, 0])
        folds["upper"] = (upper[i, 0], upper[i, 1])
    if len(lower):
        i = np.argmin(lower[:, 0])
        folds["lower"] = (lower[i, 0], lower[i, 1])

    return {"upper": upper, "middle": middle, "lower": lower, "folds": folds}


# ─────────────────────────────────────────────────────────────────────
# PDE limit-cycle helpers
# ─────────────────────────────────────────────────────────────────────

def limit_cycle_segments(theta_t, J_t, t_t):
    """Build (segments, speed) arrays for a coloured LineCollection."""
    pts = np.column_stack([theta_t, J_t])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    dtheta = np.diff(theta_t)
    dJ = np.diff(J_t)
    dt = np.diff(t_t)
    dt = np.where(dt > 0, dt, np.median(dt[dt > 0]))
    speed = np.sqrt((dtheta / dt) ** 2 + (dJ / dt) ** 2)
    return segs, speed


def detect_phase_markers(theta_t, J_t, t_t):
    """Reuse the make_fig2_mechanism phase detector.

    Returns dict with indices of (ignite, collapse, cool, swell) on the
    cycle, picked from the second full cycle in the window.
    """
    Ts = np.asarray(theta_t)
    Js = np.asarray(J_t)
    ts = np.asarray(t_t)
    peaks, _ = find_peaks(Ts, prominence=0.5, distance=30)
    troughs, _ = find_peaks(-Ts, prominence=0.5, distance=30)
    if len(peaks) < 2 or len(troughs) < 1:
        return None
    ip = peaks[0]
    ta = troughs[troughs > ip]
    if len(ta) == 0:
        return None
    it = ta[0]
    ip2 = peaks[1] if len(peaks) > 1 else len(Js) - 1
    dT = np.gradient(Ts, ts)
    dJ = np.gradient(Js, ts)
    prev_tr = troughs[troughs < ip]
    i0_ig = prev_tr[-1] if len(prev_tr) > 0 else 0
    i_ignite = i0_ig + int(np.argmax(dT[i0_ig: ip + 1]))
    i_collapse = ip + int(np.argmin(dJ[ip: it + 1]))
    i_swell = it + int(np.argmax(dJ[it: ip2 + 1]))
    i_cool = it
    return {
        "ignite": i_ignite,
        "collapse": i_collapse,
        "cool": i_cool,
        "swell": i_swell,
    }


# ─────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────

def main():
    p = finalize_params(Params(**WORKING_POINT))
    print(f"  mu_b = {p.m_b:.6f}")

    bran = trace_branches(p)
    upper, middle, lower = bran["upper"], bran["middle"], bran["lower"]
    folds = bran["folds"]
    print(f"  upper branch: {len(upper)} pts; theta in "
          f"[{upper[:,0].min():.2f}, {upper[:,0].max():.2f}]")
    print(f"  middle branch: {len(middle)} pts; theta in "
          f"[{middle[:,0].min():.2f}, {middle[:,0].max():.2f}]")
    print(f"  lower branch: {len(lower)} pts; theta in "
          f"[{lower[:,0].min():.2f}, {lower[:,0].max():.2f}]")
    for name, (th_f, J_f) in folds.items():
        print(f"  {name} fold @ theta={th_f:.3f}, J={J_f:.3f}")

    # --- Limit cycle (surface) over a steady-state window
    d = load_cache()
    t = d["t"]
    mask = (t >= 180) & (t <= 240)
    t_w = t[mask]
    J_surf = d["J"][-1, mask]
    th_surf = d["theta"][-1, mask]
    J_ctr = d["J"][0, mask]
    th_ctr = d["theta"][0, mask]
    u_surf = d["u"][-1, mask]

    segs, speed = limit_cycle_segments(th_surf, J_surf, t_w)
    print(f"  surface speed: log10 range "
          f"[{np.log10(np.maximum(speed.min(), 1e-6)):.1f}, "
          f"{np.log10(speed.max()):.1f}]")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(0.6 * PRE_DOUBLE, 3.6))

    # Bistable region shading
    if "lower" in folds and "upper" in folds:
        th_lo_f, _ = folds["lower"]
        th_hi_f, _ = folds["upper"]
        ax.axvspan(th_lo_f, th_hi_f, color="#fff4d6", alpha=0.55, lw=0,
                   zorder=0)
        ax.text(0.5 * (th_lo_f + th_hi_f), 1.78, "bistable region",
                ha="center", va="top", fontsize=6.5, color="#a07a00",
                style="italic", zorder=2)

    # Slow manifold branches
    if len(upper):
        ax.plot(upper[:, 0], upper[:, 1], "-", color=C[0], lw=1.8,
                label="swollen (stable)", zorder=2)
    if len(lower):
        ax.plot(lower[:, 0], lower[:, 1], "-", color=C[1], lw=1.8,
                label="collapsed (stable)", zorder=2)
    if len(middle):
        order = np.argsort(middle[:, 0])
        ax.plot(middle[order, 0], middle[order, 1], "--", color="#555",
                lw=1.0, label="saddle (unstable)", zorder=2)

    # Folds — labels placed inside the plot area, away from axis edges
    if "upper" in folds:
        th_f, J_f = folds["upper"]
        ax.plot(th_f, J_f, "o", mfc="white", mec="k", mew=1.0, ms=5,
                zorder=4)
        ax.annotate("upper fold", (th_f, J_f),
                    xytext=(-6, -12), textcoords="offset points",
                    fontsize=6, ha="right", color="#333", zorder=7)
    if "lower" in folds:
        th_f, J_f = folds["lower"]
        ax.plot(th_f, J_f, "o", mfc="white", mec="k", mew=1.0, ms=5,
                zorder=4)
        ax.annotate("lower fold", (th_f, J_f),
                    xytext=(-6, 14), textcoords="offset points",
                    fontsize=6, ha="right", color="#333", zorder=7,
                    arrowprops=dict(arrowstyle="-", color="#666",
                                    lw=0.4, shrinkB=2))

    # PDE limit cycle, colored by log(speed)
    norm = LogNorm(vmin=max(speed.min(), 1e-3), vmax=speed.max())
    lc = LineCollection(segs, cmap="viridis", norm=norm, lw=1.6,
                        zorder=5, alpha=0.95)
    lc.set_array(speed)
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(r"speed $|d(\theta,J)/dt|$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # Direction arrows + phase labels
    pm = detect_phase_markers(th_surf, J_surf, t_w)
    if pm is not None:
        idxs = [pm["ignite"], pm["collapse"], pm["cool"], pm["swell"]]
        for ai in idxs:
            if 1 <= ai < len(th_surf) - 2:
                ax.annotate("",
                            xy=(th_surf[ai + 1], J_surf[ai + 1]),
                            xytext=(th_surf[ai - 1], J_surf[ai - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="k",
                                            lw=0.9, mutation_scale=10,
                                            alpha=0.9),
                            zorder=8)

        labs = [
            ("① ignite",        pm["ignite"],   ( 6, -10), "left"),
            ("② collapse",      pm["collapse"], (-8,  -6), "right"),
            ("③ cool / quench", pm["cool"],     (-8,  10), "right"),
            ("④ re-swell",      pm["swell"],    (-8,  -6), "right"),
        ]
        for lab, idx, off, ha in labs:
            ax.plot(th_surf[idx], J_surf[idx], "ko", ms=3, zorder=6)
            ax.annotate(lab, (th_surf[idx], J_surf[idx]),
                        xytext=off, textcoords="offset points",
                        fontsize=6.5, ha=ha,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=0.7),
                        zorder=7)

    # Centre loop — small ellipse near (theta~3.2, J~1.5); off-manifold
    # because mu_centre != mu_b.  Annotate to the right.
    ax.plot(th_ctr, J_ctr, "-", color="#444", lw=0.9, alpha=0.95,
            label=r"centre ($\xi=0$)", zorder=3)
    ctr_x = float(np.mean(th_ctr))
    ctr_y = float(np.mean(J_ctr))
    ax.annotate(r"centre ($\xi=0$)",
                xy=(ctr_x, ctr_y),
                xytext=(ctr_x + 0.35, ctr_y + 0.05),
                fontsize=6, ha="left", color="#222",
                arrowprops=dict(arrowstyle="-", color="#444",
                                lw=0.5, shrinkA=0, shrinkB=3),
                zorder=7)

    # Axes
    ax.set_xlabel(r"$\theta_{\mathrm{surf}}$")
    ax.set_ylabel(r"$J_{\mathrm{surf}}$")
    ax.set_xlim(-0.3, 5.0)
    ax.set_ylim(0.10, 1.85)
    ax.legend(loc="lower center", fontsize=6, framealpha=0.95,
              handlelength=1.8, ncol=2,
              bbox_to_anchor=(0.5, -0.32))
    plt.subplots_adjust(bottom=0.22)

    save(fig, "slow_manifold")


if __name__ == "__main__":
    main()
