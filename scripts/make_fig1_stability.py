#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig1_stability.py — Fig 1: Linear stability of the 1D gel slab.

Layout: 2×3 double-column
  (a) Bifurcation diagram: J₀ vs Bi_T (all branches, stability-colored)
  (b) Leading eigenvalue vs Bi_T (Re and Im; shows Hopf window)
  (c) Dispersion relation σ(k) at 3 representative Bi_T values
  (d) Da–Bi_T stability map
  (e) S_χ–Bi_T stability map
  (f) Period T = 2π/ω vs Da
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import fsolve
from dataclasses import replace
from linear_stability_1d import (
    LSAParams, find_uniform_ss, build_A0, build_D2, build_D4,
    dispersion_eigenvalues, df_dJ, df_dtheta, phi_of_J,
    chem_pot, m_bath, reaction_rate, dispersion_matrix,
)
from style_pub import set_style, fig_panels, add_panel_label, save, C, PRE_DOUBLE

set_style()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def ss_equations(x, pp):
    """Steady-state equations: μ=μ_b and heat balance."""
    J, theta = x
    if J <= pp.phi_p0 * 1.01 or theta < -0.1:
        return [1e10, 1e10]
    u = 1.0 - pp.Bi_T * theta / pp.Bi_c
    if u <= 0 or u > 1.0:
        return [1e10, 1e10]
    mu_b = m_bath(pp)
    eq1 = chem_pot(J, theta, pp) - mu_b
    eq2 = pp.Bi_T * theta - pp.Da * J * reaction_rate(u, theta, J, pp)
    return [eq1, eq2]


def solve_ss(pp, J_guess, theta_guess):
    """Solve for a single SS, returning (J, u, θ) or None."""
    try:
        sol, info, ier, _ = fsolve(ss_equations, [J_guess, theta_guess],
                                   args=(pp,), full_output=True)
        if ier != 1:
            return None
        J0, theta0 = sol
        if J0 <= pp.phi_p0 * 1.01 or theta0 < -0.01:
            return None
        u0 = 1.0 - pp.Bi_T * theta0 / pp.Bi_c
        if u0 <= 0 or u0 > 1.001:
            return None
        res = np.linalg.norm(ss_equations(sol, pp))
        if res > 1e-8:
            return None
        return (J0, u0, theta0)
    except Exception:
        return None


def classify_point(evals):
    """Classify eigenvalue spectrum: 'hopf', 'mono', 'stable', or 'nan'."""
    if np.any(np.isnan(evals.real)):
        return 'nan'
    re_max = np.max(evals.real)
    idx = np.argmax(evals.real)
    im_at_max = abs(evals[idx].imag)
    if re_max > 0:
        return 'hopf' if im_at_max > 0.01 else 'mono'
    return 'stable'


def track_branch(param_name, param_vals, p_base, J_start, theta_start):
    """Track a SS branch by continuation (previous solution → next guess)."""
    n = len(param_vals)
    out = {k: np.full(n, np.nan) for k in ('J', 'u', 'theta', 'fJ')}
    out['evals'] = np.full((n, 3), np.nan, dtype=complex)
    J_g, th_g = J_start, theta_start

    for i, val in enumerate(param_vals):
        pp = replace(p_base, **{param_name: val})
        ss = solve_ss(pp, J_g, th_g)
        if ss is None:
            continue
        J0, u0, theta0 = ss
        out['J'][i] = J0
        out['u'][i] = u0
        out['theta'][i] = theta0
        out['fJ'][i] = df_dJ(J0, theta0, pp)
        A0 = build_A0(J0, u0, theta0, pp)
        out['evals'][i] = np.sort(eigvals(A0))[::-1]
        J_g, th_g = J0, theta0

    return out


def track_branch_bidirectional(param_name, param_vals, p_base, seed_val, J_start, theta_start):
    """Track a branch forward and backward from a seed point, then merge."""
    out = {k: np.full(len(param_vals), np.nan) for k in ('J', 'u', 'theta', 'fJ')}
    out['evals'] = np.full((len(param_vals), 3), np.nan, dtype=complex)
    idx = int(np.argmin(np.abs(param_vals - seed_val)))

    fw = track_branch(param_name, param_vals[idx:], p_base, J_start, theta_start)
    bw = track_branch(param_name, param_vals[:idx + 1][::-1], p_base, J_start, theta_start)
    for key in ('J', 'u', 'theta', 'fJ'):
        out[key][idx:] = fw[key]
        out[key][:idx + 1] = bw[key][::-1]
    out['evals'][idx:] = fw['evals']
    out['evals'][:idx + 1] = bw['evals'][::-1]
    return out


def collect_pointwise_branches(param_name, param_vals, p_base, max_branches=3):
    """
    Solve all SSs pointwise and assign lower/middle/upper branches from a
    maximal-multiplicity seed, avoiding continuation-induced branch switching.
    """
    all_solutions = []
    multiplicity = np.zeros(len(param_vals), dtype=int)
    for i, val in enumerate(param_vals):
        pp = replace(p_base, **{param_name: float(val)})
        ss_all = find_uniform_ss(pp)
        if ss_all is None:
            all_solutions.append([])
            continue
        multiplicity[i] = min(len(ss_all), max_branches)
        all_solutions.append(ss_all[:max_branches])

    n_branches = int(np.max(multiplicity))
    branches = []
    for _ in range(max_branches):
        out = {k: np.full(len(param_vals), np.nan) for k in ('J', 'u', 'theta', 'fJ')}
        out['evals'] = np.full((len(param_vals), 3), np.nan, dtype=complex)
        branches.append(out)

    if n_branches == 0:
        return branches, multiplicity

    seed_candidates = np.where(multiplicity == n_branches)[0]
    seed_idx = int(seed_candidates[len(seed_candidates) // 2])

    def fill_branch_point(branch_idx, point_idx, sol):
        J0, u0, theta0 = sol
        pp = replace(p_base, **{param_name: float(param_vals[point_idx])})
        branches[branch_idx]['J'][point_idx] = J0
        branches[branch_idx]['u'][point_idx] = u0
        branches[branch_idx]['theta'][point_idx] = theta0
        branches[branch_idx]['fJ'][point_idx] = df_dJ(J0, theta0, pp)
        branches[branch_idx]['evals'][point_idx] = np.sort(eigvals(build_A0(J0, u0, theta0, pp)))[::-1]

    for bidx, sol in enumerate(all_solutions[seed_idx]):
        fill_branch_point(bidx, seed_idx, sol)

    from itertools import combinations

    def assign_direction(index_range):
        prev_js = [branches[b]['J'][seed_idx] for b in range(n_branches)]
        for i in index_range:
            sols = all_solutions[i]
            m = len(sols)
            if m == 0:
                prev_js = [np.nan] * n_branches
                continue

            best_choice = None
            best_cost = np.inf
            for branch_ids in combinations(range(n_branches), m):
                cost = 0.0
                ok = True
                for branch_id, sol in zip(branch_ids, sols):
                    prev = prev_js[branch_id]
                    if np.isnan(prev):
                        ok = False
                        break
                    cost += abs(sol[0] - prev)
                if ok and cost < best_cost:
                    best_cost = cost
                    best_choice = branch_ids

            if best_choice is None:
                prev_js = [np.nan] * n_branches
                continue

            curr_js = [np.nan] * n_branches
            for branch_id, sol in zip(best_choice, sols):
                fill_branch_point(branch_id, i, sol)
                curr_js[branch_id] = sol[0]
            prev_js = curr_js

    assign_direction(range(seed_idx + 1, len(param_vals)))
    assign_direction(range(seed_idx - 1, -1, -1))
    return branches, multiplicity


def filter_middle_branch(br_mid_raw, br_lower, br_upper, margin=0.015):
    """Keep only the part of the candidate branch that lies between lower and upper branches."""
    out = {k: np.array(v, copy=True) if k != 'evals' else np.array(v, copy=True)
           for k, v in br_mid_raw.items()}
    for i in range(len(br_mid_raw['J'])):
        jm = br_mid_raw['J'][i]
        if np.isnan(jm):
            continue
        jl = br_lower['J'][i]
        ju = br_upper['J'][i]
        keep = False
        if not np.isnan(jl) and not np.isnan(ju):
            keep = (jl + margin < jm < ju - margin)
        elif not np.isnan(jl):
            keep = (jm > jl + margin)
        elif not np.isnan(ju):
            keep = (jm < ju - margin)
        if not keep:
            out['J'][i] = np.nan
            out['u'][i] = np.nan
            out['theta'][i] = np.nan
            out['fJ'][i] = np.nan
            out['evals'][i, :] = np.nan
    return out


def stability_map_2d(x_name, x_arr, y_name, y_arr, p_base):
    """
    Compute 2D stability map using multi-start search + continuation.
    For each (x,y) point, find all steady states and classify the
    most unstable one.  Uses continuation along rows for speed, with
    fallback to find_uniform_ss when continuation fails.
    Returns (class_map, omega_map, re_map) arrays of shape (ny, nx).
    """
    ny, nx = len(y_arr), len(x_arr)
    cls_map = np.full((ny, nx), np.nan)
    omega_map = np.full((ny, nx), np.nan)
    re_map = np.full((ny, nx), np.nan)

    for j, yv in enumerate(y_arr):
        pp_row = replace(p_base, **{y_name: yv})

        # Try multiple starting guesses for continuation
        guesses = [(0.25, 1.0), (0.45, 2.0), (0.35, 1.5), (0.60, 2.5)]
        branches = []
        for Jg, tg in guesses:
            br = track_branch(x_name, x_arr, pp_row, J_start=Jg, theta_start=tg)
            branches.append(br)
        # Also track backward from high values
        br_rev = track_branch(x_name, x_arr[::-1], pp_row, J_start=0.45, theta_start=2.0)
        br_rev = {k: v[::-1] for k, v in br_rev.items()}
        branches.append(br_rev)

        for i in range(nx):
            best_re = -np.inf
            best_cl = None
            best_ev = None

            for br in branches:
                if np.isnan(br['J'][i]):
                    continue
                ev = br['evals'][i]
                if np.any(np.isnan(ev.real)):
                    continue
                cl = classify_point(ev)
                idx = np.argmax(ev.real)
                re_lead = ev[idx].real
                if re_lead > best_re:
                    best_re = re_lead
                    best_cl = cl
                    best_ev = ev

            # Fallback: use multi-start find_uniform_ss
            if best_cl is None:
                pp = replace(pp_row, **{x_name: float(x_arr[i])})
                ss_all = find_uniform_ss(pp)
                if ss_all:
                    for ss in ss_all:
                        J0, u0, t0 = ss
                        ev = eigvals(build_A0(J0, u0, t0, pp))
                        cl = classify_point(ev)
                        idx = np.argmax(ev.real)
                        re_lead = ev[idx].real
                        if re_lead > best_re:
                            best_re = re_lead
                            best_cl = cl
                            best_ev = ev

            if best_cl is None:
                cls_map[j, i] = -2  # genuinely no SS
            elif best_cl == 'hopf':
                cls_map[j, i] = 1
                idx = np.argmax(best_ev.real)
                omega_map[j, i] = abs(best_ev[idx].imag)
                re_map[j, i] = best_ev[idx].real
            elif best_cl == 'mono':
                cls_map[j, i] = 2
                re_map[j, i] = best_re
            elif best_cl == 'stable':
                cls_map[j, i] = 0
                re_map[j, i] = best_re

        # Also check swollen branch
        br_sw = track_branch(x_name, x_arr[::-1], pp_row,
                             J_start=1.3, theta_start=0.3)
        for i in range(nx):
            ri = nx - 1 - i
            if not np.isnan(cls_map[j, ri]):
                continue
            ev = br_sw['evals'][i]
            if np.isnan(br_sw['J'][i]):
                continue
            cl = classify_point(ev)
            idx = np.argmax(ev.real)
            re_map[j, ri] = ev[idx].real
            if cl == 'hopf':
                cls_map[j, ri] = 1
                omega_map[j, ri] = abs(ev[idx].imag)
            elif cl == 'mono':
                cls_map[j, ri] = 2
            elif cl == 'stable':
                cls_map[j, ri] = 0

    return cls_map, omega_map, re_map


# ═══════════════════════════════════════════════════════════════════
# Compute all data
# ═══════════════════════════════════════════════════════════════════

p = LSAParams()

# ─── (a),(b): Bi_T bifurcation diagram ──────────────────────────
print("Computing Bi_T bifurcation...")
BiT_vals = np.linspace(0.02, 0.50, 300)
BiT_vals = np.unique(np.concatenate([
    np.linspace(0.02, 0.50, 260),
    np.linspace(0.215, 0.235, 160),
    np.linspace(0.248, 0.266, 160),
]))

br_coll = track_branch('Bi_T', BiT_vals, p, J_start=0.25, theta_start=1.0)
br_swol = track_branch('Bi_T', BiT_vals[::-1], p, J_start=1.30, theta_start=0.2)
br_swol = {k: v[::-1] for k, v in br_swol.items()}
BiT_branches, BiT_mult = collect_pointwise_branches('Bi_T', BiT_vals, p, max_branches=3)
br_mid_seed = BiT_branches[1]
mid_idx = np.where(~np.isnan(br_mid_seed['J']))[0]
br_mid = {k: np.full_like(v, np.nan) if k != 'evals' else np.full_like(v, np.nan)
          for k, v in br_mid_seed.items()}
if len(mid_idx) > 0:
    seed_i = int(mid_idx[len(mid_idx) // 2])
    br_mid_raw = track_branch_bidirectional(
        'Bi_T', BiT_vals, p, float(BiT_vals[seed_i]),
        float(br_mid_seed['J'][seed_i]), float(br_mid_seed['theta'][seed_i])
    )
    br_mid = filter_middle_branch(br_mid_raw, br_coll, br_swol, margin=0.015)

print(f"  Lower branch:     {np.sum(~np.isnan(br_coll['J']))} points")
print(f"  Middle branch:    {np.sum(~np.isnan(br_mid['J']))} points")
print(f"  Upper branch:     {np.sum(~np.isnan(br_swol['J']))} points")

# ─── (c): Dispersion relation at 3 Bi_T values ─────────────────
print("Computing dispersion relations...")
k_arr = np.linspace(0, 150, 1500)
BiT_disp = [0.04, 0.10, 0.40]
disp_data = {}
for BiT in BiT_disp:
    pp = replace(p, Bi_T=BiT)
    ss = find_uniform_ss(pp)
    if ss is None:
        print(f"  Bi_T={BiT}: no SS")
        continue
    J0, u0, t0 = ss[0]
    sigmas = dispersion_eigenvalues(k_arr, J0, u0, t0, pp)
    disp_data[BiT] = {'sigmas': sigmas, 'J0': J0, 'u0': u0, 'theta0': t0}
    print(f"  Bi_T={BiT:.2f}: J₀={J0:.3f}, max Re(σ)={np.max(sigmas[:,0].real):.2f}")

# ─── (d): Da–Bi_T stability map (continuation-based) ────────────
print("Computing Da-Bi_T stability map (continuation)...")
Da_arr = np.linspace(0.5, 20.0, 60)
BiT_arr = np.linspace(0.02, 0.50, 60)
map_DaBiT, map_DaBiT_omega, map_DaBiT_re = stability_map_2d(
    'Da', Da_arr, 'Bi_T', BiT_arr, p)
print("  Done.")

# ─── (e): S_chi–Bi_T stability map (continuation-based) ─────────
print("Computing S_chi-Bi_T stability map (continuation)...")
Sc_arr = np.linspace(0.1, 2.0, 60)
map_ScBiT, _, _ = stability_map_2d('S_chi', Sc_arr, 'Bi_T', BiT_arr, p)
print("  Done.")

# ─── Smooth classification maps ─────────────────────────────────
from scipy.ndimage import median_filter
# Replace NaN with -2 for filtering, then apply 3×3 median
for m in (map_DaBiT, map_ScBiT):
    m[np.isnan(m)] = -2
map_DaBiT = median_filter(map_DaBiT, size=3)
map_ScBiT = median_filter(map_ScBiT, size=3)
print("  Maps smoothed.")

# ─── (f): Period and growth rate vs Da ───────────────────────────
print("Computing Da scan for period (multi-start)...")
Da_scan = np.linspace(1.0, 20.0, 100)
period_arr = np.full(len(Da_scan), np.nan)
re_arr_da = np.full(len(Da_scan), np.nan)

# Multi-start: try several initial guesses and pick most unstable
guesses_da = [(0.25, 1.0), (0.45, 2.0), (0.35, 1.5), (0.60, 2.5)]
branches_da = []
for Jg, tg in guesses_da:
    branches_da.append(track_branch('Da', Da_scan, p, J_start=Jg, theta_start=tg))
# Also reverse
br_rev_da = track_branch('Da', Da_scan[::-1], p, J_start=0.45, theta_start=2.0)
br_rev_da = {k: v[::-1] for k, v in br_rev_da.items()}
branches_da.append(br_rev_da)

for i in range(len(Da_scan)):
    best_re = -np.inf
    best_ev = None
    for br in branches_da:
        if np.isnan(br['J'][i]):
            continue
        ev = br['evals'][i]
        if np.any(np.isnan(ev.real)):
            continue
        idx = np.argmax(ev.real)
        if ev[idx].real > best_re:
            best_re = ev[idx].real
            best_ev = ev
    # Fallback: find_uniform_ss
    if best_ev is None:
        pp = replace(p, Da=float(Da_scan[i]))
        ss_all = find_uniform_ss(pp)
        if ss_all:
            for ss in ss_all:
                J0, u0, t0 = ss
                ev = np.sort(eigvals(build_A0(J0, u0, t0, pp)))[::-1]
                idx = np.argmax(ev.real)
                if ev[idx].real > best_re:
                    best_re = ev[idx].real
                    best_ev = ev
    if best_ev is not None:
        idx = np.argmax(best_ev.real)
        re_arr_da[i] = best_ev[idx].real
        omega = abs(best_ev[idx].imag)
        if omega > 0.01:
            period_arr[i] = 2 * np.pi / omega
print("  Done.")


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig = plt.figure(figsize=(PRE_DOUBLE, 5.2))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.48,
                       left=0.07, right=0.96, top=0.94, bottom=0.09)

# Color scheme
c_hopf = '#2ca02c'   # green
c_mono = '#d62728'   # red
c_stable = '#1f77b4' # blue
c_noSS = '#dddddd'   # light gray
c_saddle = '#555555' # dark gray


def plot_branch_segments(ax, xvals, branch, color_override=None, label=None, zorder=2):
    """Plot one continued branch, grouping consecutive same-class points
    into single connected lines (avoids visual gaps at segment boundaries)."""
    Jvals = branch['J']
    evals = branch['evals']
    
    # Build list of (start, end, classification) runs
    runs = []
    i = 0
    while i < len(xvals):
        if np.isnan(Jvals[i]):
            i += 1
            continue
        cl = classify_point(evals[i])
        j = i + 1
        while j < len(xvals) and not np.isnan(Jvals[j]) and classify_point(evals[j]) == cl:
            j += 1
        runs.append((i, j, cl))
        i = j
    
    first = True
    for (i0, i1, cl) in runs:
        # Include one extra point at each boundary for continuity
        lo = max(i0 - 1, 0)
        hi = min(i1, len(xvals))
        # Only use valid (non-NaN) range
        mask = ~np.isnan(Jvals[lo:hi])
        xx = xvals[lo:hi][mask]
        yy = Jvals[lo:hi][mask]
        if len(xx) < 2:
            continue
        color = color_override
        if color is None:
            color = c_hopf if cl == 'hopf' else (c_mono if cl == 'mono' else c_stable)
        ls = '-' if cl == 'stable' else '-'  # all solid, color encodes stability
        ax.plot(xx, yy, ls, color=color, lw=2.0,
                zorder=zorder, label=label if first else None)
        first = False

# ─── Panel (a): Bifurcation diagram J₀ vs Bi_T ──────────────────
ax_a = fig.add_subplot(gs[0, 0])

plot_branch_segments(ax_a, BiT_vals, br_coll, label='Collapsed / lower')
plot_branch_segments(ax_a, BiT_vals, br_swol, label='Swollen / upper')
if br_mid is not None:
    plot_branch_segments(ax_a, BiT_vals, br_mid, color_override=c_saddle,
                         label='Saddle / middle', zorder=3)

# Fold bifurcation region
ax_a.annotate('fold', xy=(0.28, 0.70), fontsize=7, color='0.3',
              ha='center', style='italic', weight='bold')

# Mark working point
wp_idx = np.argmin(np.abs(BiT_vals - 0.10))
J_wp = br_coll['J'][wp_idx]
ax_a.plot(0.10, J_wp, 'k*', ms=10, zorder=5)

ax_a.set_xlabel(r'$\mathrm{Bi}_T$')
ax_a.set_ylabel(r'$J_0$')
ax_a.set_xlim(0, 0.50)
ax_a.set_ylim(0, 1.45)

legend_elements = [
    Line2D([0], [0], color=c_hopf, lw=2, label='Hopf'),
    Line2D([0], [0], color=c_mono, lw=2, label='Monotone'),
    Line2D([0], [0], color=c_stable, lw=2, label='Stable'),
    Line2D([0], [0], color=c_saddle, lw=2, ls='--', label='Saddle branch'),
]
ax_a.legend(handles=legend_elements, loc='upper right', fontsize=6,
            framealpha=0.9, handlelength=1.2)
add_panel_label(ax_a, 'a')

# ─── Panel (b): Eigenvalue real/imag parts vs Bi_T ──────────────
ax_b = fig.add_subplot(gs[0, 1])

mask_c = ~np.isnan(br_coll['J'])
bt_c = BiT_vals[mask_c]
ev_c = br_coll['evals'][mask_c]
re_lead = np.array([max(e.real) for e in ev_c])
im_lead = np.array([abs(e[np.argmax(e.real)].imag) for e in ev_c])

ax_b.plot(bt_c, re_lead, '-', color=c_mono, lw=1.5)
ax_b.axhline(0, color='k', lw=0.4)

ax_b2 = ax_b.twinx()
ax_b2.plot(bt_c, im_lead, '--', color='#9467bd', lw=1.2)
ax_b2.set_ylabel(r'$|\mathrm{Im}(\sigma_1)|$', color='#9467bd',
                  fontsize=7, rotation=270, labelpad=12)
ax_b2.tick_params(axis='y', colors='#9467bd', labelsize=6)

# Shade Hopf and mono windows
hopf_mask = (re_lead > 0) & (im_lead > 0.01)
mono_mask = (re_lead > 0) & (im_lead < 0.01)
if np.any(hopf_mask):
    bt_hopf = bt_c[hopf_mask]
    ax_b.axvspan(bt_hopf.min(), bt_hopf.max(), alpha=0.10, color=c_hopf, zorder=0)
if np.any(mono_mask):
    bt_mono = bt_c[mono_mask]
    ax_b.axvspan(bt_mono.min(), bt_mono.max(), alpha=0.10, color=c_mono, zorder=0)

# Region labels
y_label = ax_b.get_ylim()[1] * 0.88
if np.any(mono_mask):
    ax_b.text(np.mean(bt_c[mono_mask]), y_label, 'mono.',
              fontsize=5.5, ha='center', color=c_mono, weight='bold',
              bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
if np.any(hopf_mask):
    x_hopf = bt_hopf.min() + 0.40 * (bt_hopf.max() - bt_hopf.min())
    ax_b.text(x_hopf, ax_b.get_ylim()[1] * 0.80, 'Hopf',
              fontsize=6, ha='center', color='#1f8f1f', weight='bold',
              bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

# Working point line
ax_b.axvline(0.10, color='k', lw=0.8, ls=':', alpha=0.6)
ax_b.text(0.105, ax_b.get_ylim()[1] * 0.05, r'$\bigstar$', fontsize=8, va='bottom')

ax_b.set_xlabel(r'$\mathrm{Bi}_T$')
ax_b.set_ylabel(r'$\mathrm{Re}(\sigma)$')
ax_b.set_xlim(0.02, 0.30)

leg_b = [
    Line2D([0], [0], color=c_mono, lw=1.5, label=r'$\mathrm{Re}(\sigma_1)$'),
    Line2D([0], [0], color='#9467bd', lw=1.2, ls='--', label=r'$|\mathrm{Im}(\sigma_1)|$'),
]
ax_b.legend(handles=leg_b, loc='upper right', fontsize=5.5, framealpha=0.9)
add_panel_label(ax_b, 'b')

# ─── Panel (c): Dispersion relation ─────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

labels_disp = {0.04: r'$\mathrm{Bi}_T\!=\!0.04$',
               0.10: r'$\mathrm{Bi}_T\!=\!0.10$',
               0.40: r'$\mathrm{Bi}_T\!=\!0.40$'}
colors_disp = {0.04: c_mono, 0.10: c_hopf, 0.40: c_stable}
ls_disp = {0.04: '--', 0.10: '-', 0.40: '-.'}

for BiT in BiT_disp:
    if BiT not in disp_data:
        continue
    sig = disp_data[BiT]['sigmas']
    re_max_k = np.array([max(s.real) for s in sig])
    ax_c.plot(k_arr, re_max_k, ls_disp[BiT], color=colors_disp[BiT],
              lw=1.2, label=labels_disp[BiT])

ax_c.axhline(0, color='k', lw=0.4)
ax_c.set_xlabel(r'Wavenumber $k$')
ax_c.set_ylabel(r'max Re$(\sigma)$', fontsize=7, labelpad=2)
ax_c.set_xlim(0, 150)
# Position legend above the inset, right of the peak annotation
ax_c.legend(fontsize=5.5, framealpha=0.9,
            bbox_to_anchor=(1.0, 1.0), loc='upper right')

# Annotate spinodal peak
if 0.10 in disp_data:
    sig10 = disp_data[0.10]['sigmas']
    re10 = np.array([max(s.real) for s in sig10])
    k_peak = k_arr[np.argmax(re10)]
    re_peak = np.max(re10)
    ax_c.annotate(f'$k^*\\!\\approx\\!{k_peak:.0f}$',
                  xy=(k_peak, re_peak), xytext=(k_peak + 20, re_peak * 0.75),
                  fontsize=5.5, arrowprops=dict(arrowstyle='->', lw=0.5))

# Inset: zoom k=0 region (better positioned, larger)
ax_ins = ax_c.inset_axes([0.45, 0.08, 0.52, 0.42])
n_zoom = 60  # more points for better view
for BiT in BiT_disp:
    if BiT not in disp_data:
        continue
    sig = disp_data[BiT]['sigmas']
    re_max_k = np.array([max(s.real) for s in sig[:n_zoom]])
    ax_ins.plot(k_arr[:n_zoom], re_max_k, ls_disp[BiT],
                color=colors_disp[BiT], lw=0.8)
ax_ins.axhline(0, color='k', lw=0.3)
ax_ins.set_xlim(0, k_arr[n_zoom - 1])
ax_ins.set_ylim(-2, 10)
ax_ins.set_xlabel('$k$', fontsize=5, labelpad=0)
ax_ins.set_ylabel(r'Re$(\sigma)$', fontsize=5, labelpad=0)
ax_ins.tick_params(labelsize=5)
ax_ins.set_title(r'$k\!\to\!0$', fontsize=5, pad=2)
# Mark Hopf at k=0
if 0.10 in disp_data:
    re0 = disp_data[0.10]['sigmas'][0, 0].real
    ax_ins.plot(0, re0, 'o', color=c_hopf, ms=3, zorder=5)
add_panel_label(ax_c, 'c')

# ─── Panel (d): Da–Bi_T stability map ───────────────────────────
ax_d = fig.add_subplot(gs[1, 0])

cmap_cls = ListedColormap([c_noSS, c_stable, c_hopf, c_mono])
bounds = [-2.5, -0.5, 0.5, 1.5, 2.5]
norm_cls = BoundaryNorm(bounds, cmap_cls.N)

Da_mg, BiT_mg = np.meshgrid(Da_arr, BiT_arr)
ax_d.pcolormesh(Da_mg, BiT_mg, map_DaBiT, cmap=cmap_cls, norm=norm_cls,
                shading='nearest', rasterized=True)

ax_d.plot(4.0, 0.10, 'w*', ms=8, zorder=5, markeredgecolor='k', markeredgewidth=0.5)

ax_d.set_xlabel(r'$\mathrm{Da}$')
ax_d.set_ylabel(r'$\mathrm{Bi}_T$')

legend_d = [
    Patch(facecolor=c_hopf, edgecolor='0.3', lw=0.3, label='Hopf'),
    Patch(facecolor=c_mono, edgecolor='0.3', lw=0.3, label='Monotone'),
    Patch(facecolor=c_stable, edgecolor='0.3', lw=0.3, label='Stable'),
    Patch(facecolor=c_noSS, edgecolor='0.3', lw=0.3, label='No SS'),
]
ax_d.legend(handles=legend_d, loc='upper right', fontsize=5,
            framealpha=0.9, handlelength=1.0)
add_panel_label(ax_d, 'd')

# ─── Panel (e): S_chi–Bi_T stability map ────────────────────────
ax_e = fig.add_subplot(gs[1, 1])

Sc_mg, BiT_mg2 = np.meshgrid(Sc_arr, BiT_arr)
ax_e.pcolormesh(Sc_mg, BiT_mg2, map_ScBiT, cmap=cmap_cls, norm=norm_cls,
                shading='nearest', rasterized=True)

ax_e.plot(1.0, 0.10, 'w*', ms=8, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
ax_e.set_xlabel(r'$S_\chi$')
ax_e.set_ylabel(r'$\mathrm{Bi}_T$')
add_panel_label(ax_e, 'e')

# ─── Panel (f): Period and growth rate vs Da ─────────────────────
ax_f = fig.add_subplot(gs[1, 2])

c_per = '#9467bd'
ax_f.plot(Da_scan, period_arr, '-', color=c_per, lw=1.5)
ax_f.set_xlabel(r'$\mathrm{Da}$')
ax_f.set_ylabel(r'$T_{\rm LSA}=2\pi/\omega$  $(\tau)$', color=c_per)
ax_f.tick_params(axis='y', colors=c_per)
ax_f.set_xlim(1, 20)
ax_f.set_ylim(11, 19)

ax_f2 = ax_f.twinx()
ax_f2.plot(Da_scan, re_arr_da, '--', color=c_mono, lw=1.0)
ax_f2.set_ylabel(r'$\mathrm{Re}(\sigma_1)$', color=c_mono, fontsize=7)
ax_f2.tick_params(axis='y', colors=c_mono, labelsize=6)
ax_f2.axhline(0, color='k', lw=0.3)

# Working point line
ax_f.axvline(4.0, color='k', lw=0.5, ls=':', alpha=0.5)

leg_f = [
    Line2D([0], [0], color=c_per, lw=1.5, label=r'Period $T$'),
    Line2D([0], [0], color=c_mono, lw=1.0, ls='--', label=r'$\mathrm{Re}(\sigma_1)$'),
]
ax_f.legend(handles=leg_f, loc='center right', fontsize=5.5, framealpha=0.9)
add_panel_label(ax_f, 'f')

# ─── Save ────────────────────────────────────────────────────────
save(fig, 'fig1')
print("\n✓ Figure saved.")
