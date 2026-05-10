"""
fig1_data.py — Shared compute helpers + raw caches for Fig 1 panel scripts.

Pattern matches `fig2_data.py`:
- Heavy computations are wrapped in `load_or_compute_*()` functions that
  cache to `data/fig1/_raw_<descriptor>.npz`.
- Each `make_fig1[a-f].py` calls one of these, then writes its own
  panel-specific cache (`data/fig1/panel_<x>.npz`) and renders.
"""
import os
import sys
import numpy as np
from dataclasses import replace
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.linalg import eigvals
from scipy.optimize import fsolve
from scipy.ndimage import median_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from linear_stability_1d import (
    LSAParams, find_uniform_ss, build_A0,
    dispersion_eigenvalues, df_dJ,
    chem_pot, m_bath, reaction_rate,
)

DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "fig1"))
FIG_DIR  = os.path.normpath(os.path.join(_HERE, "..", "Figure", "fig1"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# Default parallel worker count (per project rule)
N_WORKERS_DEFAULT = int(os.environ.get("FIG4_WORKERS", os.environ.get("FIG1_WORKERS", 24)))


# ═══════════════════════════════════════════════════════════════════
# Steady-state + branch-tracking helpers (moved from make_fig1_stability.py)
# ═══════════════════════════════════════════════════════════════════

def ss_equations(x, pp):
    J, theta = x
    if J <= pp.phi_p0 * 1.01 or theta < -0.1:
        return [1e10, 1e10]
    u = 1.0 - pp.Bi_T * theta / pp.Bi_c
    if u <= 0 or u > 1.0:
        return [1e10, 1e10]
    eq1 = chem_pot(J, theta, pp) - m_bath(pp)
    eq2 = pp.Bi_T * theta - pp.Da * J * reaction_rate(u, theta, J, pp)
    return [eq1, eq2]


def solve_ss(pp, J_guess, theta_guess):
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
        if np.linalg.norm(ss_equations(sol, pp)) > 1e-8:
            return None
        return (J0, u0, theta0)
    except Exception:
        return None


def classify_point(evals_arr):
    if np.any(np.isnan(evals_arr.real)):
        return 'nan'
    re_max = np.max(evals_arr.real)
    idx = np.argmax(evals_arr.real)
    im_at_max = abs(evals_arr[idx].imag)
    if re_max > 0:
        return 'hopf' if im_at_max > 0.01 else 'mono'
    return 'stable'


def track_branch(param_name, param_vals, p_base, J_start, theta_start):
    n = len(param_vals)
    out = {k: np.full(n, np.nan) for k in ('J', 'u', 'theta', 'fJ')}
    out['evals'] = np.full((n, 3), np.nan, dtype=complex)
    J_g, th_g = J_start, theta_start
    for i, val in enumerate(param_vals):
        pp = replace(p_base, **{param_name: float(val)})
        ss = solve_ss(pp, J_g, th_g)
        if ss is None:
            continue
        J0, u0, theta0 = ss
        out['J'][i] = J0
        out['u'][i] = u0
        out['theta'][i] = theta0
        out['fJ'][i] = df_dJ(J0, theta0, pp)
        out['evals'][i] = np.sort(eigvals(build_A0(J0, u0, theta0, pp)))[::-1]
        J_g, th_g = J0, theta0
    return out


def track_branch_bidirectional(param_name, param_vals, p_base, seed_val,
                                J_start, theta_start):
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

    branches = []
    for _ in range(max_branches):
        out = {k: np.full(len(param_vals), np.nan) for k in ('J', 'u', 'theta', 'fJ')}
        out['evals'] = np.full((len(param_vals), 3), np.nan, dtype=complex)
        branches.append(out)

    n_branches = int(np.max(multiplicity))
    if n_branches == 0:
        return branches, multiplicity

    seed_candidates = np.where(multiplicity == n_branches)[0]
    seed_idx = int(seed_candidates[len(seed_candidates) // 2])

    def fill(branch_idx, point_idx, sol):
        J0, u0, theta0 = sol
        pp = replace(p_base, **{param_name: float(param_vals[point_idx])})
        branches[branch_idx]['J'][point_idx] = J0
        branches[branch_idx]['u'][point_idx] = u0
        branches[branch_idx]['theta'][point_idx] = theta0
        branches[branch_idx]['fJ'][point_idx] = df_dJ(J0, theta0, pp)
        branches[branch_idx]['evals'][point_idx] = np.sort(eigvals(build_A0(J0, u0, theta0, pp)))[::-1]

    for bidx, sol in enumerate(all_solutions[seed_idx]):
        fill(bidx, seed_idx, sol)

    def assign(direction):
        prev_js = [branches[b]['J'][seed_idx] for b in range(n_branches)]
        for i in direction:
            sols = all_solutions[i]
            m = len(sols)
            if m == 0:
                prev_js = [np.nan] * n_branches
                continue
            best_choice, best_cost = None, np.inf
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
                fill(branch_id, i, sol)
                curr_js[branch_id] = sol[0]
            prev_js = curr_js

    assign(range(seed_idx + 1, len(param_vals)))
    assign(range(seed_idx - 1, -1, -1))
    return branches, multiplicity


def filter_middle_branch(br_mid_raw, br_lower, br_upper, margin=0.015):
    out = {k: (np.array(v, copy=True)) for k, v in br_mid_raw.items()}
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


# ═══════════════════════════════════════════════════════════════════
# 2-D stability map (parallelized over rows)
# ═══════════════════════════════════════════════════════════════════

def _stability_row(args):
    """Compute one row of the (y_name, x_name) stability map."""
    yv, x_name, x_arr, y_name, p_base = args
    pp_row = replace(p_base, **{y_name: float(yv)})
    nx = len(x_arr)

    guesses = [(0.25, 1.0), (0.45, 2.0), (0.35, 1.5), (0.60, 2.5)]
    branches = []
    for Jg, tg in guesses:
        branches.append(track_branch(x_name, x_arr, pp_row, J_start=Jg, theta_start=tg))
    br_rev = track_branch(x_name, x_arr[::-1], pp_row, J_start=0.45, theta_start=2.0)
    branches.append({k: v[::-1] for k, v in br_rev.items()})

    cls_row = np.full(nx, np.nan)
    om_row = np.full(nx, np.nan)
    re_row = np.full(nx, np.nan)

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
        if best_cl is None:
            pp = replace(pp_row, **{x_name: float(x_arr[i])})
            ss_all = find_uniform_ss(pp)
            if ss_all:
                for ss in ss_all:
                    J0, u0, t0 = ss
                    ev = np.sort(eigvals(build_A0(J0, u0, t0, pp)))[::-1]
                    cl = classify_point(ev)
                    idx = np.argmax(ev.real)
                    re_lead = ev[idx].real
                    if re_lead > best_re:
                        best_re = re_lead
                        best_cl = cl
                        best_ev = ev
        if best_cl is None:
            cls_row[i] = -2
        elif best_cl == 'hopf':
            cls_row[i] = 1
            idx = np.argmax(best_ev.real)
            om_row[i] = abs(best_ev[idx].imag)
            re_row[i] = best_ev[idx].real
        elif best_cl == 'mono':
            cls_row[i] = 2
            re_row[i] = best_re
        elif best_cl == 'stable':
            cls_row[i] = 0
            re_row[i] = best_re

    # Swollen branch sweep
    br_sw = track_branch(x_name, x_arr[::-1], pp_row, J_start=1.3, theta_start=0.3)
    for i in range(nx):
        ri = nx - 1 - i
        if not np.isnan(cls_row[ri]):
            continue
        ev = br_sw['evals'][i]
        if np.isnan(br_sw['J'][i]):
            continue
        cl = classify_point(ev)
        idx = np.argmax(ev.real)
        re_row[ri] = ev[idx].real
        if cl == 'hopf':
            cls_row[ri] = 1
            om_row[ri] = abs(ev[idx].imag)
        elif cl == 'mono':
            cls_row[ri] = 2
        elif cl == 'stable':
            cls_row[ri] = 0
    return cls_row, om_row, re_row


def stability_map_2d_parallel(x_name, x_arr, y_name, y_arr, p_base,
                              n_workers=None):
    n_workers = n_workers or N_WORKERS_DEFAULT
    ny, nx = len(y_arr), len(x_arr)
    cls_map = np.full((ny, nx), np.nan)
    om_map = np.full((ny, nx), np.nan)
    re_map = np.full((ny, nx), np.nan)
    args_list = [(y_arr[j], x_name, x_arr, y_name, p_base) for j in range(ny)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_stability_row, a): j for j, a in enumerate(args_list)}
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            cls_row, om_row, re_row = fut.result()
            cls_map[j], om_map[j], re_map[j] = cls_row, om_row, re_row
            done += 1
            if done % max(1, ny // 10) == 0:
                print(f"    {done}/{ny} rows done")
    return cls_map, om_map, re_map


# ═══════════════════════════════════════════════════════════════════
# Public load_or_compute interfaces (cached to data/fig1/_raw_*.npz)
# ═══════════════════════════════════════════════════════════════════

def _raw_path(name):
    return os.path.join(DATA_DIR, f"_raw_{name}.npz")


def load_or_compute_BiT_branches(p=None):
    """3 branches (lower, middle, upper) of J0(Bi_T) + eigenvalues."""
    cache = _raw_path("BiT_branches")
    if os.path.exists(cache):
        print(f"  Using cache: {cache}")
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}

    p = p or LSAParams()
    print("  Computing Bi_T bifurcation branches ...")
    BiT_vals = np.unique(np.concatenate([
        np.linspace(0.02, 0.50, 260),
        np.linspace(0.215, 0.235, 160),
        np.linspace(0.248, 0.266, 160),
    ]))
    br_coll = track_branch('Bi_T', BiT_vals, p, J_start=0.25, theta_start=1.0)
    br_swol = track_branch('Bi_T', BiT_vals[::-1], p, J_start=1.30, theta_start=0.2)
    br_swol = {k: v[::-1] for k, v in br_swol.items()}
    pw_branches, _mult = collect_pointwise_branches('Bi_T', BiT_vals, p, max_branches=3)
    br_mid_seed = pw_branches[1]
    mid_idx = np.where(~np.isnan(br_mid_seed['J']))[0]
    br_mid = {k: np.full_like(v, np.nan) for k, v in br_mid_seed.items()}
    if len(mid_idx) > 0:
        seed_i = int(mid_idx[len(mid_idx) // 2])
        br_mid_raw = track_branch_bidirectional(
            'Bi_T', BiT_vals, p, float(BiT_vals[seed_i]),
            float(br_mid_seed['J'][seed_i]), float(br_mid_seed['theta'][seed_i]))
        br_mid = filter_middle_branch(br_mid_raw, br_coll, br_swol, margin=0.015)

    out = dict(
        BiT=BiT_vals,
        J_lower=br_coll['J'], theta_lower=br_coll['theta'], evals_lower=br_coll['evals'],
        J_middle=br_mid['J'], theta_middle=br_mid['theta'], evals_middle=br_mid['evals'],
        J_upper=br_swol['J'], theta_upper=br_swol['theta'], evals_upper=br_swol['evals'],
    )
    np.savez(cache, **out)
    print(f"  Saved cache: {cache}")
    return out


def load_or_compute_dispersion(p=None, BiT_list=(0.04, 0.10, 0.40), n_k=1500, k_max=150.0):
    cache = _raw_path("dispersion")
    if os.path.exists(cache):
        print(f"  Using cache: {cache}")
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}

    p = p or LSAParams()
    print("  Computing dispersion relations ...")
    k_arr = np.linspace(0, k_max, n_k)
    sigmas_all = np.full((len(BiT_list), n_k, 3), np.nan, dtype=complex)
    J0s = np.full(len(BiT_list), np.nan)
    u0s = np.full(len(BiT_list), np.nan)
    th0s = np.full(len(BiT_list), np.nan)
    for i, BiT in enumerate(BiT_list):
        pp = replace(p, Bi_T=float(BiT))
        ss = find_uniform_ss(pp)
        if ss is None:
            print(f"    Bi_T={BiT}: no SS")
            continue
        J0, u0, t0 = ss[0]
        sigmas_all[i] = dispersion_eigenvalues(k_arr, J0, u0, t0, pp)
        J0s[i], u0s[i], th0s[i] = J0, u0, t0
        print(f"    Bi_T={BiT:.2f}: J0={J0:.3f}, max Re={np.max(sigmas_all[i,:,0].real):.2f}")

    out = dict(k=k_arr, BiT=np.asarray(BiT_list, float),
               sigmas=sigmas_all, J0=J0s, u0=u0s, theta0=th0s)
    np.savez(cache, **out)
    print(f"  Saved cache: {cache}")
    return out


def load_or_compute_DaBiT_map(p=None, n=60, smooth=True):
    cache = _raw_path(f"DaBiT_map_n{n}")
    if os.path.exists(cache):
        print(f"  Using cache: {cache}")
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    p = p or LSAParams()
    print(f"  Computing Da-Bi_T stability map ({n}×{n}) ...")
    Da_arr = np.linspace(0.5, 20.0, n)
    BiT_arr = np.linspace(0.02, 0.50, n)
    cls_map, om_map, re_map = stability_map_2d_parallel(
        'Da', Da_arr, 'Bi_T', BiT_arr, p)
    cls_map[np.isnan(cls_map)] = -2
    if smooth:
        cls_map = median_filter(cls_map, size=3)
    out = dict(Da=Da_arr, BiT=BiT_arr, cls=cls_map, omega=om_map, re=re_map)
    np.savez(cache, **out)
    print(f"  Saved cache: {cache}")
    return out


def load_or_compute_ScBiT_map(p=None, n=60, smooth=True):
    cache = _raw_path(f"ScBiT_map_n{n}")
    if os.path.exists(cache):
        print(f"  Using cache: {cache}")
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    p = p or LSAParams()
    print(f"  Computing S_chi-Bi_T stability map ({n}×{n}) ...")
    Sc_arr = np.linspace(0.1, 2.0, n)
    BiT_arr = np.linspace(0.02, 0.50, n)
    cls_map, om_map, re_map = stability_map_2d_parallel(
        'S_chi', Sc_arr, 'Bi_T', BiT_arr, p)
    cls_map[np.isnan(cls_map)] = -2
    if smooth:
        cls_map = median_filter(cls_map, size=3)
    out = dict(S_chi=Sc_arr, BiT=BiT_arr, cls=cls_map, omega=om_map, re=re_map)
    np.savez(cache, **out)
    print(f"  Saved cache: {cache}")
    return out


def load_or_compute_Da_period(p=None, n=100):
    cache = _raw_path("Da_period")
    if os.path.exists(cache):
        print(f"  Using cache: {cache}")
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    p = p or LSAParams()
    print(f"  Computing Da scan ({n} points) for period ...")
    Da_scan = np.linspace(1.0, 20.0, n)
    period_arr = np.full(n, np.nan)
    re_arr = np.full(n, np.nan)
    guesses = [(0.25, 1.0), (0.45, 2.0), (0.35, 1.5), (0.60, 2.5)]
    branches_da = [track_branch('Da', Da_scan, p, J_start=Jg, theta_start=tg)
                   for Jg, tg in guesses]
    br_rev_da = track_branch('Da', Da_scan[::-1], p, J_start=0.45, theta_start=2.0)
    branches_da.append({k: v[::-1] for k, v in br_rev_da.items()})
    for i in range(n):
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
            re_arr[i] = best_ev[idx].real
            omega = abs(best_ev[idx].imag)
            if omega > 0.01:
                period_arr[i] = 2 * np.pi / omega
    out = dict(Da=Da_scan, period=period_arr, re=re_arr)
    np.savez(cache, **out)
    print(f"  Saved cache: {cache}")
    return out
