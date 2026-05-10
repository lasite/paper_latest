#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_optimized.py — 优化版参数扫描

优化点：
  1. 稀疏 Jacobian (jac_sparsity) — scipy BDF 自动图着色，
     N=121 时 Jacobian FD 调用从 364 次降至 ~15 次 (~24x 加速)
  2. multiprocessing.Pool 并行 — 196 个参数点同时运行
  3. 默认 N 降至 51（扫描精度足够，可用 --N 覆盖）
  4. 计时报告

用法：
  python scan_optimized.py                        # 默认: Bi_c x Da, N=51, 20进程
  python scan_optimized.py --nx 8 --ny 8          # 快速测试
  python scan_optimized.py --N 121 --workers 8    # 高精度
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import lil_matrix, csc_matrix


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:
    N: int = 51               # 扫描默认 N=51，比原来 121 快 ~8x
    t_end: float = 300.0
    n_save: int = 3000
    method: str = "BDF"
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    max_step: float = 0.5     # 比原来 0.25 宽松，对振荡检测足够

    phi_p0: float = 0.15
    chi_inf: float = 0.60
    S_chi: float = 1.00
    chi1: float = 1.10
    Omega_e: float = 0.12
    ell: float = 0.01

    Da: float = 4.0              # Optimized: 14→4 for full-domain u penetration
    delta: float = 0.08
    alpha: float = 0.20
    Gamma_A: float = 1.5         # Minimal Arrhenius (was 7.0); cleaner oscillation
    eps_T: float = 0.03
    arrh_exp_cap: float = 60.0   # Arrhenius 指数上限，降低可减少刚性

    # Hill 函数热激活（方案C）
    use_hill: bool = False        # True: 替换 Arrhenius 为 Hill 函数
    theta_c: float = 0.5          # 半激活温度（θ=θ_c 时 Hill部分=0.5）
    n_hill: float = 4.0           # Hill 系数（陡峭度，n≥2）
    hill_eps: float = 0.05        # 基底反应率（保证冷相自发点火，但远小于1）

    Bi_mu: float = 1.00
    Bi_c: float = 0.70           # Optimized: 0.25→0.70
    Bi_T: float = 0.10           # Optimized: 0.90→0.10
    B_vol:  float = 0.0   # 体积 u 源项: W_t   += B_vol*(J - W)
    B_Tvol: float = 0.0   # 体积散热项: θ_t   -= B_Tvol*θ  (分布式冷却)
    Bi_c_vol: float = 0.0 # 体积均匀供给: W_t  += Bi_c_vol*J*(1-u)  (分布式化学供给)
    Bi_J_vol: float = 0.0 # 体积溶剂交换: logJ_t -= Bi_J_vol*(μ - μ_b)/J (分布式溶胀平衡)
    m_b: float = 0.0
    auto_set_m_b: bool = True

    m_act: float = 6.0
    m_diff: float = 2.0          # Optimized: 6→2 for better u diffusion in swollen state
    m_mob: float = 1.0    # mobility exponent: M = M0*(1-phi)^m_mob; 0=constant
    M0: float = 1.0
    D0: float = 2.0              # Optimized: 1→2 for enhanced u diffusivity
    C0: float = 1.0
    K0: float = 1.0

    J_init: float = 1.30
    u_init: float = 0.02
    theta_init: float = 0.0
    eps_J: float = 5.0e-3
    eps_u: float = 5.0e-3
    eps_theta: float = 1.0e-4

    J_min: float = 0.18
    J_max: float = 8.0
    u_floor: float = 1.0e-12
    phi_floor: float = 1.0e-10
    phi_ceiling: float = 1.0 - 1.0e-10
    theta_clip: float = 25.0


# ---------------------------------------------------------------------------
# Sparsity pattern (key optimization)
# ---------------------------------------------------------------------------

def make_jac_sparsity(N: int) -> csc_matrix:
    """
    3N x 3N Jacobian 稀疏模式。
    State: [J_0..N-1, W_0..N-1, theta_0..N-1]

    Block bandwidth:
      (J,J):     2  — via J_xx in chemical potential
      (J,theta): 1  — chi(theta) coupling
      (W,J):     2  — swelling flux advects reactant
      (W,W):     1  — diffusion
      (W,theta): 1  — chi coupling
      (theta,theta): 1 — heat diffusion
    """
    size = 3 * N
    S = lil_matrix((size, size), dtype=np.float64)

    bw_table = {
        (0, 0): 2, (0, 1): -1, (0, 2): 1,
        (1, 0): 2, (1, 1): 1,  (1, 2): 1,
        (2, 0): -1,(2, 1): -1, (2, 2): 1,
    }

    for (kr, kc), w in bw_table.items():
        if w < 0:
            continue
        for i in range(N):
            row = kr * N + i
            for dj in range(-w, w + 1):
                j = i + dj
                if 0 <= j < N:
                    S[row, kc * N + j] = 1.0

    return csc_matrix(S)


# ---------------------------------------------------------------------------
# Helpers & constitutive laws  (identical physics to original scan script)
# ---------------------------------------------------------------------------

def cell_centers(N: int) -> np.ndarray:
    dx = 1.0 / N
    return (np.arange(N) + 0.5) * dx


def phi_from_J(J, p, phi_hard_ceil=0.995):
    """phi = phi_p0 / J，J 天然为正（logJ 公式），phi < phi_hard_ceil。"""
    J_safe = np.maximum(J, p.phi_p0 / phi_hard_ceil)
    return p.phi_p0 / J_safe


def harmonic_mean(a, b, eps=1e-30):
    return 2.0 * a * b / np.maximum(a + b, eps)


def laplacian_neumann(a, dx):
    out = np.empty_like(a)
    out[1:-1] = (a[2:] - 2*a[1:-1] + a[:-2]) / dx**2
    out[0]    = 2*(a[1]  - a[0])  / dx**2
    out[-1]   = 2*(a[-2] - a[-1]) / dx**2
    return out


def local_chem_pot(J, theta, p):
    phi = phi_from_J(J, p)
    m_mix = np.log(1 - phi) + phi + (p.chi_inf + p.S_chi*theta + p.chi1*phi) * phi**2
    m_el  = p.Omega_e * (J - 1.0/J)
    return m_mix + m_el


def finalize_params(p: Params) -> Params:
    if p.auto_set_m_b:
        J0 = np.array([p.J_init]); th0 = np.array([p.theta_init])
        p = replace(p, m_b=float(local_chem_pot(J0, th0, p)[0]))
    return p


def thermal_factor(theta, p):
    if p.use_hill:
        # 修正Arrhenius（方案B+）：thermal = hill_eps + exp(Γ_A·θ/(1+θ)) - 1
        # θ=0 时 = hill_eps（极小基底，允许缓慢冷相反应和自发点火）
        # θ>0 时 = hill_eps + exp(...) - 1（指数热失控特性保留）
        # hill_eps ≪ 1 控制冷相 Thiele 模数
        denom = 1.0 + p.eps_T * np.maximum(theta, -1.0/p.eps_T * 0.95)
        exp_val = np.clip(p.Gamma_A * theta / denom, -p.arrh_exp_cap, p.arrh_exp_cap)
        return p.hill_eps + np.maximum(np.exp(exp_val) - 1.0, 0.0)
    else:
        denom = 1.0 + p.eps_T * np.maximum(theta, -1.0/p.eps_T * 0.95)
        exp   = np.clip(p.Gamma_A * theta / denom, -p.arrh_exp_cap, p.arrh_exp_cap)
        return np.exp(exp)


def reaction_rate(u, theta, J, p):
    phi = phi_from_J(J, p)
    act = np.maximum(1 - phi, 1e-12)**p.m_act
    return np.maximum(u, p.u_floor) * act * thermal_factor(theta, p)


# ---------------------------------------------------------------------------
# Semi-discrete RHS  —  logJ 状态变量（天然保证 J > 0，稀疏 FD 安全）
# State: y = [logJ_0..N-1, W_0..N-1, theta_0..N-1]
# ---------------------------------------------------------------------------

_LOG_J_MAX = np.log(6.0)

def rhs_mol_logJ(t, y, p):
    n  = p.N
    dx = 1.0 / n
    log_J_min = np.log(p.phi_p0 * 1.02)

    logJ  = np.clip(y[:n],    log_J_min, _LOG_J_MAX)
    W     = y[n:2*n]
    theta = y[2*n:]

    J = np.exp(logJ)
    u = np.maximum(W / J, p.u_floor)

    phi = phi_from_J(J, p)

    # chemical potential
    m_local = local_chem_pot(J, theta, p)
    m       = m_local - p.ell**2 * laplacian_neumann(J, dx)

    # swelling flux q
    q = np.zeros(n+1)
    M_cell = p.M0 * np.maximum(1 - phi, 1e-12)**p.m_mob
    M_face = harmonic_mean(M_cell[:-1], M_cell[1:])
    q[1:n] = -M_face * (m[1:] - m[:-1]) / dx
    q[n]   = p.Bi_mu * (m[-1] - p.m_b)

    # logJ_t = -q_x / J + volumetric solvent exchange
    logJ_t = -(q[1:] - q[:-1]) / (dx * J) \
             - p.Bi_J_vol * (m_local - p.m_b) / J

    # reaction
    R = reaction_rate(u, theta, J, p)

    # reactant flux
    nflux = np.zeros(n+1)
    D_cell = p.D0 * np.maximum(1 - phi, 1e-12)**p.m_diff
    D_face = harmonic_mean(D_cell[:-1], D_cell[1:])
    q_int  = q[1:n]
    u_up   = np.where(q_int >= 0, u[:-1], u[1:])
    nflux[1:n] = q_int * u_up - p.delta * D_face * (u[1:] - u[:-1]) / dx
    nflux[n]   = p.Bi_c * (u[-1] - 1.0)
    # 体积源: B_vol*(J - W) = B_vol*(1 - u_raw)*J, 直接用W(光滑,无截断)
    # 体积均匀供给: Bi_c_vol*J*(1-u) = Bi_c_vol*(J - W)
    W_t = -(nflux[1:] - nflux[:-1]) / dx - p.Da * J * R \
          + p.B_vol * (J - W) \
          + p.Bi_c_vol * (J - W)

    # heat flux
    h = np.zeros(n+1)
    h[1:n] = -p.alpha * p.K0 * (theta[1:] - theta[:-1]) / dx
    h[n]   = p.Bi_T * theta[-1]
    theta_t = (-(h[1:] - h[:-1]) / dx + p.Da * J * R) / p.C0 \
              - p.B_Tvol * theta   # 体积散热: 每点向θ=0的浴槽散热

    return np.concatenate([logJ_t, W_t, theta_t])


# ---------------------------------------------------------------------------
# Simulation (logJ + sparse Jacobian)
# ---------------------------------------------------------------------------

def initial_state(p: Params) -> Tuple[np.ndarray, np.ndarray]:
    x  = cell_centers(p.N)
    log_J_min = np.log(p.phi_p0 * 1.02)
    J0 = np.maximum(p.J_init + p.eps_J * np.cos(np.pi * x),
                    np.exp(log_J_min) + 1e-6)
    u0 = np.maximum(p.u_init + p.eps_u * np.cos(np.pi * x), p.u_floor)
    t0 = p.theta_init + p.eps_theta * x
    logJ0 = np.log(J0)
    W0    = J0 * u0
    return x, np.concatenate([logJ0, W0, t0])


def make_fixed_fd_jac(rhs_fn, n3: int):
    """
    固定步长前向 FD Jacobian。
    h = max(1e-8, 1e-6*|y_i|)，不使用 scipy 的自适应步长（避免溢出）。
    配合稀疏模式跳过全零列，减少 ~(1-density)*n3 次 RHS 调用。
    """
    sparsity_cols = None   # lazy init（在 worker 进程里初始化）

    def jac(t, y):
        nonlocal sparsity_cols
        f0 = rhs_fn(t, y)
        J  = np.zeros((n3, n3))
        cols = sparsity_cols if sparsity_cols is not None else range(n3)
        for i in cols:
            hi  = max(1e-8, 1e-6 * abs(y[i]))
            yp  = y.copy(); yp[i] += hi
            J[:, i] = (rhs_fn(t, yp) - f0) / hi
        return J

    return jac


def make_sparse_fd_jac(rhs_fn, S_pattern, n3: int,
                       atol_h: float = 1.0e-8, rtol_h: float = 1.0e-6):
    """
    Graph-colored sparse FD Jacobian.

    Same fixed-step forward FD as `make_fixed_fd_jac` (h = max(atol_h,
    rtol_h*|y_i|), avoids scipy `num_jac` overflow on this stiff problem)
    but groups non-overlapping columns via `group_columns` so each
    Jacobian costs only ~bandwidth RHS evaluations instead of n3.
    Returns a csc_matrix so scipy.integrate.BDF uses sparse LU
    (factorisation O(n3·bw²) instead of O(n3³)).
    """
    from scipy.optimize._numdiff import group_columns

    S_csc   = S_pattern.tocsc()
    indptr  = S_csc.indptr.astype(np.int32, copy=True)
    indices = S_csc.indices.astype(np.int32, copy=True)
    nnz     = int(S_csc.nnz)

    # group_columns returns int array: groups[c] = group id for col c
    groups = group_columns(S_pattern)
    n_groups = int(groups.max()) + 1
    cols_in_group = [np.where(groups == g)[0].astype(np.int32)
                     for g in range(n_groups)]

    # Pre-extract the row indices for each column (already in CSC layout)
    col_rows = [indices[indptr[c]:indptr[c + 1]] for c in range(n3)]

    def jac(t, y):
        f0 = rhs_fn(t, y)
        data = np.zeros(nnz, dtype=np.float64)
        for cols_g in cols_in_group:
            hs = np.maximum(atol_h, rtol_h * np.abs(y[cols_g]))
            yp = y.copy()
            yp[cols_g] += hs
            df = rhs_fn(t, yp) - f0
            for c, h in zip(cols_g, hs):
                rs = col_rows[c]
                data[indptr[c]:indptr[c + 1]] = df[rs] / h
        return csc_matrix((data, indices, indptr), shape=(n3, n3))

    return jac, n_groups


def simulate(p: Params) -> Dict:
    p = finalize_params(p)
    x, y0 = initial_state(p)
    n3     = 3 * p.N

    rhs_fn = lambda t, y: rhs_mol_logJ(t, y, p)

    # ★ 固定步长 FD Jacobian（scipy adaptive FD 在极刚性问题下会溢出）
    # ★ Graph-colored sparse Jacobian → BDF uses sparse LU
    #   (~10 RHS calls/Jac instead of 3N; LU O(N·bw²) instead of O(N³))
    S          = make_jac_sparsity(p.N)
    jac_sparse, _n_color_groups = make_sparse_fd_jac(rhs_fn, S, n3)

    sol = solve_ivp(
        fun=rhs_fn,
        jac=jac_sparse,            # csc_matrix → sparse LU
        t_span=(0.0, p.t_end),
        y0=y0,
        t_eval=np.linspace(0, p.t_end, p.n_save),
        method=p.method,
        rtol=p.rtol, atol=p.atol,
        max_step=p.max_step,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    n = p.N
    log_J_min = np.log(p.phi_p0 * 1.02)
    J     = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
    W     = sol.y[n:2*n]
    theta = sol.y[2*n:]
    u     = np.maximum(W / J, p.u_floor)

    return {"x": x, "t": sol.t, "J": J, "u": u, "theta": theta,
            "phi": phi_from_J(J, p), "success": True, "nfev": sol.nfev}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def tail_slice(n_t, frac0=0.60):
    return slice(max(0, int(frac0 * n_t)), n_t)


def detrend(t, y):
    c = np.polyfit(t, y, 1)
    return y - np.polyval(c, t)


def oscillation_metrics(t, y, frac0=0.60, amp_floor=1e-3):
    sl = tail_slice(len(t), frac0)
    tt, yy = t[sl], y[sl]
    if len(tt) < 20:
        return {"amp": 0.0, "period": np.nan, "n_peaks": 0, "oscillatory": False}
    yd  = detrend(tt, yy)
    amp = float(np.max(yd) - np.min(yd))
    prom = max(0.15*amp, amp_floor)
    mdist = max(3, len(yd)//20)
    peaks,  _ = find_peaks( yd, prominence=prom, distance=mdist)
    troughs,_ = find_peaks(-yd, prominence=prom, distance=mdist)
    period = np.nan; peak_cv = np.nan
    if len(peaks) >= 2:
        dT = np.diff(tt[peaks])
        period = float(np.mean(dT))
        peak_cv = float(np.std(dT)/np.mean(dT)) if len(dT) >= 2 else 0.0
    osc = (amp > amp_floor and len(peaks) >= 2 and len(troughs) >= 2
           and (np.isnan(peak_cv) or peak_cv < 0.40))
    return {"amp": amp, "period": period, "n_peaks": int(len(peaks)), "oscillatory": osc}


def classify_run(data: Dict) -> Dict:
    J, u, theta = data["J"], data["u"], data["theta"]
    t = data["t"]
    J_mean     = np.mean(J, axis=0)
    theta_mean = np.mean(theta, axis=0)
    J_std      = np.std(J,     axis=0)
    theta_std  = np.std(theta, axis=0)

    th_m = oscillation_metrics(t, theta_mean, amp_floor=2e-3)
    J_m  = oscillation_metrics(t, J_mean,     amp_floor=2e-4)

    osc  = bool(th_m["oscillatory"] or J_m["oscillatory"])
    sl   = tail_slice(len(t))
    non_uni = max(np.mean(J_std[sl]), np.mean(theta_std[sl])) > 2e-3

    theta_f = float(theta_mean[-1])
    ts = "hot" if theta_f > 0.25 else ("cold" if theta_f < 0.05 else "warm")

    if osc and non_uni:       label = "oscillatory_nonuniform"
    elif osc:                 label = "oscillatory_uniform"
    elif non_uni:             label = f"steady_{ts}_nonuniform"
    else:                     label = f"steady_{ts}_uniform"

    def _f(x): return float(x) if np.isfinite(x) else float("nan")

    return {
        "label": label, "is_oscillatory": int(osc), "is_nonuniform": int(non_uni),
        "thermal_state": ts,
        "theta_amp": _f(th_m["amp"]),     "J_amp": _f(J_m["amp"]),
        "theta_period": _f(th_m["period"] or float("nan")),
        "J_period":     _f(J_m["period"]  or float("nan")),
        "theta_peaks": th_m["n_peaks"],   "J_peaks": J_m["n_peaks"],
        "J_mean_final": _f(J_mean[-1]),
        "theta_mean_final": theta_f,
        "u_mean_final": _f(np.mean(u, axis=0)[-1]),
        "nfev": data.get("nfev", -1),
    }


# ---------------------------------------------------------------------------
# Worker function (module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

_BASE_PARAMS: Params = None   # set in child init

def _worker_init(base_params_dict):
    """每个 worker 进程初始化时接收 base_params（spawn 安全）。"""
    global _BASE_PARAMS
    _BASE_PARAMS = Params(**base_params_dict)


def _worker_run(task):
    """单个参数点任务，在 worker 进程中执行。"""
    x_param, xv, y_param, yv = task
    p = replace(_BASE_PARAMS, **{x_param: float(xv), y_param: float(yv)})
    p = finalize_params(p)
    row_base = {x_param: float(xv), y_param: float(yv)}
    try:
        data = simulate(p)
        info = classify_run(data)
        return {**row_base, **info, "status": "ok"}
    except Exception as e:
        return {**row_base,
                "label": "solve_failed", "is_oscillatory": 0, "is_nonuniform": 0,
                "thermal_state": "failed", "theta_amp": float("nan"),
                "J_amp": float("nan"), "theta_period": float("nan"),
                "J_period": float("nan"), "theta_peaks": 0, "J_peaks": 0,
                "J_mean_final": float("nan"), "theta_mean_final": float("nan"),
                "u_mean_final": float("nan"), "nfev": -1,
                "status": f"failed: {e}"}


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------

def sweep_parallel(base_params: Params,
                   x_param: str, x_vals: np.ndarray,
                   y_param: str, y_vals: np.ndarray,
                   outdir: Path, n_workers: int = None) -> List[Dict]:

    tasks = [(x_param, float(xv), y_param, float(yv))
             for yv in y_vals for xv in x_vals]
    n_total = len(tasks)
    n_workers = n_workers or min(n_total, os.cpu_count() or 4)

    # spawn 避免 CUDA fork 问题
    ctx = mp.get_context("spawn")
    base_dict = base_params.__dict__.copy()

    print(f"  Launching {n_total} tasks on {n_workers} workers (spawn) ...")
    t0 = time.perf_counter()

    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(base_dict,)
    ) as pool:
        rows = pool.map(_worker_run, tasks, chunksize=1)

    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s  ({elapsed/n_total:.2f}s/task avg)")

    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, outdir / "scan_results.csv")
    return rows


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _phase_code(label: str) -> int:
    return {"steady_cold_uniform": 0, "steady_cold_nonuniform": 1,
            "steady_warm_uniform": 2, "steady_warm_nonuniform": 3,
            "steady_hot_uniform":  4, "steady_hot_nonuniform":  5,
            "oscillatory_uniform": 6, "oscillatory_nonuniform": 7,
            "solve_failed": -1}.get(label, -1)


def _to_grid(rows, x_param, x_vals, y_param, y_vals, key, fill=np.nan):
    grid = np.full((len(y_vals), len(x_vals)), fill, dtype=float)
    xm = {float(v): i for i, v in enumerate(x_vals)}
    ym = {float(v): j for j, v in enumerate(y_vals)}
    for r in rows:
        i, j = xm[float(r[x_param])], ym[float(r[y_param])]
        try: grid[j, i] = float(r.get(key, fill))
        except: pass
    return grid


def plot_scan(rows, x_param, x_vals, y_param, y_vals, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    ext = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

    specs = [
        ("phase",         [_phase_code(str(r["label"])) for r in rows], "Phase code",       None),
        ("oscillation",   [float(r.get("is_oscillatory",0)) for r in rows], "Oscillatory",  [0,1]),
        ("theta_period",  None, "Period (theta)",   None),
        ("theta_amp",     None, "Amplitude (theta)",None),
        ("theta_final",   None, "Final <theta>",    None),
    ]

    keys_map = {"phase": None, "oscillation": None,
                "theta_period": "theta_period", "theta_amp": "theta_amp",
                "theta_final": "theta_mean_final"}

    for name, _, title, vlim in specs:
        key = keys_map[name]
        if key:
            grid = _to_grid(rows, x_param, x_vals, y_param, y_vals, key)
        else:
            grid = np.full((len(y_vals), len(x_vals)), np.nan)
            xm = {float(v): i for i, v in enumerate(x_vals)}
            ym = {float(v): j for j, v in enumerate(y_vals)}
            for r in rows:
                i, j = xm[float(r[x_param])], ym[float(r[y_param])]
                try:
                    if name == "phase":
                        grid[j,i] = _phase_code(str(r["label"]))
                    elif name == "oscillation":
                        grid[j,i] = float(r.get("is_oscillatory", 0))
                except: pass

        fig, ax = plt.subplots(figsize=(7, 5))
        kw = {"vmin": vlim[0], "vmax": vlim[1]} if vlim else {}
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=ext, **kw)
        ax.set_xlabel(x_param); ax.set_ylabel(y_param); ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_map.png", dpi=160)
        plt.close(fig)

    print(f"  Plots saved to {outdir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="优化版参数扫描 (稀疏Jac + 多进程)")
    ap.add_argument("--x-param",  default="Bi_c")
    ap.add_argument("--x-min",    type=float, default=0.05)
    ap.add_argument("--x-max",    type=float, default=0.80)
    ap.add_argument("--nx",       type=int,   default=14)
    ap.add_argument("--y-param",  default="Da")
    ap.add_argument("--y-min",    type=float, default=4.0)
    ap.add_argument("--y-max",    type=float, default=24.0)
    ap.add_argument("--ny",       type=int,   default=14)
    ap.add_argument("--N",        type=int,   default=51, help="空间网格数 (默认51)")
    ap.add_argument("--t-end",    type=float, default=300.0)
    ap.add_argument("--workers",  type=int,   default=None, help="并行进程数 (默认=CPU核数-2)")
    ap.add_argument("--outdir",   default="scan_results_optimized")
    # 物理参数覆盖
    ap.add_argument("--BiT",      type=float, default=None, help="Bi_T (默认0.9)")
    ap.add_argument("--BiC",      type=float, default=None, help="Bi_c 基准值")
    ap.add_argument("--GammaA",   type=float, default=None)
    ap.add_argument("--Schi",     type=float, default=None)
    ap.add_argument("--Da",       type=float, default=None)
    ap.add_argument("--alpha",    type=float, default=None)
    ap.add_argument("--arrh-cap", type=float, default=None,
                    help="Arrhenius 指数上限 arrh_exp_cap (默认60)")
    return ap.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.BiT      is not None: overrides["Bi_T"]         = args.BiT
    if args.BiC      is not None: overrides["Bi_c"]         = args.BiC
    if args.GammaA   is not None: overrides["Gamma_A"]      = args.GammaA
    if args.Schi     is not None: overrides["S_chi"]        = args.Schi
    if args.Da       is not None: overrides["Da"]           = args.Da
    if args.alpha    is not None: overrides["alpha"]        = args.alpha
    if args.arrh_cap is not None: overrides["arrh_exp_cap"] = args.arrh_cap

    p = finalize_params(Params(
        N=args.N, t_end=args.t_end,
        n_save=max(1000, int(args.t_end * 10)),
        **overrides,
    ))

    x_vals = np.linspace(args.x_min, args.x_max, args.nx)
    y_vals = np.linspace(args.y_min, args.y_max, args.ny)
    outdir = Path(args.outdir)
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    print("=" * 60)
    print("scan_optimized.py  — 稀疏Jacobian + 多进程并行")
    print("=" * 60)
    print(f"  N={p.N}, t_end={p.t_end}, method={p.method}")
    print(f"  Scan: {args.x_param} ∈ [{x_vals[0]:.3g}, {x_vals[-1]:.3g}] × {args.nx}")
    print(f"        {args.y_param} ∈ [{y_vals[0]:.3g}, {y_vals[-1]:.3g}] × {args.ny}")
    print(f"  Total points: {args.nx * args.ny},  workers: {n_workers}")
    n3 = 3 * p.N
    S  = make_jac_sparsity(p.N)
    print(f"  Jac sparsity: {S.nnz}/{n3**2} = {S.nnz/n3**2*100:.1f}%  "
          f"(dense needs {n3+1} FD cols, sparse ~{int(S.nnz**0.5)+5})")
    print()

    t_total = time.perf_counter()
    rows = sweep_parallel(p, args.x_param, x_vals, args.y_param, y_vals,
                          outdir, n_workers)
    elapsed_total = time.perf_counter() - t_total

    n_ok  = sum(1 for r in rows if r.get("status") == "ok")
    n_osc = sum(int(r.get("is_oscillatory", 0)) for r in rows if r.get("status") == "ok")
    print(f"\n  成功: {n_ok}/{len(rows)},  振荡: {n_osc}/{n_ok or 1}")
    print(f"  总耗时: {elapsed_total:.1f}s")

    plot_scan(rows, args.x_param, x_vals, args.y_param, y_vals, outdir)
    print("\n完成。")


if __name__ == "__main__":
    main()
