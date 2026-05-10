#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseA5_convergence.py — N convergence study at 3 representative
(Bi_T, S_chi) points spanning the LCST-front region.

Per (point, N), we measure
  T_PDE          — limit-cycle period (peak-to-peak in J_surf)
  delta_theta    — surface theta amplitude
  delta_J        — surface J amplitude
  xi_LCST        — innermost xi where phi crosses 0.5 during cycle
  theta_surf_max — top of cycle (should approach theta_up)
  theta_surf_min — bottom of cycle (should approach theta_lo)

Convergence: a quantity is "converged at N=k" if the relative change
|q(N_k) - q(N_{k-1})| / |q(N_k)| < TOL for all subsequent grids.

Robustness vs the previous version
----------------------------------
- Per-sim hard timeout: a sim that exceeds PER_SIM_TIMEOUT_S is abandoned
  (worker keeps running but its result is recorded as "TIMEOUT" and we
  move on). At end of run, pool.terminate() kills any still-running
  worker.
- Streaming output: results are printed and saved incrementally as soon
  as each sim completes, using multiprocessing.Pool.apply_async + a
  polling loop on AsyncResult.ready(). No more waiting for the slowest
  sim to flush.
- Incremental save: convergence_raw.json is rewritten after every result.

Outputs
-------
data/iv_c/phaseA5/convergence_raw.json  — per-sim records (incremental)
data/iv_c/phaseA5/convergence.npz       — reshaped grid (final)
data/iv_c/phaseA5/convergence_report.md — per-observable converged N
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseA5"
OUT.mkdir(parents=True, exist_ok=True)

POINTS = [
    {"label": "WP",       "Bi_T": 0.10, "S_chi": 1.0},
    {"label": "shallow",  "Bi_T": 0.05, "S_chi": 0.7},
    {"label": "nearSNIC", "Bi_T": 0.15, "S_chi": 1.3},
]
N_GRID_VALS = [51, 101, 151, 201, 301, 401, 501]
T_END = 400.0
T_WINDOW = (200.0, T_END)
N_SAVE = 4000

PER_SIM_TIMEOUT_S = 35 * 60     # 35 min hard timeout per sim
CONV_TOL = 0.05                 # 5% relative change criterion
PHI_LCST = 0.5
OSC_J_STD_FLOOR = 0.05


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _surface_in_window(result):
    t = result["t"]
    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    return mask, t[mask], result["J"][-1, mask], result["theta"][-1, mask]


def measure_period(result):
    _, t_w, J_surf, _ = _surface_in_window(result)
    if len(t_w) < 50 or np.std(J_surf) < OSC_J_STD_FLOOR:
        return float("nan"), 0
    yd = J_surf - J_surf.mean()
    amp = float(yd.max() - yd.min())
    prom = max(0.15 * amp, 1e-3)
    mdist = max(3, len(yd) // 20)
    peaks, _ = find_peaks(yd, prominence=prom, distance=mdist)
    if len(peaks) < 3:
        return float("nan"), int(len(peaks))
    periods = np.diff(t_w[peaks])
    return float(np.mean(periods)), int(len(peaks))


def measure_amplitudes(result):
    _, _, J_surf, th_surf = _surface_in_window(result)
    if len(J_surf) < 50:
        return None
    osc = bool(np.std(J_surf) >= OSC_J_STD_FLOOR)
    return {
        "is_oscillating": osc,
        "delta_theta": float(th_surf.max() - th_surf.min()) if osc else float("nan"),
        "delta_J": float(J_surf.max() - J_surf.min()) if osc else float("nan"),
        "theta_surf_max": float(th_surf.max()),
        "theta_surf_min": float(th_surf.min()),
        "J_surf_max": float(J_surf.max()),
        "J_surf_min": float(J_surf.min()),
        "theta_surf_mean": float(th_surf.mean()),
        "J_surf_mean": float(J_surf.mean()),
        "J_surf_std": float(np.std(J_surf)),
    }


def measure_xi_LCST(result):
    t = result["t"]
    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    if mask.sum() < 50:
        return float("nan")
    phi = result["phi"][:, mask]
    x = result["x"]
    phi_max_xi = phi.max(axis=1)
    above = phi_max_xi > PHI_LCST
    if not above.any():
        return 1.0
    if above.all():
        return 0.0
    return float(x[int(np.argmax(above))])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(task):
    label = task["label"]; Bi_T = task["Bi_T"]; S_chi = task["S_chi"]; N = task["N"]
    p = Params(Bi_T=Bi_T, S_chi=S_chi, N=N, t_end=T_END, n_save=N_SAVE)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        amp = measure_amplitudes(r)
        T, npeaks = measure_period(r)
        xi_L = measure_xi_LCST(r)
        return {
            "label": label, "Bi_T": Bi_T, "S_chi": S_chi, "N": N,
            "wall_s": time.perf_counter() - t0,
            "nfev": int(r.get("nfev", -1)),
            "period": T, "n_peaks": npeaks,
            "xi_LCST": xi_L, "amp": amp,
            "error": None,
        }
    except Exception as e:
        return {
            "label": label, "Bi_T": Bi_T, "S_chi": S_chi, "N": N,
            "wall_s": time.perf_counter() - t0, "nfev": -1,
            "period": float("nan"), "n_peaks": 0,
            "xi_LCST": float("nan"), "amp": None,
            "error": f"{type(e).__name__}: {e}",
        }


def make_timeout_record(task):
    return {
        "label": task["label"], "Bi_T": task["Bi_T"], "S_chi": task["S_chi"],
        "N": task["N"], "wall_s": PER_SIM_TIMEOUT_S, "nfev": -1,
        "period": float("nan"), "n_peaks": 0,
        "xi_LCST": float("nan"), "amp": None,
        "error": f"TIMEOUT (> {PER_SIM_TIMEOUT_S}s)",
    }


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def converged_N(N_arr, q_arr, tol=CONV_TOL):
    """Smallest N at which q is within `tol` of the reference (highest
    finite-N value), and stays within `tol` for every higher finite N.

    NaN entries (e.g. timed-out sims) are skipped — we do not penalize a
    grid for a missing intermediate value. Returns -1 if never within tol.
    """
    N_arr = np.asarray(N_arr)
    q_arr = np.asarray(q_arr, dtype=float)
    finite = np.isfinite(q_arr)
    if finite.sum() < 2:
        return -1

    ref_idx = int(np.where(finite)[0][-1])  # highest finite N
    ref = q_arr[ref_idx]
    if abs(ref) < 1e-12:
        return -1

    rel = np.abs(q_arr - ref) / abs(ref)
    # Find smallest N (in grid order) at which rel < tol AND every later
    # finite N also satisfies rel < tol.
    for k in range(len(N_arr)):
        if not finite[k] or rel[k] >= tol:
            continue
        ok = True
        for j in range(k + 1, len(N_arr)):
            if finite[j] and rel[j] >= tol:
                ok = False
                break
        if ok:
            return int(N_arr[k])
    return -1


# ---------------------------------------------------------------------------
# Save helpers (incremental)
# ---------------------------------------------------------------------------

RAW = OUT / "convergence_raw.json"


def save_incremental(results):
    """Atomic write: tmp file + rename."""
    tmp = RAW.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2, default=str)
    tmp.replace(RAW)


def write_grid_and_report(results, elapsed):
    obs_keys = ["period", "delta_theta", "delta_J",
                "xi_LCST", "theta_surf_max", "theta_surf_min"]
    grids = {pt["label"]: {k: np.full(len(N_GRID_VALS), np.nan) for k in obs_keys}
             for pt in POINTS}
    timeouts = []
    errors = []
    for r in results:
        if r is None:
            continue
        if r.get("error"):
            if "TIMEOUT" in r["error"]:
                timeouts.append((r["label"], r["N"]))
            else:
                errors.append((r["label"], r["N"], r["error"]))
        if r["N"] not in N_GRID_VALS or r["label"] not in [p["label"] for p in POINTS]:
            continue
        i = N_GRID_VALS.index(r["N"])
        g = grids[r["label"]]
        g["period"][i] = r.get("period", np.nan)
        g["xi_LCST"][i] = r.get("xi_LCST", np.nan)
        amp = r.get("amp")
        if amp:
            g["delta_theta"][i] = amp.get("delta_theta", np.nan)
            g["delta_J"][i] = amp.get("delta_J", np.nan)
            g["theta_surf_max"][i] = amp.get("theta_surf_max", np.nan)
            g["theta_surf_min"][i] = amp.get("theta_surf_min", np.nan)

    N_arr = np.array(N_GRID_VALS)
    payload = {"N_grid": N_arr,
               "point_labels": np.array([p["label"] for p in POINTS]),
               "Bi_T": np.array([p["Bi_T"] for p in POINTS]),
               "S_chi": np.array([p["S_chi"] for p in POINTS])}
    for pt in POINTS:
        for k in obs_keys:
            payload[f"{pt['label']}__{k}"] = grids[pt["label"]][k]
    np.savez(OUT / "convergence.npz", **payload)

    md = []
    md.append("# Phase A.5 - N convergence report\n")
    md.append(f"Tolerance: relative change between consecutive N "
              f"< {CONV_TOL:.0%} (and stays converged at all higher N).\n")
    md.append(f"\n_Settings_: `t_end={T_END}`, `t_window={T_WINDOW}`, "
              f"`n_save={N_SAVE}`, per-sim timeout = "
              f"{PER_SIM_TIMEOUT_S}s ({PER_SIM_TIMEOUT_S/60:.0f} min). "
              f"Total wall-clock {elapsed/60:.1f} min.\n")
    if timeouts:
        md.append(f"\n_Timeouts_: {len(timeouts)} sims abandoned: "
                  + ", ".join(f"{lbl}@N={N}" for lbl, N in timeouts) + "\n")
    if errors:
        md.append(f"\n_Errors_: {len(errors)}: "
                  + "; ".join(f"{lbl}@N={N} ({e[:40]})"
                              for lbl, N, e in errors) + "\n")

    for pt in POINTS:
        lbl = pt["label"]
        md.append(f"\n## Point `{lbl}` -- Bi_T={pt['Bi_T']}, S_chi={pt['S_chi']}\n")
        md.append("\n| N | T | dTh | dJ | xi_L | th_max | th_min |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|")
        g = grids[lbl]
        for i, N in enumerate(N_GRID_VALS):
            md.append(
                f"| {N} | {g['period'][i]:.3f} | {g['delta_theta'][i]:.3f}"
                f" | {g['delta_J'][i]:.3f} | {g['xi_LCST'][i]:.3f}"
                f" | {g['theta_surf_max'][i]:.3f} | {g['theta_surf_min'][i]:.3f} |"
            )
        md.append("")
        md.append(f"| observable | converged N (tol {CONV_TOL:.0%}) |")
        md.append("|---|---:|")
        for k in obs_keys:
            n_conv = converged_N(N_arr, g[k])
            tag = str(n_conv) if n_conv > 0 else "**not converged**"
            md.append(f"| {k} | {tag} |")

    md.append("\n## Aggregate -- minimum N to converge at all 3 points\n")
    md.append("| observable | min converged N | recommended phase |")
    md.append("|---|---:|---|")
    aggregate = {}
    for k in obs_keys:
        per_point = [converged_N(N_arr, grids[p["label"]][k]) for p in POINTS]
        if any(v < 0 for v in per_point):
            agg = -1
        else:
            agg = max(per_point)
        aggregate[k] = agg
        rec = ""
        if k == "period":
            rec = "Phase B (period scaling)"
        elif k == "xi_LCST":
            rec = "Phase D (front depth)"
        elif k == "delta_theta":
            rec = "Phase A (amplitude -- re-check?)"
        tag = str(agg) if agg > 0 else "not converged at any tested N"
        md.append(f"| {k} | {tag} | {rec} |")

    md.append("\n## Recommendation for downstream phases\n")
    n_amp = aggregate["delta_theta"]; n_per = aggregate["period"]; n_xi = aggregate["xi_LCST"]
    md.append("- **Phase A re-do?** If `delta_theta` already converged at "
              "N<=101 and within 5% of N=301 value, do NOT re-run Phase A. "
              "Otherwise re-run at the recommended N.")
    md.append(f"- **Phase B (period):** use N = "
              f"{max(101, n_per) if n_per>0 else 'N>=301 (not yet converged in tested range)'}")
    md.append(f"- **Phase C (onset):** N=151 typically sufficient (long-wave Hopf)")
    md.append(f"- **Phase D (front depth):** use N = "
              f"{max(151, n_xi) if n_xi>0 else 'N>=501 (not converged)'}")

    md_path = OUT / "convergence_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  saved {OUT / 'convergence.npz'}")
    print(f"  saved {md_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print(" Phase A.5 - N convergence study (with per-sim timeout)", flush=True)
    print("=" * 70, flush=True)
    tasks = [{**pt, "N": int(N)} for pt in POINTS for N in N_GRID_VALS]
    n_workers = min(len(tasks), 24)
    print(f"  points: {[p['label'] for p in POINTS]}", flush=True)
    print(f"  N values: {N_GRID_VALS}", flush=True)
    print(f"  total sims: {len(tasks)}, workers: {n_workers}", flush=True)
    print(f"  t_end={T_END}, t_window={T_WINDOW}, n_save={N_SAVE}", flush=True)
    print(f"  per-sim timeout: {PER_SIM_TIMEOUT_S}s ({PER_SIM_TIMEOUT_S/60:.0f} min)",
          flush=True)
    print(flush=True)

    t_total = time.perf_counter()
    pool = mp.Pool(processes=n_workers)
    pending = {}            # ar -> (idx, task, deadline)
    for i, t in enumerate(tasks):
        ar = pool.apply_async(run_one, (t,))
        pending[ar] = (i, t, time.perf_counter() + PER_SIM_TIMEOUT_S)

    results = [None] * len(tasks)
    printed = set()
    completed = 0

    def _print_one(r, t_elapsed):
        tag = (f"[{t_elapsed/60:5.1f}m] {r['label']:8s} "
               f"BiT={r['Bi_T']:.2f} S={r['S_chi']:.1f} N={r['N']:3d}")
        if r["error"]:
            print(f"  {tag}: {r['error'][:50]}  [{r['wall_s']:.1f}s]", flush=True)
            return
        amp = r["amp"]
        osc = bool(amp and amp.get("is_oscillating", False))
        if osc:
            print(f"  {tag}: T={r['period']:6.3f}  "
                  f"dTh={amp['delta_theta']:.3f}  dJ={amp['delta_J']:.3f}  "
                  f"xi_L={r['xi_LCST']:.3f}  npk={r['n_peaks']:2d}  "
                  f"[{r['wall_s']:5.1f}s]", flush=True)
        else:
            J_m = amp.get("J_surf_mean", float("nan")) if amp else float("nan")
            th_m = amp.get("theta_surf_mean", float("nan")) if amp else float("nan")
            print(f"  {tag}: NOT_OSC  J={J_m:.3f}  th={th_m:.3f}  "
                  f"xi_L={r['xi_LCST']:.3f}  [{r['wall_s']:5.1f}s]", flush=True)

    while completed < len(tasks):
        finished_now = []
        for ar, (i, t, deadline) in list(pending.items()):
            if ar.ready():
                try:
                    r = ar.get(timeout=0.0)
                except Exception as e:
                    r = {**make_timeout_record(t),
                         "error": f"exception: {type(e).__name__}: {e}"}
                results[i] = r
                finished_now.append(ar)
                completed += 1
            elif time.perf_counter() > deadline:
                results[i] = make_timeout_record(t)
                finished_now.append(ar)
                completed += 1

        for ar in finished_now:
            del pending[ar]

        # Print and save any newly-completed results
        for idx, r in enumerate(results):
            if r is None or idx in printed:
                continue
            _print_one(r, time.perf_counter() - t_total)
            printed.add(idx)

        if finished_now:
            save_incremental(results)

        if completed < len(tasks):
            time.sleep(1.0)

    elapsed = time.perf_counter() - t_total
    print(f"\n  total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)
    print("  terminating pool (kills any worker still running a timed-out sim)...",
          flush=True)
    pool.terminate()
    pool.join()
    print("  pool closed.", flush=True)

    save_incremental(results)
    write_grid_and_report(results, elapsed)


if __name__ == "__main__":
    main()
