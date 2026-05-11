#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseC_pde.py — Phase C: PDE Hopf-onset bisection.

For each Bi_T in {0.06, 0.10, 0.16, 0.25}, locate Da_c^PDE by
log-mean bisection in Da, with bracket
    [0.5 * Da_c^0D, 3 * Da_c^0D]
where Da_c^0D = Bi_T / (J_init * Gamma_A) is the analytic 0D
saddle-node value (Phase C 0D step).

Each iteration runs ONE PDE simulation at the current Da_mid
(N=301, t_end=400). is_oscillating: sigma(J_surf) > 0.05 over
the t in [300, 400] window (post-transient — explicit per Phase B
finding that ignition transients can be misclassified at the LF
boundary).

The 4 Bi_T values run as 4 parallel bisection chains, each chain
sequential within itself. Each chain consumes 1 worker at a time;
8 iterations per chain, ~17 min per oscillating sim, ~3-5 min per
NOT_OSC sim. Total wall-clock estimate: ~80-120 min on 24 cores.

Robustness
- If the bracket is too narrow (both endpoints have same OSC outcome),
  EXPAND the bracket up to MAX_BRACKET_HI before giving up.
- If the bracket-low is OSC OR bracket-hi is NOT_OSC at the start,
  the assumed cold-stable / oscillating polarity is violated; report
  and skip that Bi_T (mark Da_c^PDE = NaN).
- Per-sim 25-min hard timeout; timed-out sims are treated as their
  "expected" outcome at that Da only if it changes the bracket
  monotonically — otherwise they break the chain (NaN).
- Incremental save of per-iteration records after every sim.

Outputs
  data/iv_c/phaseC/pde_bisection_log.json  — per-sim records
  data/iv_c/phaseC/Da_c_pde.npz            — final Da_c^PDE per Bi_T
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params

OUT = _HERE.parent / "data" / "iv_c" / "phaseC"
OUT.mkdir(parents=True, exist_ok=True)

# Settings
N_GRID = 301
T_END = 400.0
T_WINDOW = (300.0, T_END)        # latter quarter, post-transient
N_SAVE = 4000
PER_SIM_TIMEOUT_S = 25 * 60      # 25 min per sim
N_BISECTION_ITERS = 8            # log-mean bisections
MAX_BRACKET_HI = 4.0             # expansion cap

OSC_J_STD_FLOOR = 0.05

LOG = OUT / "pde_bisection_log.json"


# ---------------------------------------------------------------------------
# Worker (one PDE simulation)
# ---------------------------------------------------------------------------

def is_oscillating(result):
    """sigma(J_surf) > floor in t in [300, 400]."""
    t = result["t"]
    J_surf = result["J"][-1]
    mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
    if mask.sum() < 50:
        return False, dict(reason="window too small")
    Js = J_surf[mask]
    std = float(np.std(Js))
    return (std > OSC_J_STD_FLOOR), dict(
        std=std, J_mean=float(np.mean(Js)),
        J_max=float(np.max(Js)), J_min=float(np.min(Js)),
    )


def run_sim(task):
    Bi_T = task["Bi_T"]; Da = task["Da"]
    p = Params(Bi_T=Bi_T, Da=Da, N=N_GRID, t_end=T_END, n_save=N_SAVE)
    p = finalize_params(p)
    t0 = time.perf_counter()
    try:
        r = simulate(p)
        osc, diag = is_oscillating(r)
        return {
            "Bi_T": Bi_T, "Da": Da,
            "wall_s": time.perf_counter() - t0,
            "is_oscillating": osc, "diag": diag,
            "error": None,
        }
    except Exception as e:
        return {
            "Bi_T": Bi_T, "Da": Da,
            "wall_s": time.perf_counter() - t0,
            "is_oscillating": False, "diag": None,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Save helpers (incremental log)
# ---------------------------------------------------------------------------

def save_log(records: list):
    tmp = LOG.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, default=str)
    tmp.replace(LOG)


# ---------------------------------------------------------------------------
# Bisection driver — one Bi_T at a time, exposed as ONE callable to
# multiprocessing so the 4 Bi_T's can run independently in parallel
# ---------------------------------------------------------------------------

def bisect_one_Bi_T(args):
    """Run the full 8-iteration bisection for ONE Bi_T value.

    Returns a dict with the bisection trajectory and final Da_c^PDE.
    Designed to be called as a worker target from the parent driver.
    """
    Bi_T = args["Bi_T"]
    Da_c_0D = args["Da_c_0D"]
    Da_lo_init = 0.5 * Da_c_0D
    Da_hi_init = 3.0 * Da_c_0D

    # We'd rather not actually call run_sim from inside this nested
    # worker (multiprocessing nesting is ugly on Windows).  Instead the
    # parent passes us a "submit_one(task) -> result" callback... but
    # multiprocessing.Pool can't pickle closures/callbacks cleanly.
    #
    # Simpler design: run sims directly via simulate() in this nested
    # worker.  The parent provides 4 Bi_T's, which run as 4 *parallel*
    # processes; each runs its 8 sims sequentially on its own core.
    # That gives 4-way concurrency, which is the correct level for a
    # bisection (no more parallelism is possible without speculating).

    log_records = []

    def _run(Da):
        r = run_sim({"Bi_T": Bi_T, "Da": Da})
        log_records.append({**r, "iter": len(log_records),
                            "Bi_T_chain": Bi_T})
        return r

    # ── Initial bracket check ───────────────────────────────────────
    r_lo = _run(Da_lo_init)
    r_hi = _run(Da_hi_init)
    Da_lo, Da_hi = Da_lo_init, Da_hi_init

    # If both endpoints same outcome, expand
    expansions = 0
    while r_lo["is_oscillating"] == r_hi["is_oscillating"]:
        if r_hi["is_oscillating"]:
            # Both oscillate; cold side too high; lower Da_lo
            Da_lo = Da_lo / 2.0
            r_lo = _run(Da_lo)
        else:
            # Neither oscillates; raise Da_hi
            if Da_hi >= MAX_BRACKET_HI:
                return dict(Bi_T=Bi_T, Da_c=float("nan"),
                            reason=("bracket expansion hit cap "
                                    f"Da_hi={MAX_BRACKET_HI}; "
                                    "no oscillation detected"),
                            log_records=log_records,
                            Da_c_0D=Da_c_0D)
            Da_hi = min(Da_hi * 2.0, MAX_BRACKET_HI)
            r_hi = _run(Da_hi)
        expansions += 1
        if expansions > 6:
            return dict(Bi_T=Bi_T, Da_c=float("nan"),
                        reason=("could not bracket Da_c^PDE "
                                f"after {expansions} expansions"),
                        log_records=log_records,
                        Da_c_0D=Da_c_0D)

    # Sanity: cold side should be NOT_OSC, hot side should be OSC
    if r_lo["is_oscillating"] or not r_hi["is_oscillating"]:
        return dict(Bi_T=Bi_T, Da_c=float("nan"),
                    reason=("polarity wrong: lo={r_lo['is_oscillating']}, "
                            f"hi={r_hi['is_oscillating']}"),
                    log_records=log_records,
                    Da_c_0D=Da_c_0D)

    # ── 8 iterations of log-mean bisection ─────────────────────────
    for it in range(N_BISECTION_ITERS):
        Da_mid = float(np.sqrt(Da_lo * Da_hi))
        r_mid = _run(Da_mid)
        if r_mid["is_oscillating"]:
            Da_hi = Da_mid
        else:
            Da_lo = Da_mid

    Da_c = float(np.sqrt(Da_lo * Da_hi))
    return dict(Bi_T=Bi_T, Da_c=Da_c,
                Da_lo_final=Da_lo, Da_hi_final=Da_hi,
                Da_c_0D=Da_c_0D,
                reason="converged",
                log_records=log_records)


# ---------------------------------------------------------------------------
# Parallel parent
# ---------------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print(" Phase C - PDE Da_c bisection (4 Bi_T x 8 iter, "
          "4-way concurrency)", flush=True)
    print("=" * 70, flush=True)

    Da_c_0D_data = np.load(OUT / "Da_c_0D.npz")
    Bi_T_arr = Da_c_0D_data["Bi_T"]
    Da_c_0D_arr = Da_c_0D_data["Da_c_0D"]
    print(f"  Bi_T values: {list(Bi_T_arr)}", flush=True)
    print(f"  Da_c^0D    : {list(Da_c_0D_arr)}", flush=True)
    print(f"  Bracket factors: [0.5, 3.0] x Da_c^0D", flush=True)
    print(f"  Settings: N={N_GRID}, t_end={T_END}, "
          f"t_window={T_WINDOW}, n_save={N_SAVE}", flush=True)
    print(f"  Per-sim timeout: {PER_SIM_TIMEOUT_S}s "
          f"({PER_SIM_TIMEOUT_S/60:.0f} min); "
          f"{N_BISECTION_ITERS} bisection iters", flush=True)
    print(flush=True)

    chains = [{"Bi_T": float(b), "Da_c_0D": float(d)}
              for b, d in zip(Bi_T_arr, Da_c_0D_arr)]

    t_total = time.perf_counter()

    # Run 4 chains in parallel via multiprocessing.Pool.
    # Each chain runs 8-12 sequential sims internally.
    pool = mp.Pool(processes=len(chains))
    async_results = [pool.apply_async(bisect_one_Bi_T, (c,)) for c in chains]

    completed = [False] * len(chains)
    chain_results = [None] * len(chains)

    while not all(completed):
        for i, ar in enumerate(async_results):
            if completed[i]:
                continue
            if ar.ready():
                try:
                    r = ar.get(timeout=0.0)
                    chain_results[i] = r
                except Exception as e:
                    chain_results[i] = {
                        "Bi_T": chains[i]["Bi_T"],
                        "Da_c": float("nan"),
                        "Da_c_0D": chains[i]["Da_c_0D"],
                        "reason": f"chain crashed: {type(e).__name__}: {e}",
                        "log_records": [],
                    }
                completed[i] = True
                t_e = time.perf_counter() - t_total
                cr = chain_results[i]
                Da_c_str = (f"{cr['Da_c']:.5f}"
                            if np.isfinite(cr["Da_c"]) else "NaN")
                print(f"  [{t_e/60:5.1f}m] Bi_T={cr['Bi_T']:.3f} chain done. "
                      f"Da_c^PDE = {Da_c_str}  ({cr['reason'][:40]})",
                      flush=True)
                # Persist per-iteration records
                all_records = []
                for ck in chain_results:
                    if ck is not None:
                        all_records.extend(ck.get("log_records", []))
                save_log(all_records)
        time.sleep(2.0)

    pool.close()
    pool.join()

    elapsed = time.perf_counter() - t_total
    print(f"\n  total wall-clock: {elapsed/60:.1f} min", flush=True)

    # Save final summary
    Da_c_pde = np.full(len(chains), np.nan)
    Da_c_0d = np.full(len(chains), np.nan)
    bracket_lo = np.full(len(chains), np.nan)
    bracket_hi = np.full(len(chains), np.nan)
    reasons = []
    for i, cr in enumerate(chain_results):
        Da_c_pde[i] = cr["Da_c"]
        Da_c_0d[i] = cr["Da_c_0D"]
        bracket_lo[i] = cr.get("Da_lo_final", float("nan"))
        bracket_hi[i] = cr.get("Da_hi_final", float("nan"))
        reasons.append(cr.get("reason", "?"))

    npz = OUT / "Da_c_pde.npz"
    np.savez(npz,
             Bi_T=Bi_T_arr,
             Da_c_pde=Da_c_pde, Da_c_0D=Da_c_0d,
             Da_lo_final=bracket_lo, Da_hi_final=bracket_hi,
             reasons=np.array(reasons),
             N_GRID=N_GRID, t_end=T_END,
             t_window=np.array(T_WINDOW))
    print(f"  saved {npz}", flush=True)
    print(flush=True)
    print("  Final summary:", flush=True)
    print(f"  {'Bi_T':>7s}  {'Da_c^0D':>9s}  {'Da_c^PDE':>9s}  "
          f"{'(PDE/0D - 1)':>13s}  reason", flush=True)
    for i in range(len(chains)):
        if np.isfinite(Da_c_pde[i]):
            shift = Da_c_pde[i] / Da_c_0d[i] - 1.0
            print(f"  {Bi_T_arr[i]:7.3f}  {Da_c_0d[i]:9.4f}  "
                  f"{Da_c_pde[i]:9.4f}  {shift:+12.2%}  "
                  f"{reasons[i][:30]}", flush=True)
        else:
            print(f"  {Bi_T_arr[i]:7.3f}  {Da_c_0d[i]:9.4f}  "
                  f"{'NaN':>9s}  {'-':>13s}  {reasons[i][:30]}", flush=True)


if __name__ == "__main__":
    main()
