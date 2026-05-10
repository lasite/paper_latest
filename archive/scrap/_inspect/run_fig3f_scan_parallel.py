#!/usr/bin/env python3
"""
run_fig3f_scan_parallel.py — Drive make_fig3f_xi_peak_scan in parallel.

The default load_scans() in make_fig3f_xi_peak_scan.py runs each
parameter value sequentially. With N=121 each simulation costs ~2.5 min,
24 of them serially is ~60 min. Running 6 sims at a time finishes the
whole scan in ~10–15 min wall time.

Each sim is wrapped in a worker process to avoid contaminating the
parent's matplotlib / numpy state.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _worker(task):
    param, val = task
    # Lazy-import inside the worker so spawn is clean.
    from make_fig3f_xi_peak_scan import _run_one
    xp, xL, ok = _run_one(param, float(val))
    return (param, float(val), xp, xL, bool(ok))


def main():
    from make_fig3f_xi_peak_scan import (
        SCANS, _cache_path, SCAN_N, SCAN_T_END,
    )
    DATA_DIR = _HERE.parent.parent / "data" / "fig3"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  scan resolution: N={SCAN_N}, t_end={SCAN_T_END}")

    # Build the flat task list across all SCANS that don't already have caches.
    tasks = []
    for s in SCANS:
        param = s["param"]
        cache = Path(_cache_path(param))
        if cache.exists():
            print(f"  cache exists for {param}: {cache.name}")
            continue
        for v in s["vals"]:
            tasks.append((param, float(v)))
    if not tasks:
        print("  Nothing to do — all caches already present.")
        return
    print(f"  Submitting {len(tasks)} simulations ...")

    n_workers = min(len(tasks), 6)
    t0 = time.perf_counter()
    results = {s["param"]: {"vals": s["vals"],
                             "xi_peak": np.full_like(s["vals"], np.nan, dtype=float),
                             "xi_LCST": np.full_like(s["vals"], np.nan, dtype=float),
                             "ok":      np.zeros_like(s["vals"], dtype=bool)}
               for s in SCANS}
    val_to_idx = {s["param"]: {float(v): k for k, v in enumerate(s["vals"])}
                  for s in SCANS}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for f in as_completed(futs):
            param, val, xp, xL, ok = f.result()
            i = val_to_idx[param][float(val)]
            results[param]["xi_peak"][i] = xp
            results[param]["xi_LCST"][i] = xL
            results[param]["ok"][i]      = ok
            print(f"    [{param}={val:.4f}]  ξ_peak={xp:.4f}  ξ_LCST={xL:.4f}  ok={ok}")

    dt = time.perf_counter() - t0
    print(f"\n  Total {dt:.0f}s ({dt/60:.1f} min) on {n_workers} workers")

    for s in SCANS:
        path = _cache_path(s["param"])
        if Path(path).exists():
            continue
        np.savez_compressed(path, **results[s["param"]])
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
