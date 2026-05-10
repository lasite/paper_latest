#!/usr/bin/env python3
r"""
scan_SNIC_S_chi.py — C3: fine S_chi scan near the upper-S_chi LCST-front
boundary at fixed Bi_T (working point row), to test SNIC vs homoclinic
period scaling.

At Bi_T = 0.10 (working-point row), the LCST-front cycle persists from
roughly S_chi ~ 0.1 up to S_chi ~ 1.3 in the existing 25x25 grid; just
above S_chi ~ 1.3 the cycle disappears.  This script samples 25 S_chi
values densely in [0.85, 1.30] (where the period diverges) and 5 more
points outside the band, runs the PDE from default cold IC, extracts
the steady-state cycle period, and stores it as a cache.

The renderer (make_SNIC_scaling.py) fits T(S_chi) against the two
canonical scenarios:
  SNIC      : T \propto (S_chi^c - S_chi)^{-1/2}
  homoclinic: T \propto -log(S_chi^c - S_chi)
and reports which model fits best.

Output: data/fig5/SNIC_scan_S_chi.npz
  S_chi_vals     : axis grid
  period         : extracted PDE period (NaN if not oscillating)
  J_amp          : surface swelling amplitude
  regime         : integer regime label
  Bi_T_fixed     : fixed Bi_T

Cost: ~30 sims at working-point parameters.  Each sim is ~60-80s, so
~3-5 min wall on 24 workers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

DATA_DIR = _HERE.parent / "data" / "fig5"

FIXED_BI_T = 0.10   # working-point row
N_GRID = 121
T_END  = 200.0
N_SAVE = 2000


def _worker(task):
    idx, S_chi = task
    from scan_optimized import (
        Params, finalize_params, simulate, classify_run,
    )
    from fig2_data import WORKING_POINT
    from scipy.signal import find_peaks

    p_dict = dict(WORKING_POINT)
    p_dict.update(Bi_T=FIXED_BI_T, S_chi=S_chi,
                  N=N_GRID, t_end=T_END, n_save=N_SAVE)
    p = Params(**p_dict)
    p = finalize_params(p)

    t0 = time.perf_counter()
    res = simulate(p)
    dt_w = time.perf_counter() - t0

    try:
        c = classify_run(res)
    except Exception as exc:
        c = {"regime": -1, "period": np.nan, "J_amp_max": np.nan}

    # Re-extract period directly from surface time series for higher
    # fidelity (classify_run uses J_mean across xi).
    period = float(c.get("period", np.nan))
    try:
        t = res["t"]
        J_surf = res["J"][-1, :]
        late = t > 0.5 * t.max()
        ts = t[late]
        Js = J_surf[late]
        peaks, _ = find_peaks(Js, prominence=0.05, distance=20)
        if len(peaks) >= 3:
            dts = np.diff(ts[peaks])
            period = float(np.median(dts))
    except Exception:
        pass

    return dict(
        idx=idx, S_chi=S_chi, regime=int(c.get("regime", -1)),
        period=period, J_amp=float(c.get("J_amp_max", np.nan)),
        dt=dt_w,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_band", type=int, default=22,
                    help="dense sample count in the SNIC-divergence band")
    ap.add_argument("--n_outside", type=int, default=6,
                    help="sparse sample count outside the band (sanity)")
    ap.add_argument("--Schi_band_min", type=float, default=0.85)
    ap.add_argument("--Schi_band_max", type=float, default=1.30)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("FIG4_WORKERS", 24)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "SNIC_scan_S_chi.npz"
    if cache.exists() and not args.force:
        print(f"  Cache exists: {cache} (use --force to overwrite)",
              flush=True)
        return

    band = np.linspace(args.Schi_band_min, args.Schi_band_max,
                       args.n_band)
    outside = np.linspace(0.20, args.Schi_band_min - 0.05,
                          args.n_outside)
    S_chi_vals = np.unique(np.concatenate([outside, band]))

    print(f"=== C3 SNIC scan at Bi_T={FIXED_BI_T} ===", flush=True)
    print(f"  S_chi grid ({len(S_chi_vals)} points): "
          f"[{S_chi_vals.min():.3f}, {S_chi_vals.max():.3f}]", flush=True)
    print(f"  workers: {args.workers}", flush=True)

    n = len(S_chi_vals)
    period = np.full(n, np.nan)
    regime = np.full(n, -1, dtype=int)
    J_amp = np.full(n, np.nan)

    tasks = [(k, float(S_chi_vals[k])) for k in range(n)]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, task): task for task in tasks}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            k = r["idx"]
            period[k] = r["period"]
            regime[k] = r["regime"]
            J_amp[k] = r["J_amp"]
            done += 1
            print(f"  [{done:>2}/{n}] S_chi={r['S_chi']:.3f} "
                  f"regime={r['regime']:>2d} period={r['period']:.2f} "
                  f"J_amp={r['J_amp']:.3f} {r['dt']:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {elapsed:.0f}s = {elapsed/60:.1f} min", flush=True)

    np.savez_compressed(
        cache,
        S_chi_vals=S_chi_vals,
        period=period, regime=regime, J_amp=J_amp,
        Bi_T_fixed=FIXED_BI_T,
        N=N_GRID, t_end=T_END,
    )
    print(f"  Saved: {cache}", flush=True)


if __name__ == "__main__":
    main()
