#!/usr/bin/env python3
"""
bench_sparse.py — Verify the new sparse-LU path in scan_optimized.simulate().

Steps:
  1. Run the working point at N=41 with the (current) sparse path.
  2. Load the existing cache (data/fig2/cache.npz) which was built with
     the previous dense path; compare J/u/theta on the analysis window.
  3. Run a quick benchmark at N=121 and N=301 to gauge wall time.
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from fig2_data import WORKING_POINT, T_START, T_END
from scan_optimized import Params, simulate, make_jac_sparsity
from scipy.optimize._numdiff import group_columns


def time_run(N, t_end=None, n_save=None, label=""):
    p_dict = dict(WORKING_POINT)
    p_dict["N"] = int(N)
    if t_end is not None:
        p_dict["t_end"] = float(t_end)
    if n_save is not None:
        p_dict["n_save"] = int(n_save)
    p = Params(**p_dict)
    print(f"\n  [{label}] N={N}, t_end={p.t_end}, n_save={p.n_save}")
    S = make_jac_sparsity(N)
    n3 = 3 * N
    n_g = int(group_columns(S).max()) + 1
    print(f"        sparsity nnz={S.nnz}/{n3*n3}={S.nnz/(n3*n3)*100:.2f}%, "
          f"color groups={n_g}, bw≈{n_g}")
    t0 = time.perf_counter()
    res = simulate(p)
    dt = time.perf_counter() - t0
    print(f"        simulate(): {dt:.1f}s, nfev={res.get('nfev')}")
    return p_dict, res, dt


def crosscheck_N41():
    print("=" * 60)
    print("Cross-check: N=41 sparse vs cached dense result")
    print("=" * 60)
    cache_path = _HERE.parent.parent / "data" / "fig2" / "cache.npz"
    if not cache_path.exists():
        print(f"  No cache at {cache_path}; skipping cross-check.")
        return None

    z = np.load(cache_path)
    if z["J"].shape[0] != 41:
        print(f"  Cache has N={z['J'].shape[0]}, not 41; skipping cross-check.")
        return None

    p_dict, res, dt = time_run(41, label="N=41 sparse")
    t_old = z["t"]; J_old = z["J"]; u_old = z["u"]; th_old = z["theta"]
    t_new = res["t"]; J_new = res["J"]; u_new = res["u"]; th_new = res["theta"]
    if not np.allclose(t_old, t_new):
        print("  ! t arrays differ in shape/values; aligning by time window")
    idx_old = (t_old >= T_START) & (t_old <= T_END)
    idx_new = (t_new >= T_START) & (t_new <= T_END)
    print(f"  Window samples: old={idx_old.sum()}  new={idx_new.sum()}")

    def relmax(a, b):
        d = np.abs(a - b)
        s = np.maximum(np.abs(a), np.abs(b))
        s = np.where(s < 1e-12, 1.0, s)
        return float((d / s).max())

    if idx_old.sum() == idx_new.sum():
        rJ  = relmax(J_old[:, idx_old],  J_new[:, idx_new])
        ru  = relmax(u_old[:, idx_old],  u_new[:, idx_new])
        rth = relmax(th_old[:, idx_old], th_new[:, idx_new])
        print(f"  Max rel-diff over window:  J={rJ:.2e}  u={ru:.2e}  θ={rth:.2e}")
        ok = max(rJ, ru, rth) < 5e-3
        print(f"  Cross-check: {'PASS (<5e-3)' if ok else 'FAIL (>=5e-3)'}")
    return dt


def main():
    crosscheck_N41()

    # ── Benchmark at intermediate / target N
    print()
    print("=" * 60)
    print("Benchmark sparse path at higher N")
    print("=" * 60)
    # Short t_end for benchmarking only — full t_end will be used for the
    # final cache.  This lets us compare cost/step quickly.
    BENCH_T = 50.0
    BENCH_NS = 800
    _, _, dt121 = time_run(121, t_end=BENCH_T, n_save=BENCH_NS,
                           label="N=121 bench (short t_end)")
    _, _, dt301 = time_run(301, t_end=BENCH_T, n_save=BENCH_NS,
                           label="N=301 bench (short t_end)")

    # Extrapolate full run cost
    full_t_end = WORKING_POINT["t_end"]
    print()
    print(f"  Extrapolated full t_end={full_t_end} cost (linear in t_end):")
    print(f"    N=121 ≈ {dt121 * full_t_end / BENCH_T:.0f}s")
    print(f"    N=301 ≈ {dt301 * full_t_end / BENCH_T:.0f}s")


if __name__ == "__main__":
    main()
