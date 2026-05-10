#!/usr/bin/env python3
"""
parse_D1_partial.py — Recover the D1 dense basin scan from the partial
stdout of the killed scan_basin_C_dense.py run.

Reads the worker-completion lines (one per sim) from the captured
output file and reconstructs the (NJ x NT) Jm_term and attractor
grids, with NaN for the 2 sims that did not finish.

Output: data/fig6/basin_C_dense.npz   (same schema as the original
        scan would produce, so make_basin_C.py just works).
"""
import re
import sys
from pathlib import Path

import numpy as np

OUT_PATH = Path(
    r"C:\Users\Administrator\AppData\Local\Temp\claude"
    r"\E--wang-paper-latest"
    r"\045507e0-2090-44ac-9294-72f393aba345\tasks\b57wt6u5a.output"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fig6"

# Match lines like:
# [119/121] J0=0.86 ?0=3.00 -> attractor=1 (J=4.88) 3120s
LINE = re.compile(
    r"\[\s*\d+/\d+\] J0=([\d.]+) [^=]+0=([\d.]+) -> attractor=(-?\d+) "
    r"\(J=([\d.]+)\)"
)


def main():
    text = OUT_PATH.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in LINE.finditer(text):
        J0 = float(m.group(1))
        th0 = float(m.group(2))
        attr = int(m.group(3))
        Jt = float(m.group(4))
        rows.append((J0, th0, attr, Jt))
    print(f"  recovered {len(rows)} sim results from stdout")

    # Reconstruct the axis grids the script would have used
    # (--n_J 11 --n_theta 11, J in [0.20, 1.30], theta in [0.0, 10.0])
    J0_vals     = np.linspace(0.20, 1.30, 11)
    theta0_vals = np.linspace(0.0, 10.0, 11)
    NJ = len(J0_vals); NT = len(theta0_vals)

    Jm_term  = np.full((NT, NJ), np.nan)
    thm_term = np.full((NT, NJ), np.nan)
    attr     = np.full((NT, NJ), -1, dtype=np.int8)
    succ     = np.zeros((NT, NJ), dtype=bool)

    def closest_idx(value, axis):
        return int(np.argmin(np.abs(axis - value)))

    seen = 0
    for J0, th0, a, Jt in rows:
        i = closest_idx(J0, J0_vals)
        j = closest_idx(th0, theta0_vals)
        # Sanity check
        if abs(J0_vals[i] - J0) > 0.05 or abs(theta0_vals[j] - th0) > 0.5:
            print(f"  WARN: ({J0}, {th0}) not on grid (closest "
                  f"({J0_vals[i]}, {theta0_vals[j]}))")
            continue
        Jm_term[j, i]  = Jt
        thm_term[j, i] = np.nan  # not captured in stdout
        attr[j, i]     = a
        succ[j, i]     = (a >= 0)
        seen += 1

    print(f"  populated {seen} cells of {NJ*NT}")
    print(f"  missing cells: ", flush=True)
    for j in range(NT):
        for i in range(NJ):
            if attr[j, i] < 0:
                print(f"    J0={J0_vals[i]:.2f}, theta0={theta0_vals[j]:.1f}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "basin_C_dense.npz"
    np.savez_compressed(
        cache,
        J0_vals=J0_vals, theta0_vals=theta0_vals,
        Jm_term=Jm_term, thm_term=thm_term,
        attractor=attr, success=succ,
        cell_Bi_T=0.059, cell_S_chi=1.80,
        u0=0.5, N=121, t_end=200.0,
    )
    print(f"  Saved: {cache}")


if __name__ == "__main__":
    main()
