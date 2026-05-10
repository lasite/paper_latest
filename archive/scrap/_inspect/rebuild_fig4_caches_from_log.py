#!/usr/bin/env python3
"""
rebuild_fig4_caches_from_log.py — Plan-B recovery.

The dual-grid sweep (run_fig4_grids_parallel.py) hung on 2 cells at the
SNIC bifurcation boundary (Bi_T ≈ 0.14, 0.17 at S_chi = 2.10). We killed
the master before it could write the .npz caches. This script parses the
master's stdout log file for the regime / J_amp_max / phi_max of every
completed cell and writes both grid caches with REG_FAILED for the 2
missing cells. Other diagnostics (surf_amp, phi_max_min, J_mean, J_eq,
period) were not printed and are filled with NaN — panel_e (period
heatmap) will degrade to a placeholder until those cells are
re-simulated. Panels (a, b, c, d, f) are unaffected.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from fig4_data import (
    BI_T_VALS, S_CHI_VALS, DA_VALS, _grid_path, REG_FAILED, REGIME_NAMES,
)

# Map regime name string → integer code by inverting REGIME_NAMES
NAME_TO_CODE = {v: k for k, v in REGIME_NAMES.items()}

LOG = Path(
    "C:/Users/Administrator/AppData/Local/Temp/claude/"
    "E--wang-paper-latest/045507e0-2090-44ac-9294-72f393aba345/"
    "tasks/bgw5kvu7l.output"
)

# A typical line:
#   [183/450] [main] Bi_T= 0.0590 S_chi= 1.6500        lcst_front  J_amp=2.131  phi_max=0.980  (xx min ...)
LINE_RE = re.compile(
    r"\[\s*\d+/\d+\]\s+\[(?P<grid>main|da)\]\s+"
    r"(?P<px>\w+)=\s*(?P<xv>[0-9.]+)\s+"
    r"(?P<py>\w+)=\s*(?P<yv>[0-9.]+)\s+"
    r"(?P<regime>\S+)\s+"
    r"J_amp=(?P<jamp>[0-9.eE+-]+|nan)\s+"
    r"phi_max=(?P<phimax>[0-9.eE+-]+|nan)"
)


def _nearest(vals, target):
    """Snap target to the closest value in vals (within tolerance)."""
    diffs = np.abs(vals - target)
    i = int(np.argmin(diffs))
    if diffs[i] > 1e-3:
        raise ValueError(f"no match for {target} in {vals}")
    return i


def main():
    if not LOG.exists():
        print(f"  Log not found: {LOG}")
        sys.exit(1)

    text = LOG.read_text(errors="replace")
    matches = LINE_RE.findall(text)
    print(f"  parsed {len(matches)} cell completions from log")

    # Allocate empty grids (NaN / REG_FAILED)
    GRIDS = {
        "main": dict(px="Bi_T",  xv=BI_T_VALS,
                     py="S_chi", yv=S_CHI_VALS),
        "da":   dict(px="Bi_T",  xv=BI_T_VALS,
                     py="Da",    yv=DA_VALS),
    }
    state = {}
    for k, g in GRIDS.items():
        NX, NY = len(g["xv"]), len(g["yv"])
        state[k] = dict(
            x=g["xv"], y=g["yv"], px=g["px"], py=g["py"],
            regime      = np.full((NY, NX), REG_FAILED, dtype=int),
            J_amp_max   = np.full((NY, NX), np.nan),
            surf_amp    = np.full((NY, NX), np.nan),
            phi_max     = np.full((NY, NX), np.nan),
            phi_max_min = np.full((NY, NX), np.nan),
            J_mean      = np.full((NY, NX), np.nan),
            J_eq        = np.full((NY, NX), np.nan),
            period      = np.full((NY, NX), np.nan),
        )

    n_seen = {"main": 0, "da": 0}
    n_unknown_regime = 0
    for grid, px, xv, py, yv, regime, jamp, phimax in matches:
        s = state[grid]
        try:
            i = _nearest(s["x"], float(xv))
            j = _nearest(s["y"], float(yv))
        except ValueError as e:
            print(f"    skip out-of-grid: {e}")
            continue
        code = NAME_TO_CODE.get(regime)
        if code is None:
            n_unknown_regime += 1
            continue
        s["regime"][j, i]    = code
        s["J_amp_max"][j, i] = float(jamp) if jamp != "nan" else np.nan
        s["phi_max"][j, i]   = float(phimax) if phimax != "nan" else np.nan
        n_seen[grid] += 1

    print(f"  main grid: {n_seen['main']}/{len(BI_T_VALS)*len(S_CHI_VALS)} cells filled")
    print(f"  da   grid: {n_seen['da']}/{len(BI_T_VALS)*len(DA_VALS)} cells filled")
    if n_unknown_regime:
        print(f"  WARNING: {n_unknown_regime} unknown-regime lines skipped")

    # Report missing cells
    for k, s in state.items():
        miss = np.where(s["regime"] == REG_FAILED)
        if miss[0].size > 0:
            print(f"\n  {k} grid missing cells (REG_FAILED):")
            for jj, ii in zip(miss[0], miss[1]):
                print(f"    {s['px']}={s['x'][ii]:.4f}  "
                      f"{s['py']}={s['y'][jj]:.4f}")

    # Save .npz with the same schema as fig4_data.build_grid
    for k, s in state.items():
        path = _grid_path(s["px"], s["py"])
        np.savez_compressed(
            path,
            x=s["x"], y=s["y"], regime=s["regime"],
            J_amp_max=s["J_amp_max"], surf_amp=s["surf_amp"],
            phi_max=s["phi_max"], phi_max_min=s["phi_max_min"],
            J_mean=s["J_mean"], J_eq=s["J_eq"], period=s["period"],
        )
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
