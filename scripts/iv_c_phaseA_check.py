#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseA_check.py — Phase A decision: amplitude scaling (iii) PASS/FAIL.

Reads data/iv_c/phaseA/amplitude_results.json and
data/iv_c/folds/S_chi_sweep.npz, computes Δθ_surf · S_chi at each oscillating
test point, compares to the analytic h(material) ≈ 2.34, prints a per-point
table and the PASS/FAIL verdict.

PASS criterion: max relative deviation across oscillating points < 20%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PHASE_A = Path(__file__).resolve().parents[1] / "data" / "iv_c" / "phaseA"
FOLDS = Path(__file__).resolve().parents[1] / "data" / "iv_c" / "folds"

PASS_THRESHOLD = 0.20   # max relative deviation


def main():
    results = json.loads((PHASE_A / "amplitude_results.json").read_text())
    folds = np.load(FOLDS / "S_chi_sweep.npz")

    h_arr = folds["delta_theta"] * folds["S_chi"]
    h_pred = float(np.nanmean(h_arr))           # ~2.34, exactly constant
    h_std = float(np.nanstd(h_arr))

    print("=" * 70)
    print(" Phase A — amplitude scaling (iii) decision")
    print("=" * 70)
    print(f"  Analytic h = delta_theta * S_chi     "
          f"= {h_pred:.4f} +/- {h_std:.4f}  (from Phase P)")
    print(f"  PASS threshold: max relative deviation < {PASS_THRESHOLD:.0%}")
    print()
    print(f"  {'Bi_T':>6}  {'S_chi':>6}  {'delta_theta':>12}  "
          f"{'meas h':>10}  {'rel err':>10}  status")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  ------")

    deviations = []
    rows = []
    not_osc = []

    for r in results:
        Bi_T = r["Bi_T"]
        S_chi = r["S_chi"]
        amp = r["amp"]
        if r["error"]:
            not_osc.append((Bi_T, S_chi, f"sim error: {r['error']}"))
            continue
        if amp is None or not amp.get("is_oscillating", False):
            tag = amp.get("reason", "no amp") if amp else "no amp"
            J_m = amp.get("J_surf_mean", float("nan")) if amp else float("nan")
            th_m = amp.get("theta_surf_mean", float("nan")) if amp else float("nan")
            not_osc.append((Bi_T, S_chi,
                            f"NOT_OSC J_mean={J_m:.3f} theta_mean={th_m:.3f}"))
            continue

        delta_theta = amp["delta_theta_surf"]
        meas_h = delta_theta * S_chi
        rel_err = abs(meas_h - h_pred) / h_pred
        status = "ok " if rel_err < PASS_THRESHOLD else "OUT"
        print(f"  {Bi_T:6.2f}  {S_chi:6.2f}  {delta_theta:12.4f}  "
              f"{meas_h:10.4f}  {rel_err:10.2%}  {status}")
        deviations.append(rel_err)
        rows.append({
            "Bi_T": Bi_T, "S_chi": S_chi,
            "delta_theta": delta_theta, "meas_h": meas_h,
            "rel_err": rel_err, "in_threshold": rel_err < PASS_THRESHOLD,
        })

    if not_osc:
        print()
        print(f"  Non-oscillating ({len(not_osc)}):")
        for bt, sc, why in not_osc:
            print(f"    Bi_T={bt:.2f} S_chi={sc:.1f}: {why}")

    if not deviations:
        print("\n  >>> Phase A FAIL — no oscillating points. <<<")
        sys.exit(1)

    n_osc = len(deviations)
    n_total = len(results)
    max_dev = max(deviations)
    mean_dev = float(np.mean(deviations))

    print()
    print(f"  oscillating: {n_osc}/{n_total}")
    print(f"  max relative deviation: {max_dev:.2%}")
    print(f"  mean relative deviation: {mean_dev:.2%}")

    PASS = max_dev < PASS_THRESHOLD

    # Save summary npz
    np.savez(PHASE_A / "phaseA_check.npz",
             h_pred=h_pred,
             Bi_T=np.array([r["Bi_T"] for r in rows]),
             S_chi=np.array([r["S_chi"] for r in rows]),
             delta_theta=np.array([r["delta_theta"] for r in rows]),
             meas_h=np.array([r["meas_h"] for r in rows]),
             rel_err=np.array([r["rel_err"] for r in rows]),
             max_dev=max_dev, mean_dev=mean_dev,
             n_osc=n_osc, n_total=n_total,
             pass_threshold=PASS_THRESHOLD)

    print()
    if PASS:
        print(f"  >>> Phase A PASS  (max_dev = {max_dev:.2%} < {PASS_THRESHOLD:.0%}) <<<")
    else:
        print(f"  >>> Phase A FAIL  (max_dev = {max_dev:.2%} >= {PASS_THRESHOLD:.0%}) <<<")
        # write FAILURE_REPORT
        report = PHASE_A / "FAILURE_REPORT.md"
        with open(report, "w") as f:
            f.write("# Phase A Failure Report\n\n")
            f.write(f"Analytic h = {h_pred:.4f} (from Phase P, std = {h_std:.4f})\n\n")
            f.write(f"max relative deviation: {max_dev:.2%}\n")
            f.write(f"mean relative deviation: {mean_dev:.2%}\n\n")
            f.write("## Per-point breakdown\n\n")
            f.write("| Bi_T | S_chi | delta_theta | meas h | rel err | in threshold? |\n")
            f.write("|------|-------|-------------|--------|---------|---------------|\n")
            for row in rows:
                f.write(f"| {row['Bi_T']:.2f} | {row['S_chi']:.2f} | "
                        f"{row['delta_theta']:.4f} | {row['meas_h']:.4f} | "
                        f"{row['rel_err']:.2%} | "
                        f"{'YES' if row['in_threshold'] else 'NO'} |\n")
            if not_osc:
                f.write("\n## Non-oscillating points\n\n")
                for bt, sc, why in not_osc:
                    f.write(f"- Bi_T={bt:.2f}, S_chi={sc:.1f}: {why}\n")
        print(f"  Failure report written to {report}")
        sys.exit(1)


if __name__ == "__main__":
    main()
