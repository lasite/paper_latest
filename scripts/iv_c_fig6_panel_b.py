#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fig6_panel_b.py — 5-way attractor map on (m_act, m_diff).

Categorical heatmap showing where each cell lands among: cycle,
overswollen_front, frozen_front, hot_runaway, cold_SS (plus TIMEOUT
and sim_error book-keeping). xi_LCST value is overlaid where
meaningful.

Source data: data/iv_c/phaseD/phaseD_check.npz
Cached data: data/iv_c/phaseE/fig6_panel_b.npz
Output:      Figure/pub/iv_c_fig6_panel_b.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE  # type: ignore
PANEL_W = PRE_DOUBLE / 2.0
PANEL_H = PANEL_W * 0.78

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

CACHE = _HERE.parent / "data" / "iv_c" / "phaseE"
CACHE.mkdir(parents=True, exist_ok=True)
FIG_DIR = _HERE.parent / "Figure" / "pub"
FIG_DIR.mkdir(parents=True, exist_ok=True)


CLASS_ORDER = [
    "cold_SS",
    "overswollen_uniform",
    "overswollen_front",
    "frozen_front",
    "cycle",
    "hot_runaway",
    "TIMEOUT",
    "sim_error",
]
COLORS = {
    "cold_SS":             "#1f3a93",
    "overswollen_uniform": "#74b9ff",
    "overswollen_front":   "#ffb142",
    "frozen_front":        "#e67e22",
    "cycle":               "#27ae60",
    "hot_runaway":         "#c0392b",
    "TIMEOUT":             "#2c2c2c",
    "sim_error":           "#7f8c8d",
}


def build_cache():
    src = np.load(_HERE.parent / "data" / "iv_c" / "phaseD" /
                  "phaseD_check.npz", allow_pickle=True)
    m_act = np.array(src["m_act"])
    m_diff = np.array(src["m_diff"])
    classification = np.array(src["classification"])
    xi_LCST = np.array(src["xi_LCST"])

    npz = CACHE / "fig6_panel_b.npz"
    np.savez(npz,
             m_act=m_act, m_diff=m_diff,
             classification=classification,
             xi_LCST=xi_LCST)
    return npz


def plot(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    m_act = d["m_act"]
    m_diff = d["m_diff"]
    classification = d["classification"]
    xi_LCST = d["xi_LCST"]

    classes_present = [c for c in CLASS_ORDER
                       if c in set(classification.flatten().tolist())]
    cls_to_idx = {c: i for i, c in enumerate(classes_present)}
    code = np.vectorize(lambda c: cls_to_idx.get(str(c), -1))(classification)

    cmap = ListedColormap([COLORS[c] for c in classes_present])
    bounds = np.arange(len(classes_present) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    im = ax.imshow(code, origin="lower", aspect="auto",
                   extent=[m_diff[0] - 0.5, m_diff[-1] + 0.5,
                           m_act[0] - 0.5, m_act[-1] + 0.5],
                   cmap=cmap, norm=norm)

    # Overlay xi values for cycle cells
    for i, ma in enumerate(m_act):
        for j, md in enumerate(m_diff):
            cls = str(classification[i, j])
            if cls == "cycle":
                ax.text(md, ma, f"{xi_LCST[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=5.5, color="white")
            elif cls in ("overswollen_front", "frozen_front"):
                ax.text(md, ma, f"{xi_LCST[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=5.5, color="black")

    ax.set_xticks(m_diff)
    ax.set_yticks(m_act)
    ax.set_xlabel(r"$m_\mathrm{diff}$")
    ax.set_ylabel(r"$m_\mathrm{act}$")
    ax.set_title("(b) 5-way attractor map")

    handles = [Patch(color=COLORS[c], label=c.replace("_", " "))
               for c in classes_present]
    ax.legend(handles=handles, fontsize=5.5,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.8)
    fig.tight_layout()

    pdf = FIG_DIR / "iv_c_fig6_panel_b.pdf"
    png = FIG_DIR / "iv_c_fig6_panel_b.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {pdf}", flush=True)
    print(f"   saved {png}", flush=True)


def main():
    npz = build_cache()
    print(f"   cached {npz}", flush=True)
    plot(npz)


if __name__ == "__main__":
    main()
