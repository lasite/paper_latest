#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_fig5.py — Composite tile for Fig 5 (period + amplitude).

Only tiles the two pre-rendered panel PDFs/PNGs side by side.
Re-runs the panel scripts if their PNG outputs are missing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

FIG_DIR = _HERE.parent / "Figure" / "pub"


PANELS = [
    ("iv_c_fig5_panel_a.png", "iv_c_fig5_panel_a.py"),
    ("iv_c_fig5_panel_b.png", "iv_c_fig5_panel_b.py"),
]


def ensure_panels():
    for png_name, script_name in PANELS:
        png = FIG_DIR / png_name
        if not png.exists():
            print(f"   running {script_name} (missing {png_name})", flush=True)
            subprocess.run([sys.executable, str(_HERE / script_name)],
                           check=True)


def main():
    ensure_panels()

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(PRE_DOUBLE, PRE_DOUBLE * 0.42))

    for ax, (png_name, _) in zip(axes, PANELS):
        img = imread(FIG_DIR / png_name)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    pdf = FIG_DIR / "iv_c_fig5.pdf"
    png = FIG_DIR / "iv_c_fig5.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {pdf}", flush=True)
    print(f"   saved {png}", flush=True)


if __name__ == "__main__":
    main()
