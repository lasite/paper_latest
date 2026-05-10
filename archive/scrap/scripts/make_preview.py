#!/usr/bin/env python3
"""
make_preview.py — Composite preview of all §IV restructure panels.

Reads the standalone PNG figures already rendered by the per-task
scripts and tiles them into a single image so the user can audit
all panels at once.  No simulation/computation; pure image collage.

Output: figures_pub/section_IV_preview.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_HERE = Path(__file__).resolve().parent
PUB = _HERE / "figures_pub"

PANELS = [
    ("B2 slow manifold + limit cycle", "slow_manifold.png"),
    ("C1 homogeneous SS branches",     "homogeneous_ss.png"),
    ("B3 period landscape",            "period_scaling.png"),
    ("C3 SNIC scaling",                "SNIC_scaling.png"),
    ("D1 basin boundary at C-cell",    "basin_C.png"),
    ("D2 IC-induced hysteresis",       "hysteresis_C_theta0.png"),
]


def main():
    available = [(name, path) for name, path in PANELS
                 if (PUB / path).exists()]
    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.0 * rows),
                             squeeze=False)
    fig.suptitle("§IV restructure — new panels (preview)", fontsize=11)

    for ax, (name, fname) in zip(axes.flatten(), available):
        img = mpimg.imread(PUB / fname)
        ax.imshow(img)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for k in range(len(available), rows * cols):
        axes.flatten()[k].axis("off")

    out = PUB / "section_IV_preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
