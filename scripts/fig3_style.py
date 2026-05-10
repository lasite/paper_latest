"""
fig3_style.py — Canonical per-panel size + axes rectangle for Fig 3.

Mirrors `fig1_style.py` so panels of the two figures share identical
font sizes, line widths, axes margins. The 9 panels of fig3 (3×3 grid)
all share `(PANEL_W, PANEL_H)` and `AXES_RECT` so the composite tiler
can stack PNGs without per-panel rescaling.

For kymograph panels (cols 1, 2 of every row) a fixed `CBAR_RECT`
reserves a vertical strip in the right margin for the colorbar — every
panel uses the same colorbar pixel rectangle so the tiled composite
stays aligned even on the profile column (which leaves that strip
empty).
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt
from style_pub import set_style as _set_style_base


# ── Per-panel canonical size (matches fig1 for cross-figure consistency) ──
PANEL_W = 2.7
PANEL_H = 2.7
DPI_PNG = 300
DPI_PDF = 600

# ── Inner axes rectangle, figure-relative coords (matches fig1) ──
AXES_LEFT = 0.2607
AXES_BOTTOM = 0.1970
AXES_WIDTH = 0.5511
AXES_HEIGHT = 0.6578
AXES_RECT = (AXES_LEFT, AXES_BOTTOM, AXES_WIDTH, AXES_HEIGHT)

# ── Colorbar rectangle (used by kymograph panels) ──
# Right of the axes spine (axes-right at AXES_LEFT + AXES_WIDTH = 0.8118),
# leaves room for cbar tick labels in (CBAR_LEFT + CBAR_WIDTH, 1.0).
CBAR_LEFT = 0.83
CBAR_WIDTH = 0.03
CBAR_RECT = (CBAR_LEFT, AXES_BOTTOM, CBAR_WIDTH, AXES_HEIGHT)

# ── Panel label (figure-relative coords; identical pixel position per panel) ──
LABEL_X = 0.04
LABEL_Y = 0.95
LABEL_FONTSIZE = 10

# ── Output directory ──
FIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "Figure", "fig3"))


def set_style():
    """Apply project base style + panel-uniformity overrides."""
    _set_style_base()
    plt.rcParams.update({
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 5.5,
        # Override the global 'tight' bbox (set by style_pub) so every
        # saved panel PNG has the same pixel dimensions.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0,
    })


def new_panel_fig(with_cbar=False):
    """Return `(fig, ax)` or `(fig, ax, cax)` with absolutely-positioned
    axes (and optional colorbar). Same axes pixel rectangle in every panel."""
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AXES_RECT)
    if with_cbar:
        cax = fig.add_axes(CBAR_RECT)
        return fig, ax, cax
    return fig, ax


def add_panel_label(ax, label):
    """Place '(x)' label at fixed figure-relative coords (uniform pixel
    position across the 3×3 tile)."""
    fig = ax.figure
    fig.text(LABEL_X, LABEL_Y, f"({label})",
             fontsize=LABEL_FONTSIZE, fontweight='bold',
             va='top', ha='left')


def save_panel(fig, name):
    """Save panel without bbox_inches='tight' so all PNGs have identical
    (PANEL_W*DPI_PNG, PANEL_H*DPI_PNG) pixel dimensions."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out + ".pdf", dpi=DPI_PDF, bbox_inches=None, pad_inches=0)
    fig.savefig(out + ".png", dpi=DPI_PNG, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out}.pdf / .png")
