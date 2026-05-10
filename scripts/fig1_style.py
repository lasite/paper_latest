"""
fig1_style.py — Canonical per-panel size + axes rectangle for Fig 1.

All `make_fig1[a-f].py` scripts must import `new_panel_fig` and
`add_panel_label` from here so that:

(1) Outer panel size is identical: PANEL_W × PANEL_H inches at fixed DPI →
    every PNG has the same pixel dimensions.

(2) Inner axes rectangle is identical: every panel uses
    `fig.add_axes(AXES_RECT)` (absolute figure-relative coords) instead of
    `add_subplot + subplots_adjust`. This guarantees the PLOT BOX itself
    (where data is drawn) has identical width/height across panels even
    when content (twinx, legends, long y-labels) varies.

(3) Panel labels are positioned in figure-relative coords via `fig.text(...)`
    so (a), (b), ... land at the EXACT same pixel position in every panel.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt
from style_pub import set_style as _set_style_base


# ── Per-panel canonical size (inches) ──
# Outer panel = 2.7 × 2.7 in; inner axes box held constant at 1.488 × 1.776 in
# (same as the previous 2.4-in design). The extra 0.3 in / 0.3 in of margin
# goes to the left (room for ylabels like 'Re(σ_1)' / 'max Re(σ)') and the
# top (room for the panel label outside the axes — no longer competes with
# the topmost y-tick).
PANEL_W = 2.7
PANEL_H = 2.7
DPI_PNG = 300
DPI_PDF = 600

# ── Inner axes rectangle, figure-relative coords ──
# Absolute: AXES_WIDTH × PANEL_W = 1.488 in, AXES_HEIGHT × PANEL_H = 1.776 in.
# Margins (in inches): left 0.704, right 0.508, top 0.392, bottom 0.532.
# All panels use this rectangle, including ones without twinx — the empty
# right margin on those is the price of a uniform tile.
AXES_LEFT = 0.2607
AXES_BOTTOM = 0.1970
AXES_WIDTH = 0.5511
AXES_HEIGHT = 0.6578
AXES_RECT = (AXES_LEFT, AXES_BOTTOM, AXES_WIDTH, AXES_HEIGHT)

# ── Panel label (figure-relative coords; identical pixel position per panel) ──
# At LABEL_X=0.04, label sits at ~0.11 in from the panel left edge, well to
# the LEFT of the y-tick labels (which live at ~0.45-0.65 in). At LABEL_Y=0.95,
# label sits in the top margin ABOVE the axes box (axes top is at fig y=0.855).
LABEL_X = 0.04
LABEL_Y = 0.95
LABEL_FONTSIZE = 10

# ── Color scheme ──
C_HOPF = '#2ca02c'    # green
C_MONO = '#d62728'    # red
C_STABLE = '#1f77b4'  # blue
C_NO_SS = '#dddddd'   # light gray
C_SADDLE = '#555555'  # dark gray
C_PERIOD = '#9467bd'  # purple
C_OMEGA = '#9467bd'

FIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "Figure", "fig1"))


def set_style():
    """Apply the project base style (fonts, line widths, etc.)."""
    _set_style_base()
    plt.rcParams.update({
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 5.5,
        # CRITICAL: style_pub sets savefig.bbox='tight' which auto-trims
        # whitespace per panel, producing PNGs of inconsistent pixel sizes
        # (panels with twinx end up wider). Override so every panel is
        # saved at exactly figsize × DPI px.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0,
    })


def new_panel_fig():
    """Return `(fig, ax)` with the canonical axes rectangle absolutely
    positioned. Use `fig.add_axes(AXES_RECT)` so every panel's plot box
    occupies the same pixel rectangle, regardless of content."""
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AXES_RECT)
    return fig, ax


def add_panel_label(ax, label):
    """Place '(x)' panel label in **figure-relative** coords so every panel's
    label lands at the same pixel position when tiled. With the 2.7-in panel
    layout, the label sits in the outer top-left margin (above the axes,
    left of the y-tick labels) so a white bbox is no longer needed."""
    fig = ax.figure
    fig.text(LABEL_X, LABEL_Y, f"({label})",
             fontsize=LABEL_FONTSIZE, fontweight='bold',
             va='top', ha='left')


def save_panel(fig, name):
    """Save panel with bbox_inches=None (standard, NOT tight) so all PNGs
    share the same pixel dimensions: (PANEL_W*DPI_PNG, PANEL_H*DPI_PNG)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    # Explicit bbox_inches=None overrides any leftover rcParam default.
    fig.savefig(out + ".pdf", dpi=DPI_PDF, bbox_inches=None, pad_inches=0)
    fig.savefig(out + ".png", dpi=DPI_PNG, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out}.pdf / .png")
