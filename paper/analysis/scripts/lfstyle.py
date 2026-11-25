"""Shared figure style, layout helpers and export gate for the manuscript figures.

All figures are drawn with Matplotlib at final physical size (180 mm wide,
Arial/Helvetica, 5-7 pt text, 8 pt bold lowercase panel letters). Axes are
placed in millimetres from the top-left corner so that comparable panels share
exact edges; the Nature-figure alignment gate is run before export when the
audit scripts are available.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "source_data"
FIGDIR = ROOT / "figures"
QADIR = FIGDIR / "qa"

MM = 1 / 25.4
FULL_W = 180.0  # mm, double column

# One restrained palette for every figure: method families keep their colour.
C = {
    "net": "#0F4D92",        # feed-forward network (this work)
    "net_soft": "#8FB3DA",
    "fit": "#2A9D8F",        # per-exposure Gaussian fitting (this work)
    "fit_soft": "#A8D8D1",
    "classic": "#767676",    # backprojection / plane sweep
    "classic_soft": "#CFCECE",
    "bad": "#B64342",        # disrupted or mismatched condition
    "bad_soft": "#E9A6A1",
    "alt": "#9A4D8E",        # alternative route (output conversion, learned lookup)
    "alt_soft": "#D9B8D3",
    "gold": "#C98A1A",
    "truth": "#272727",
    "ink": "#272727",
    "mid": "#767676",
    "light": "#CFCECE",
    "faint": "#EDEDED",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C["ink"],
    "axes.labelcolor": C["ink"],
    "text.color": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "mathtext.default": "regular",
    "hatch.linewidth": 0.5,
})


def new_figure(height_mm: float, width_mm: float = FULL_W):
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)  # type: ignore[attr-defined]
    return fig


def ax_mm(fig, left: float, top: float, width: float, height: float, **kw):
    """Add axes whose plot area is given in mm from the figure's top-left corner."""
    W, H = fig._mm
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H], **kw)


def panel_label(fig, ax, letter: str, dx: float = -9.0, dy: float = -3.2):
    """Bold lowercase panel letter anchored dx/dy mm from the axes' top-left corner."""
    W, H = fig._mm
    pos = ax.get_position()
    x = pos.x0 + dx / W
    y = pos.y1 - dy / H
    fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")


def title(ax, text: str, pad: float = 3.0, **kw):
    ax.set_title(text, loc="left", pad=pad, fontsize=6.8, fontweight="bold", **kw)


def image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def boot_ci(values, n: int = 10000, seed: int = 20261121, stat=np.mean):
    """Percentile 95% interval from resampling whole fields (rows of `values`)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stats = stat(values[idx], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def jitter(n: int, width: float, seed: int = 0):
    """Deterministic horizontal jitter for strip plots (display only; no data are simulated)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def strip(ax, x, values, color, width=0.16, size=5, alpha=0.55, seed=0, zorder=2, **kw):
    values = np.asarray(values, dtype=float)
    ax.scatter(x + jitter(len(values), width, seed), values, s=size, color=color,
               alpha=alpha, linewidths=0, zorder=zorder, rasterized=len(values) > 400, **kw)


def mean_bar(ax, x, value, color=None, half=0.26, lw=1.3, zorder=4):
    ax.plot([x - half, x + half], [value, value], color=color or C["ink"], lw=lw,
            solid_capstyle="butt", zorder=zorder)


def save(fig, name: str, axes=None, panel_ids=None, row_groups=None, column_groups=None,
         exemptions=(), exclude_axes=()):
    """Run the panel-alignment gate, then export PDF, SVG, PNG and TIFF."""
    FIGDIR.mkdir(exist_ok=True)
    QADIR.mkdir(exist_ok=True)
    skill = Path(os.environ.get(
        "NATURE_FIGURE_SCRIPTS", Path.home() / ".codex/skills/nature-figure/scripts"))
    report = None
    if axes is not None and len(axes) > 1 and (skill / "audit_panel_alignment.py").exists():
        sys.path.insert(0, str(skill))
        from audit_panel_alignment import require_matplotlib_panel_alignment  # type: ignore
        report = require_matplotlib_panel_alignment(
            fig, axes=list(axes), panel_ids=list(panel_ids),
            row_groups=row_groups, column_groups=column_groups,
            exemptions=list(exemptions), exclude_axes=list(exclude_axes),
            json_out=QADIR / f"{name}.alignment.json",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    base = str(FIGDIR / name)
    fig.savefig(base + ".pdf")                      # vector, editable TrueType text (pdf.fonttype 42)
    fig.savefig(base + ".svg")                      # vector, text kept as text (svg.fonttype none)
    fig.savefig(base + ".png", dpi=400)             # preview
    fig.savefig(base + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    status = report.get("verdict") if isinstance(report, dict) else "not run"
    print(f"saved {name}  (alignment: {status})")
    return report


def mixed_row(panels, reason="mixed-modality row: panel widths and gutters follow the content "
              "(image plates, square axes, plots); shared top and bottom edges are still enforced"):
    """Exemption for rows whose panels intentionally differ in width."""
    return {"panels": list(panels), "checks": ["panel-width", "horizontal-gutter"], "reason": reason}


def load_json(rel: str):
    return json.loads((DATA / rel).read_text())


def read_csv(rel: str):
    import csv
    with (DATA / rel).open() as f:
        return list(csv.DictReader(f))
