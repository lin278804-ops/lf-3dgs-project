"""Fig. 2 | Gaussian particles are recoverable from sub-aperture images and
amortization makes their recovery faster and more complete.

Core conclusion: an additive Gaussian particle field can be fitted to the
observed views without particle labels, and a network that predicts the same
particles in one pass is both about two orders of magnitude faster and more
complete, with depth remaining the hard axis.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from lfmetrics import match
from lfstyle import (C, DATA, ax_mm, image_axes, load_json, mixed_row, new_figure, panel_label,
                     read_csv, save, title)

FIELD = "sample_00000_diam50um_conc0p025"
CROP = (slice(66, 114), slice(86, 146))  # rows, columns of the native 186 x 232 view
SLAB_Y = (-1.5, 1.5)                    # mm
WIN_X = (0.0, 10.0)


def load_fields():
    out = {}
    for d in sorted(glob.glob(str(DATA / "fig2/fields/sample_*"))):
        out[os.path.basename(d)] = np.load(os.path.join(d, "figure_data.npz"))
    return out


def slab_panel(ax, gt, pred, color, label):
    m = match(pred, gt)
    in_gt = (gt[:, 1] >= SLAB_Y[0]) & (gt[:, 1] < SLAB_Y[1]) & (gt[:, 0] >= WIN_X[0]) & (gt[:, 0] < WIN_X[1])
    in_pr = (pred[:, 1] >= SLAB_Y[0]) & (pred[:, 1] < SLAB_Y[1]) & (pred[:, 0] >= WIN_X[0]) & (pred[:, 0] < WIN_X[1])
    matched_gt = np.zeros(len(gt), bool)
    matched_gt[m["gt_idx"]] = True
    matched_pr = np.zeros(len(pred), bool)
    matched_pr[m["pred_idx"]] = True
    # links between matched pairs whose reference lies in the slab
    for pi, gi in zip(m["pred_idx"], m["gt_idx"]):
        if in_gt[gi]:
            ax.plot([gt[gi, 0], pred[pi, 0]], [gt[gi, 2], pred[pi, 2]], color=color, lw=0.6, zorder=2)
    ax.scatter(gt[in_gt & matched_gt, 0], gt[in_gt & matched_gt, 2], s=9, facecolors="white",
               edgecolors=C["truth"], linewidths=0.5, zorder=3)
    sel = np.isin(np.arange(len(pred)), m["pred_idx"][in_gt[m["gt_idx"]]])
    ax.scatter(pred[sel, 0], pred[sel, 2], s=5, color=color, linewidths=0, zorder=4)
    ax.scatter(gt[in_gt & ~matched_gt, 0], gt[in_gt & ~matched_gt, 2], s=9, facecolors="white",
               edgecolors=C["bad"], linewidths=0.7, zorder=3)
    ax.scatter(pred[in_pr & ~matched_pr, 0], pred[in_pr & ~matched_pr, 2], s=9, marker="x",
               color=C["bad"], linewidths=0.7, zorder=4)
    n_gt = int(in_gt.sum())
    n_hit = int((in_gt & matched_gt).sum())
    n_fp = int((in_pr & ~matched_pr).sum())
    ax.set_xlim(*WIN_X)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("x (mm)")
    ax.text(0.0, 1.115, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=color,
            fontweight="bold")
    ax.text(0.0, 1.025, f"{n_hit} of {n_gt} found, {n_fp} spurious", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=5.8, color=C["mid"])
    return n_gt, n_hit, n_fp


def main():
    fields = load_fields()
    audit = load_json("fig2/gaussian_fitting_8fields.json")["records"]
    runtime = load_json("fig2/pipeline_runtime.json")
    net120 = read_csv("fig2/network_120_fields.csv")

    fig = new_figure(103)
    # ---------------- a: observed versus rendered view ----------------------
    f0 = fields[FIELD]
    render = np.load(DATA / f"fig2/fields/{FIELD}/rendered_views.npz")["render"]
    obs = f0["central_sai"][CROP] / 255.0
    ren = render[0][CROP]
    a1 = ax_mm(fig, 11, 11, 36, 28.8)
    a2 = ax_mm(fig, 49, 11, 36, 28.8)
    vmax_o = np.percentile(obs, 99.8)
    vmax_r = np.percentile(ren, 99.8)
    a1.imshow(obs, cmap="gray", vmin=0, vmax=vmax_o, interpolation="nearest")
    a2.imshow(ren, cmap="gray", vmin=0, vmax=vmax_r, interpolation="nearest")
    for a, t in ((a1, "Observed view"), (a2, "Rendered from fitted particles")):
        image_axes(a)
        a.text(0.0, 1.03, t, transform=a.transAxes, ha="left", va="bottom", fontsize=6.5)
    a1.plot([3, 13], [45.0, 45.0], color="white", lw=1.2, solid_capstyle="butt")
    a1.text(8, 43.6, "10 px", color="white", fontsize=5.5, ha="center", va="bottom")
    panel_label(fig, a1, "a", dx=-8)

    # ---------------- b: recovered centres in a slab -------------------------
    b1 = ax_mm(fig, 103, 11, 28.8, 28.8)
    b2 = ax_mm(fig, 141, 11, 28.8, 28.8)
    slab_panel(b1, f0["gt"], f0["stage1"], C["fit"], "Per-exposure fit")
    slab_panel(b2, f0["gt"], f0["network"], C["net"], "Feed-forward network")
    b1.set_ylabel("Depth z (mm)")
    b2.set_yticklabels([])
    panel_label(fig, b1, "b", dx=-9.5)
    # shared legend under panel b
    h = [
        b1.scatter([], [], s=9, facecolors="white", edgecolors=C["truth"], linewidths=0.5, label="Reference"),
        b1.scatter([], [], s=6, color=C["mid"], linewidths=0, label="Matched estimate"),
        b1.scatter([], [], s=9, facecolors="white", edgecolors=C["bad"], linewidths=0.7, label="Missed"),
        b1.scatter([], [], s=9, marker="x", color=C["bad"], linewidths=0.7, label="Spurious"),
    ]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(100 / 180, 1 - 46.5 / 104), ncol=4,
               columnspacing=1.0, handletextpad=0.2, fontsize=5.8)

    # ---------------- c: completeness against time, all methods, sealed fields ----
    import json as _json
    gres = DATA / "gap" / "results"

    def gsum(name):
        f = gres / f"{name}.json"
        return _json.loads(f.read_text())["summary"] if f.exists() else None

    timing = _json.loads((gres / "timing.json").read_text()) if (gres / "timing.json").exists() else {}
    t_fit = np.array([r["seconds"]["stage1"] for r in audit])
    c = ax_mm(fig, 12, 60, 64, 32)
    # name, result file, seconds per exposure, colour, marker, label position (x factor, y offset), alignment
    def secs(key, fallback=None):
        return timing[key]["ms_mean"] / 1000 if key in timing else fallback
    methods = [
        ("View-preserving, 0.5 M", ["viewlift_s0__sealed_std", "viewlift_s1__sealed_std"], secs("viewlift_s0__network"), C["net"], "o", (0.36, 1.03)),
        ("Lifting, 9.6 M", ["lift_s0__sealed_std", "lift_s1__sealed_std"], secs("lift_s0__network"), C["alt"], "o", (0.36, 0.95)),
        ("Pooled, 9.4 M", ["projlift_s0__sealed_std", "projlift_s1__sealed_std"], secs("projlift_s0__network"), C["net"], "D", (0.36, 0.87)),
        ("Pooled, 0.25 M", "p7__sealed_std", secs("p7__network"), C["net_soft"], "o", (0.36, 0.79)),
        ("Pooled, single stage", ["scratch_s0__sealed_std", "seed1__sealed_std", "seed2__sealed_std"], secs("p7__network"), C["net_soft"], "s",
         (0.36, 0.71)),
        ("Gaussian fit", "fit__sealed_std", None, C["fit"], "o", None),
        ("MART fine", "mart484_up2__sealed_std", None, C["classic"], "o", None),
        ("MART 484", "mart484__sealed_std", None, C["classic"], "o", None),
        ("MART 32", "mart32__sealed_std", None, C["classic"], "o", None),
        ("Sweep", "planesweep__sealed_std", None, C["classic"], "o", None),
        ("Backprojection", "backprojection__sealed_std", None, C["classic"], "o", None),
    ]
    pts = {}
    for lab, name, t, col, mk, pos in methods:
        if isinstance(name, list):                 # mean over independent trainings
            runs = [gsum(n) for n in name if gsum(n)]
            sm = dict(f1=float(np.mean([r["f1"] for r in runs])), seconds_per_field=None) if runs else None
        else:
            sm = gsum(name)
        if sm is None:
            continue
        t = t if t is not None else sm["seconds_per_field"]
        if t is None:
            continue
        pts[lab] = (t, sm["f1"])
        c.scatter([t], [sm["f1"]], s=16, color=col if mk == "o" else "white", edgecolors=col, linewidths=0.8,
                  marker=mk, zorder=4)
        if pos is not None:                       # stacked labels with leaders for the three networks
            c.plot([t * 1.12, pos[0] * 0.93], [sm["f1"], pos[1]], color=C["light"], lw=0.5, zorder=2)
            c.text(pos[0], pos[1], f"{lab}  {sm['f1']:.2f}", color=col, fontsize=5.8, va="center", ha="left",
                   fontweight="bold" if mk == "o" else "normal")
    place = {"Gaussian fit": (1.25, -0.1, "right"), "MART fine": (1.3, 0.0, "left"), "MART 484": (1.3, -0.05, "left"),
             "MART 32": (1.2, 0.085, "left"), "Sweep": (0.8, -0.03, "right"), "Backprojection": (1.3, -0.03, "left")}
    for lab, (fx, dy, ha) in place.items():
        if lab in pts:
            t, f = pts[lab]
            c.text(t * fx, f + dy, f"{lab}  {f:.2f}", color=C["fit"] if lab == "Gaussian fit" else C["mid"], fontsize=5.6,
                   va="center", ha=ha, fontweight="bold" if lab == "Gaussian fit" else "normal")
    c.set_xscale("log")
    c.set_xlim(0.025, 120)
    c.set_xticks([0.1, 1, 10, 100])
    c.set_xticklabels(["0.1", "1", "10", "100"])
    c.minorticks_off()
    c.set_ylim(0, 1.1)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.set_xlabel("Time per exposure (s)")
    c.set_ylabel("Detection F1, 120 sealed fields")
    ratio = t_fit.mean() / (runtime["decode_nms_one_frame"]["mean_ms"] / 1000)
    title(c, "One pass against iterative reconstruction")
    panel_label(fig, c, "c", dx=-11)

    # ---------------- d: localization error by axis --------------------------
    d = ax_mm(fig, 91, 60, 30, 32)
    axes_lbl = ["x", "y", "z"]
    for i, axn in enumerate(axes_lbl):
        for j, (key, col) in enumerate((("stage1", C["fit"]), ("network", C["net"]))):
            vals = np.array([r[key][f"sigma_{axn}"] for r in audit]) * 1000
            xpos = i + (-0.17 if j == 0 else 0.17)
            d.scatter(np.full(len(vals), xpos) + np.linspace(-0.07, 0.07, len(vals)), vals, s=8, color=col,
                      linewidths=0, alpha=0.8, zorder=3)
            d.plot([xpos - 0.13, xpos + 0.13], [vals.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
    d.set_xticks(range(3))
    d.set_xticklabels(["x", "y", "z"])
    d.set_ylabel("Error s.d. of matched centres (µm)")
    d.set_xlabel("Axis")
    d.set_ylim(0, 140)
    d.set_yticks([0, 40, 80, 120])
    d.set_xlim(-0.5, 2.5)
    d.grid(axis="y", color=C["faint"], lw=0.5, zorder=0)
    d.set_axisbelow(True)
    d.scatter([], [], s=8, color=C["fit"], label="Fit")
    d.scatter([], [], s=8, color=C["net"], label="Network")
    d.legend(loc="center left", bbox_to_anchor=(0.0, 0.714), handletextpad=0.1, borderaxespad=0.3,
             labelspacing=0.3)
    title(d, "Depth is the hard axis")
    panel_label(fig, d, "d", dx=-10)

    # ---------------- e: recall against distance from the focal plane --------
    g = ax_mm(fig, 135, 60, 40, 32)
    bands = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    depth_recall = {}
    for key, col in (("backprojection", C["classic"]), ("stage1", C["fit"]), ("network", C["net"])):
        per = np.zeros((len(fields), len(bands)))
        for i, f in enumerate(fields.values()):
            m = match(f[key], f["gt"])
            hit = np.zeros(len(f["gt"]), bool)
            hit[m["gt_idx"]] = True
            az = np.abs(f["gt"][:, 2])
            per[i] = [hit[(az >= lo) & (az < hi)].mean() for lo, hi in bands]
        for row in per:
            g.plot(range(5), 100 * row, color=col, lw=0.4, alpha=0.35, zorder=2)
        g.plot(range(5), 100 * per.mean(0), color=col, lw=1.2, marker="o", ms=2.6, zorder=3)
        depth_recall[key] = per.mean(0)
    g.set_xticks(range(5))
    g.set_xticklabels(["0–1", "1–2", "2–3", "3–4", "4–5"])
    g.set_xlim(-0.3, 4.3)
    g.set_ylim(0, 100)
    g.set_yticks([0, 25, 50, 75, 100])
    g.set_xlabel("Distance from focus |z| (mm)")
    g.set_ylabel("Particles recovered (%)")
    title(g, "Largest gain near focus")
    panel_label(fig, g, "e", dx=-10)

    save(fig, "fig2_reconstruction", axes=[a1, a2, b1, b2, c, d, g],
         panel_ids=["a1", "a2", "b1", "b2", "c", "d", "e"],
         row_groups=[["a1", "a2", "b1", "b2"], ["c", "d", "e"]], column_groups=[],
         exemptions=[mixed_row(["a1", "a2", "b1", "b2"]), mixed_row(["c", "d", "e"])])
    print("recall by depth band:", {k: (100 * v).round(1) for k, v in depth_recall.items()})
    print("fit seconds mean (eight fields)", t_fit.mean(), "pipeline ms", runtime["decode_nms_one_frame"]["mean_ms"],
          "ratio", ratio, "| sealed-field points", {k: (round(v[0], 3), round(v[1], 3)) for k, v in pts.items()})


if __name__ == "__main__":
    main()
