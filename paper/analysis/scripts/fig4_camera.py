"""Fig. 4 | The camera enters as a replaceable condition.

Core conclusion: for images from a camera the network never saw, substituting
the new calibration (in the feature lookup, or in the coordinates of the
detected particles) recovers most particles without retraining; a calibrated
plane sweep given the same geometry does not, and neither route closes the gap
to the training camera.
"""
from __future__ import annotations

import collections
import glob

import numpy as np

from lfstyle import (C, DATA, ax_mm, boot_ci, mixed_row, new_figure, panel_label, read_csv, save,
                     title)

F_MAIN, L_OBJ, P_MLA, P_PIX = 100.0, 200.0, 0.099, 0.0045
V_MAIN = 1.0 / (1.0 / F_MAIN - 1.0 / L_OBJ)
BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def angular_shift(z, f):
    """Native-pixel shift of a particle image per step in angular index."""
    d = L_OBJ - z
    return P_PIX * V_MAIN * z / (P_MLA * f * L_OBJ * d * (1.0 / d - 1.0 / F_MAIN))


def main():
    fig = new_figure(121)
    top1, h1 = 10.0, 36.0

    # ------------------------------------------------------------------ a
    a = ax_mm(fig, 14, top1, 44, h1)
    z = np.linspace(-5, 5, 201)
    a.plot(z, angular_shift(z, 0.8), color=C["bad"], lw=1.1)
    a.plot(z, angular_shift(z, 1.0), color=C["net"], lw=1.1)
    s4 = angular_shift(4.0, 1.0)
    za = 4.0 * 0.8 * 100 / (100 + 4.0 * (0.8 - 1.0))  # depth with the same shift under 0.80 mm
    a.plot([4.0, 4.0], [-0.3, s4], color=C["mid"], lw=0.6, ls=(0, (2, 1.5)))
    a.plot([za, za], [-0.3, s4], color=C["mid"], lw=0.6, ls=(0, (2, 1.5)))
    a.plot([za, 4.0], [s4, s4], color=C["mid"], lw=0.6, ls=(0, (2, 1.5)))
    a.scatter([4.0], [s4], s=10, color=C["net"], zorder=4)
    a.scatter([za], [s4], s=10, color=C["bad"], zorder=4)
    a.text(4.12, -0.135, "4.00", ha="left", va="center", fontsize=5.6, color=C["net"])
    a.text(za - 0.12, -0.262, f"{za:.2f}", ha="right", va="center", fontsize=5.6, color=C["bad"])
    a.text(-3.0, 0.215, "0.80 mm", color=C["bad"], fontsize=6, ha="left", va="center", fontweight="bold")
    a.text(-4.7, 0.085, "1.00 mm", color=C["net"], fontsize=6, ha="left", va="center", fontweight="bold")
    a.axhline(0, color=C["light"], lw=0.5, zorder=0)
    a.set_xlim(-5, 5)
    a.set_ylim(-0.3, 0.3)
    a.set_yticks([-0.2, 0, 0.2])
    a.set_xticks([-4, -2, 0, 2, 4])
    a.set_xlabel("Particle depth z (mm)")
    a.set_ylabel("Image shift per view step (px)")
    title(a, "Microlens focal length sets the parallax")
    panel_label(fig, a, "a", dx=-12)

    # ------------------------------------------------------------------ b
    b = ax_mm(fig, 79, top1, h1, h1)
    cross = read_csv("fig4/camera_cross_scores.csv")
    f1 = collections.defaultdict(list)
    for r in cross:
        f1[r["arm"]].append(float(r["f1"]))
    grid = np.array([[np.mean(f1["AA"]), np.mean(f1["AB"])], [np.mean(f1["BA"]), np.mean(f1["BB"])]])
    b.imshow(grid, cmap="Blues", vmin=0, vmax=1.0)
    for i in range(2):
        for j in range(2):
            b.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                   color="white" if grid[i, j] > 0.55 else C["ink"])
    b.set_xticks([0, 1])
    b.set_xticklabels(["0.80", "1.00"])
    b.set_yticks([0, 1])
    b.set_yticklabels(["0.80", "1.00"])
    b.set_xlabel("Focal length in the lookup (mm)")
    b.set_ylabel("Focal length that formed\nthe images (mm)")
    b.tick_params(length=0)
    for s in b.spines.values():
        s.set_visible(False)
    title(b, "Detection F1, same frozen network")
    panel_label(fig, b, "b", dx=-13)

    # ------------------------------------------------------------------ c
    c = ax_mm(fig, 131, top1, 44, h1)
    scan = read_csv("fig4/focal_scan_by_field.csv")
    true_f = {"A": 0.80, "C": 0.90, "B": 1.00}
    cols = {"A": C["bad"], "C": C["gold"], "B": C["net"]}
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in scan:
        by[r["observed"]][r["scene"]][float(r["lookup"])] = float(r["f1"])
    for cam in ("A", "C", "B"):
        xs = sorted(next(iter(by[cam].values())).keys())
        arr = np.array([[by[cam][s][x] for x in xs] for s in sorted(by[cam])])
        for row in arr:
            c.plot(xs, row, color=cols[cam], lw=0.4, alpha=0.35, zorder=2)
        c.plot(xs, arr.mean(0), color=cols[cam], lw=1.2, marker="o", ms=2.3, zorder=3)
        c.scatter([true_f[cam]], [arr.mean(0)[xs.index(true_f[cam])]], s=22, facecolors="none",
                  edgecolors=C["ink"], linewidths=0.7, zorder=5)
        c.text(true_f[cam], arr.max(0)[xs.index(true_f[cam])] + 0.085, f"{true_f[cam]:.2f}", color=cols[cam],
               fontsize=6, ha="center", va="bottom", fontweight="bold")
    c.set_xlim(0.73, 1.07)
    c.set_ylim(0, 1.1)
    c.set_xticks([0.75, 0.85, 0.95, 1.05])
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.set_xlabel("Focal length in the lookup (mm)")
    c.set_ylabel("Detection F1")
    c.text(0.985, 0.03, "Rings: focal length that\nformed the images", transform=c.transAxes, ha="right",
           va="bottom", fontsize=5.5, color=C["mid"], linespacing=1.1)
    title(c, "Three cameras, seven lookups")
    panel_label(fig, c, "c", dx=-11.5)

    # ------------------------------------------------------------------ d
    top2, h2 = 73.0, 38.0
    d = ax_mm(fig, 14, top2, 44, h2)
    rec = collections.defaultdict(lambda: np.zeros((24, 5, 2)))
    files = sorted(glob.glob(str(DATA / "fig4/calibration_location_particles/scene_*.npz")))
    for i, f in enumerate(files):
        p = np.load(f)
        az = np.abs(p["gt_xyz"][:, 2])
        for arm in ("old", "output", "lookup"):
            hit = p[f"{arm}_pred_for_gt"] >= 0
            for k, (lo, hi) in enumerate(BANDS):
                m = (az >= lo) & (az < hi)
                rec[arm][i, k] = [hit[m].sum(), m.sum()]
    depth = read_csv("fig4/camera_cross_depth.csv")
    aa = np.zeros((24, 5, 2))
    scenes = sorted({r["scene"] for r in depth})
    for r in depth:
        if r["arm"] == "AA" and r["stratum"].startswith("abs_z"):
            k = int(r["stratum"].split("_")[2])
            aa[scenes.index(r["scene"]), k] = [int(r["tp"]), int(r["truth"])]
    xb = np.arange(5)
    spec = [("old", C["bad"], "s", "Previous calibration"), ("output", C["alt"], "D", "New, applied to output"),
            ("lookup", C["net"], "o", "New, applied to lookup")]
    d.plot(xb, 100 * (aa[:, :, 0] / aa[:, :, 1]).mean(0), color=C["mid"], lw=0.8, ls=(0, (3, 2)), zorder=2)
    for arm, col, mk, lab in spec:
        y = 100 * (rec[arm][:, :, 0] / rec[arm][:, :, 1]).mean(0)
        d.plot(xb, y, color=col, lw=1.1, marker=mk, ms=3.0, label=lab, zorder=3)
    d.text(2.0, 93.8, "Training camera", fontsize=5.5, color=C["mid"], ha="center", va="center")
    d.set_xticks(xb)
    d.set_xticklabels(["0–1", "1–2", "2–3", "3–4", "4–5"])
    d.set_xlim(-0.3, 4.3)
    d.set_ylim(0, 105)
    d.set_yticks([0, 25, 50, 75, 100])
    d.set_xlabel("Distance from focal plane |z| (mm)")
    d.set_ylabel("Reference particles recovered (%)")
    d.grid(axis="y", color=C["faint"], lw=0.5, zorder=0)
    d.set_axisbelow(True)
    title(d, "Images from the 1.00-mm camera")
    panel_label(fig, d, "d", dx=-12)

    # ------------------------------------------------------------------ e
    e = ax_mm(fig, 79, top2, h2, h2)
    rr, pp = np.meshgrid(np.linspace(0.01, 1, 200), np.linspace(0.01, 1, 200))
    e.contour(rr, pp, 2 * rr * pp / (rr + pp), levels=[0.2, 0.4, 0.6, 0.8], colors=C["light"], linewidths=0.5,
              zorder=1)
    for lvl in (0.2, 0.4, 0.6, 0.8):
        e.text(1.03, lvl / (2 - lvl), f"{lvl:.1f}", fontsize=5.5, color=C["mid"], ha="left", va="center",
               clip_on=False)
    e.text(1.03, 0.80, "F1", fontsize=5.5, color=C["mid"], ha="left", va="center", clip_on=False)
    loc = read_csv("fig4/calibration_location_fields.csv")
    for arm, col, mk, _ in spec:
        e.scatter([float(r[f"{arm}_recall"]) for r in loc], [float(r[f"{arm}_precision"]) for r in loc], s=7,
                  color=col, marker=mk, alpha=0.75, linewidths=0, zorder=3)
    sweep = read_csv("fig4/plane_sweep_fields.csv")
    for arm, mk in (("old", "s"), ("output_xyz", "D"), ("sampling", "o")):
        rws = [r for r in sweep if r["arm"] == arm]
        e.scatter([float(r["recall"]) for r in rws],
                  [int(r["tp"]) / (int(r["tp"]) + int(r["fp"])) for r in rws], s=7, marker=mk,
                  facecolors="none", edgecolors=C["classic"], linewidths=0.5, alpha=0.9, zorder=3)
    e.text(0.72, 0.86, "Network", fontsize=6, color=C["ink"], ha="right", va="center", fontweight="bold")
    e.text(0.33, 0.065, "Plane sweep (open)", fontsize=5.6, color=C["classic"], ha="left", va="center")
    e.set_xlim(0, 1)
    e.set_ylim(0, 1)
    e.set_xticks([0, 0.5, 1])
    e.set_yticks([0, 0.5, 1])
    e.set_xlabel("Recall")
    e.set_ylabel("Precision")
    title(e, "Where the calibration is applied")
    panel_label(fig, e, "e", dx=-11)

    # ------------------------------------------------------------------ f
    f = ax_mm(fig, 135, top2, 40, h2)
    c3 = {k: np.array([float(r[f"common3_{k}_mae_mm"]) for r in loc]) * 1000 for k in ("old", "output", "lookup")}
    c2 = {k: np.array([float(r[f"common2_{k}_mae_mm"]) for r in loc]) * 1000 for k in ("output", "lookup")}
    pos = {("c3", "old"): 0.0, ("c3", "output"): 1.0, ("c3", "lookup"): 2.0, ("c2", "output"): 3.6,
           ("c2", "lookup"): 4.6}
    for (grp, arm), x in pos.items():
        v = (c3 if grp == "c3" else c2)[arm]
        col = {"old": C["bad"], "output": C["alt"], "lookup": C["net"]}[arm]
        mk = {"old": "s", "output": "D", "lookup": "o"}[arm]
        f.scatter(x + np.linspace(-0.22, 0.22, len(v)), v, s=6, color=col, marker=mk, alpha=0.6, linewidths=0,
                  zorder=3)
        f.plot([x - 0.33, x + 0.33], [v.mean()] * 2, color=C["ink"], lw=1.1, zorder=4)
        f.text(x, v.max() + 3.4, f"{v.mean():.0f}", fontsize=6, ha="center", va="bottom", color=col,
               fontweight="bold")
    f.set_xticks([1.0, 4.1])
    f.set_xticklabels(["Found by all\nthree (5,039)", "Found by both\ncorrected (17,183)"])
    f.tick_params(axis="x", length=0)
    f.set_xlim(-0.7, 5.3)
    f.set_ylim(95, 172)
    f.set_yticks([100, 120, 140, 160])
    f.set_ylabel("Axial error of matched particles (µm)")
    title(f, "Depth accuracy depends on the cohort")
    panel_label(fig, f, "f", dx=-11.5)

    # shared legend for d-f
    handles = [d.plot([], [], color=col, marker=mk, ms=3.0, lw=1.1, label=lab)[0] for _, col, mk, lab in spec]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 61.5 / 121), ncol=3,
               columnspacing=2.0, fontsize=6.2)

    save(fig, "ed_fig8_camera", axes=[a, b, c, d, e, f], panel_ids=list("abcdef"),
         row_groups=[["a", "b", "c"], ["d", "e", "f"]], column_groups=[["a", "d"], ["b", "e"]],
         exemptions=[mixed_row(["a", "b", "c"]), mixed_row(["d", "e", "f"]),
                     {"panels": ["b", "e"], "checks": ["column"],
                      "reason": "square matrix and square precision-recall axes share the left edge only"}])
    print("grid", grid)
    print("apparent depth", za)
    for arm in ("old", "output", "lookup"):
        print(arm, (100 * (rec[arm][:, :, 0] / rec[arm][:, :, 1]).mean(0)).round(2))
    print("AA", (100 * (aa[:, :, 0] / aa[:, :, 1]).mean(0)).round(2))
    print({k: v.mean().round(2) for k, v in c3.items()}, {k: v.mean().round(2) for k, v in c2.items()})
    print("c3 lookup-output", boot_ci(c3["output"] - c3["lookup"]), "c2", boot_ci(c2["lookup"] - c2["output"]))


if __name__ == "__main__":
    main()
