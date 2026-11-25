"""Fig. 6 | Recorded light fields delimit what transfers from simulation.

Core conclusion: on recorded images the Gaussian model, fitted per exposure, reads the angular
geometry and stepped axial displacement of a physical target. With conditioned input the frozen
network tracks the lateral displacement of 143 recorded double-frame pairs in agreement with
cross-correlation of views it does not read, but the depth it returns for the target does not
follow the stage.
"""
from __future__ import annotations

import numpy as np

from lfstyle import (C, DATA, ax_mm, image_axes, load_json, mixed_row, new_figure, panel_label,
                     read_csv, save, title)


def main():
    fig = new_figure(108)
    top1, h1 = 10.0, 33.0

    # ------------------------------------------------------------------ a
    tv = np.load(DATA / "fig6/target_view.npz")
    img = tv["image"].astype(float)
    a = ax_mm(fig, 8, top1, h1 * 230 / 184, h1)
    a.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.95), interpolation="nearest")
    xy = tv["assigned_xy"]
    a.scatter(xy[:, 0], xy[:, 1], s=14, facecolors="none", edgecolors="#57C7F2", linewidths=0.5)
    image_axes(a)
    a.plot([8, 48], [176, 176], color="white", lw=1.2, solid_capstyle="butt")
    a.text(28, 172.5, "40 px", color="white", fontsize=5.5, ha="center", va="bottom")
    a.text(0, 1.03, "Perforated target, held-out view", transform=a.transAxes, fontsize=6.8, fontweight="bold",
           ha="left", va="bottom")
    panel_label(fig, a, "a", dx=-5.5)

    # ------------------------------------------------------------------ b
    b = ax_mm(fig, 76, top1, 34, h1)
    ang = {r["method"]: float(r["readable"]) for r in read_csv("fig6/target_angular_prediction.csv")}
    labels = ["Fixed position", "Affine in angle", "Quadratic in angle", "Quadratic +\noptical term"]
    vals = [ang["constant"], ang["linear"], ang["quadratic"], ang["quadratic_mixed"]]
    cols = [C["classic_soft"], C["classic"], C["fit"], C["fit_soft"]]
    yb = np.arange(4)[::-1]
    b.barh(yb, vals, height=0.62, color=cols, zorder=3)
    for y, v in zip(yb, vals):
        b.text(v + 0.015, y, f"{v:.3f}", va="center", ha="left", fontsize=6)
    b.set_yticks(yb)
    b.set_yticklabels(labels)
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 0.95)
    b.set_xticks([0, 0.25, 0.5, 0.75])
    b.set_xlabel("Error of predicted spot centre (px)")
    title(b, "Angular motion is predictable")
    panel_label(fig, b, "b", dx=-22)

    # ------------------------------------------------------------------ c
    c = ax_mm(fig, 132, top1, 43, h1)
    mag = load_json("fig6/target_step_magnitude.json")
    steps = np.array([0.5, 1.0, 1.5])
    spec = [("weighted_centroid", C["classic"], "s", "Weighted centroid"),
            ("original_shared", C["fit_soft"], "D", "Gaussian, per frame"),
            ("gaussian_pair", C["fit"], "o", "Gaussian, two frames")]
    for key, col, mk, lab in spec:
        m = [mag[key]["matched_all"][i]["mae"] for i in range(3)]
        c.plot(steps, m, color=col, marker=mk, ms=3.2, lw=1.1, label=lab, zorder=3)
    c.set_xticks(steps)
    c.set_xlim(0.35, 1.65)
    c.set_ylim(0, 1.5)
    c.set_yticks([0, 0.5, 1.0, 1.5])
    c.set_xlabel("Imposed axial step (label units)")
    c.set_ylabel("Mean absolute step error\n(label units)")
    c.grid(axis="y", color=C["faint"], lw=0.5, zorder=0)
    c.set_axisbelow(True)
    c.legend(loc="upper left", fontsize=5.6, borderaxespad=0.2, labelspacing=0.3)
    title(c, "Fitted Gaussians read axial steps")
    panel_label(fig, c, "c", dx=-12.5)

    # ------------------------------------------------------------------ d: recorded flow exposure
    top2, h2 = 63.0, 33.0
    gap = DATA / "gap" / "results"
    geo = load_json("gapA/camera_geometry.json")["plate_june"]
    um = geo["um_per_sai_px"]
    real = load_json("gap/results/real__p7__heatmap.json")
    rows = [p for p in real["pairs"] if p.get("tracks")]
    ex = np.load(DATA / "gapA/example_pair_30_029_030.npz")
    im = ex["central"][0]
    d = ax_mm(fig, 8, top2, h2 * 232 / 186, h2)
    d.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.8), interpolation="nearest")
    xyz = np.load(gap / "real__p7__heatmap.npz")["30_029_030_xyz0"]
    d.scatter(116 - 10.10 * xyz[:, 0], 93 - 10.10 * xyz[:, 1], s=14, facecolors="none", edgecolors="#57C7F2", linewidths=0.5)
    image_axes(d)
    d.plot([8, 8 + 2000 / um], [178, 178], color="white", lw=1.2, solid_capstyle="butt")
    d.text(8 + 1000 / um, 174.5, "2 mm", color="white", fontsize=5.5, ha="center", va="bottom")
    d.text(0, 1.03, "Seeded flow, tracked centres", transform=d.transAxes, fontsize=6.8, fontweight="bold",
           ha="left", va="bottom")
    panel_label(fig, d, "d", dx=-5.5)

    # ------------------------------------------------------------------ e: displacement per pair
    e = ax_mm(fig, 63, top2, h2, h2)
    ref = np.array([[p["ref_row"], p["ref_col"]] for p in rows])
    est = np.array([[p["track_row"], p["track_col"]] for p in rows])
    lim = 5.2
    e.plot([-lim, lim], [-lim, lim], color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    for k, (col, lab) in enumerate(((C["net"], "Along rows"), (C["gold"], "Along columns"))):
        e.scatter(ref[:, k], est[:, k], s=5, color=col, alpha=0.75, linewidths=0, label=lab, zorder=3)
    e.set_xlim(-lim, lim)
    e.set_ylim(-lim, lim)
    e.set_xticks([-4, -2, 0, 2, 4])
    e.set_yticks([-4, -2, 0, 2, 4])
    e.set_xlabel("Cross-correlation, held-out views (px)")
    e.set_ylabel("Linked reconstructions (px)")
    e.legend(loc="upper left", fontsize=5.6, handletextpad=0.1, borderaxespad=0.1)
    title(e, "Displacement of 143 pairs")
    panel_label(fig, e, "e", dx=-11)

    # ------------------------------------------------------------------ f: control
    f = ax_mm(fig, 109, top2, 20, h2)
    tr = np.array([p["tracks"] for p in rows])
    ct = np.array([p["ctrl_tracks"] for p in rows if "ctrl_tracks" in p])
    rng = np.random.default_rng(1)
    for i, (v, col) in enumerate(((tr, C["net"]), (ct, C["classic"]))):
        f.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=col, alpha=0.55, linewidths=0, zorder=3)
        f.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=C["ink"], lw=1.0, zorder=4)
        f.text(i, v.max() + 5, f"{np.median(v):.0f}", ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")
    f.set_xticks([0, 1])
    f.set_xticklabels(["Pair", "Mis-\nmatched"], fontsize=5.8)
    f.set_xlim(-0.6, 1.6)
    f.set_ylim(0, 135)
    f.set_yticks([0, 50, 100])
    f.set_ylabel("Consistent tracks per pair")
    title(f, "Control")
    panel_label(fig, f, "f", dx=-11)

    # ------------------------------------------------------------------ g: depth of the plate
    g = ax_mm(fig, 144, top2, 31, h2)
    plate = load_json("gap/results/plate__p7__heatmap.json")
    pl = [p for p in plate["planes"] if p.get("holes_found")]
    st = np.array([p["stage_mm"] for p in pl])
    zz = np.array([p["z_median"] for p in pl])
    nf = np.array([p["holes_found"] for p in pl])
    g.plot([-5, 5], [5, -5], color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    many = nf >= 50
    g.scatter(st[many], zz[many], s=9, color=C["net"], linewidths=0, zorder=3, label="50 or more of 100 holes")
    g.scatter(st[~many], zz[~many], s=9, facecolors="none", edgecolors=C["net"], linewidths=0.5, zorder=3, label="Fewer")
    g.set_xlim(-8.6, 5.6)
    g.set_ylim(-5.6, 5.6)
    g.set_xticks([-8, -4, 0, 4])
    g.set_yticks([-4, 0, 4])
    g.set_xlabel("Stage position (mm)")
    g.set_ylabel("Depth returned (mm)")
    g.legend(loc="lower left", fontsize=5.3, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    title(g, "Target depth")
    panel_label(fig, g, "g", dx=-10)

    save(fig, "fig6_recordings", axes=[a, b, c, d, e, f, g], panel_ids=list("abcdefg"),
         row_groups=[["a", "b", "c"], ["d", "e", "f", "g"]], column_groups=[["a", "d"]],
         exemptions=[mixed_row(["a", "b", "c"]), mixed_row(["d", "e", "f", "g"]),
                     {"panels": ["a", "d"], "checks": ["column"],
                      "reason": "image plates keep native aspect ratios (184x230 and 186x232 pixels)"}])
    print("real pairs", real["summary"]["all"])
    print("plate", plate["summary"])
    print("angular", vals)
    for key, *_ in spec:
        print(key, [(round(mag[key]["matched_all"][i]["mean"], 3), round(mag[key]["matched_all"][i]["mae"], 3))
                    for i in range(3)], mag[key]["matched_trends"]["strict_order"])


if __name__ == "__main__":
    main()
