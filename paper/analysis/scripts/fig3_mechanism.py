"""Fig. 3 | The network reads image evidence at the positions the renderer predicts.

Core conclusion: with images and weights untouched, changing only the address
at which each view is read collapses the reconstruction, and re-pairing every
view with its own address restores it. The dependence is on the image-coordinate
correspondence, not on the order in which views are stored.
"""
from __future__ import annotations

import collections

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from lfstyle import (C, ax_mm, load_json, mixed_row, new_figure, panel_label, read_csv, save,
                     strip, title)

ARMS = [
    ("calibrated", "Calibrated lookup", "each view read where the\nrenderer projects the particle"),
    ("paired_reorder", "Views and addresses reordered together", "storage order reversed,\ncorrespondence kept"),
    ("half_aperture", "Angular offsets halved", "addresses move half as far\nwith depth"),
    ("central_lookup", "Central address in every view", "no depth-dependent offset"),
    ("mismatched", "Angular labels reversed", "each view read at the address\nof its mirror view"),
]
COL = {"calibrated": C["net"], "paired_reorder": C["net"], "half_aperture": C["bad"],
       "central_lookup": C["bad"], "mismatched": C["bad"]}


def draw_views(ax, y, arm):
    """Three views of one off-focus particle; ring = address read by the network."""
    size, gap, x0 = 6.4, 1.6, 0.0
    shifts = np.array([-1.7, 0.0, 1.7])          # particle image shift across views (mm on page)
    names = ["−", "0", "+"]
    order = [0, 1, 2]
    if arm == "paired_reorder":
        order = [2, 1, 0]
    for slot, v in enumerate(order):
        left = x0 + slot * (size + gap)
        cx, cy = left + size / 2, y
        ax.add_patch(Rectangle((left, y - size / 2), size, size, fc="#1E1E1E", ec="none", zorder=1))
        px = cx + shifts[v]
        for r, a in ((1.5, 0.25), (1.05, 0.5), (0.6, 1.0)):
            ax.add_patch(Circle((px, cy), r, fc="white", ec="none", alpha=a, zorder=2))
        if arm in ("calibrated", "paired_reorder"):
            ring = shifts[v]
        elif arm == "half_aperture":
            ring = shifts[v] / 2
        elif arm == "central_lookup":
            ring = 0.0
        else:
            ring = -shifts[v]
        ok = abs(ring - shifts[v]) < 1e-6
        ax.add_patch(Circle((cx + ring, cy), 1.25, fc="none", ec="#57C7F2" if ok else "#FF8A7A", lw=0.9, zorder=3))
        ax.text(cx, y - size / 2 - 0.5, names[v], ha="center", va="top", fontsize=5.2, color=C["mid"])


def main():
    rows = read_csv("fig3/lookup_interventions_120_fields.csv")
    f1 = collections.defaultdict(list)
    cnt = collections.defaultdict(lambda: np.zeros(3))
    for r in rows:
        f1[r["arm"]].append(float(r["f1"]))
        cnt[r["arm"]] += np.array([int(r["tp"]), int(r["fp"]), int(r["fn"])])

    fig = new_figure(118)
    # ------------------------------------------------------------------ a
    row_h = 11.0
    top, h = 9.0, row_h * len(ARMS)
    sch = ax_mm(fig, 6, top, 26, h)
    sch.set_xlim(-1.0, 25.0)
    sch.set_ylim(0, h)
    sch.axis("off")
    lab = ax_mm(fig, 33, top, 46, h)
    lab.set_xlim(0, 1)
    lab.set_ylim(0, h)
    lab.axis("off")
    a = ax_mm(fig, 84, top, 56, h)
    cntax = ax_mm(fig, 147, top, 28, h)
    for i, (arm, name, gloss) in enumerate(ARMS):
        y = h - (i + 0.5) * row_h
        draw_views(sch, y + 0.6, arm)
        lab.text(0.0, y + 2.3, name, fontsize=6.5, fontweight="bold", va="center", ha="left")
        lab.text(0.0, y - 2.1, gloss, fontsize=5.6, color=C["mid"], va="center", ha="left", linespacing=1.35)
        vals = np.array(f1[arm])
        rng = np.random.default_rng(i)
        a.scatter(vals, y + rng.uniform(-2.6, 2.6, len(vals)), s=3.5, color=COL[arm], alpha=0.5, linewidths=0,
                  zorder=3)
        a.plot([vals.mean()] * 2, [y - 3.6, y + 3.6], color=C["ink"], lw=1.1, zorder=4)
        ha, dx = ("right", -0.035) if vals.mean() > 0.5 else ("left", 0.035)
        a.text(vals.mean() + dx, y, f"{vals.mean():.3f}", ha=ha, va="center", fontsize=6.3, fontweight="bold",
               color=COL[arm])
        tp, fp, fn = cnt[arm] / 1000.0
        cntax.barh(y, tp, height=5.0, color=COL[arm], zorder=3)
        cntax.barh(y, fp, left=tp, height=5.0, color=C["light"], zorder=3)
    for axx in (a, cntax):
        axx.set_ylim(0, h)
        axx.set_yticks([])
        axx.spines["left"].set_visible(False)
    a.set_xlim(0, 1)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("Detection F1 (120 fields, same images and weights)")
    cntax.set_xlim(0, 170)
    cntax.set_xticks([0, 50, 100, 150])
    cntax.set_xlabel("Candidates returned (thousands)")
    cntax.barh([-10], [0], color=C["ink"], label="Matched")
    cntax.barh([-10], [0], color=C["light"], label="Unmatched")
    cntax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, columnspacing=0.8, fontsize=5.8,
                 handlelength=1.0)
    sch.text(-1.0, h + 1.2, "Address read in three views", fontsize=6.3, va="bottom", ha="left", color=C["mid"])
    panel_label(fig, sch, "a", dx=-3.0, dy=-3.6)
    panel_label(fig, cntax, "b", dx=-4.0, dy=-3.6)

    # ------------------------------------------------------------------ c
    top2 = 82.0
    c = ax_mm(fig, 14, top2, 34, 27)
    light = read_csv("fig3/observed_light_at_projection.csv")
    reg = np.array([float(r["registered"]) for r in light])
    rev = np.array([float(r["reversed_labels"]) for r in light])
    for x, v, col in ((0, reg, C["net"]), (1, rev, C["bad"])):
        strip(c, x, v, col, width=0.2, size=4, seed=x)
        c.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=C["ink"], lw=1.1, zorder=4)
        c.text(x + 0.36, v.mean(), f"{v.mean():.2f}×", va="center", ha="left", fontsize=6.3, fontweight="bold",
               color=col)
    c.axhline(1.0, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    c.text(1.72, 1.12, "View mean", fontsize=5.5, color=C["mid"], ha="right", va="bottom")
    c.set_xlim(-0.55, 1.75)
    c.set_ylim(0, 10)
    c.set_xticks([0, 1])
    c.set_xticklabels(["Calibrated", "Labels\nreversed"], linespacing=1.3)
    c.set_ylabel("Brightness at projected\nparticle positions (× view mean)")
    title(c, "Light is where the camera model predicts")
    panel_label(fig, c, "c", dx=-12)

    # ------------------------------------------------------------------ d
    d = ax_mm(fig, 76, top2, 42, 27)
    pref = read_csv("fig3/feature_depth_preference.csv")
    grp = collections.defaultdict(list)
    for r in pref:
        grp[(r["readout"], r["arm"])].append(float(r["pairwise"]) * 100)
    xs = {("cosine", "calibrated"): 0, ("cosine", "mismatched"): 1,
          ("negative_variance", "calibrated"): 2.6, ("negative_variance", "mismatched"): 3.6}
    for key, x in xs.items():
        v = np.array(grp[key])
        col = C["net"] if key[1] == "calibrated" else C["bad"]
        strip(d, x, v, col, width=0.2, size=4, seed=int(x * 10))
        d.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=C["ink"], lw=1.1, zorder=4)
        d.text(x, v.max() + 1.0, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6, color=col,
               fontweight="bold")
    d.axhline(50, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    d.text(-0.55, 49.4, "Chance", fontsize=5.5, color=C["mid"], ha="left", va="top")
    d.set_xlim(-0.6, 4.3)
    d.set_ylim(40, 75)
    d.set_yticks([40, 50, 60, 70])
    d.set_xticks([0.5, 3.1])
    d.set_xticklabels(["Feature agreement\n(cosine)", "Feature variance"], linespacing=1.3)
    d.tick_params(axis="x", length=0)
    d.set_ylabel("True depth preferred over\n±1 mm alternative (%)")
    title(d, "Sampled features favour the true depth")
    panel_label(fig, d, "d", dx=-12)

    # ------------------------------------------------------------------ e
    e = ax_mm(fig, 138, top2, 37, 27)
    prof = read_csv("fig1/single_particle_depth_readout.csv")
    off = np.array([float(r["depth_offset_mm"]) for r in prof])
    lightp = np.array([float(r["mean_light_over_view_mean"]) for r in prof])
    agree = np.array([float(r["feature_agreement"]) for r in prof])
    e.plot(off, lightp / lightp.max(), color=C["gold"], lw=1.0, label="Brightness")
    e.plot(off, agree / agree.max(), color=C["net"], lw=1.0, label="Feature agreement")
    e.axvline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    e.set_xlim(-2, 2)
    e.set_ylim(0, 1.08)
    e.set_yticks([0, 0.5, 1.0])
    e.set_xticks([-2, -1, 0, 1, 2])
    e.set_xlabel("Candidate depth − true depth (mm)")
    e.set_ylabel("Readout (normalized to peak)")
    e.scatter([0.03], [0.045], marker="^", s=16, color=C["ink"], zorder=5, clip_on=False)
    e.text(-0.14, 0.05, "Network output", fontsize=5.6, va="center", ha="right")
    e.text(1.95, 0.35, "Feature\nagreement", color=C["net"], fontsize=5.8, ha="right", va="center", linespacing=1.35)
    e.text(1.95, 0.165, "Brightness", color=C["gold"], fontsize=5.8, ha="right", va="center")
    title(e, "Local cues for one particle")
    panel_label(fig, e, "e", dx=-11)

    save(fig, "fig3_mechanism", axes=[a, cntax, c, d, e], panel_ids=["a", "b", "c", "d", "e"],
         row_groups=[["a", "b"], ["c", "d", "e"]], column_groups=[],
         exemptions=[mixed_row(["a", "b"]), mixed_row(["c", "d", "e"])])
    for arm, *_ in ARMS:
        print(arm, round(float(np.mean(f1[arm])), 5), cnt[arm])
    print("light", reg.mean(), rev.mean(), (reg - rev).mean())


if __name__ == "__main__":
    main()
