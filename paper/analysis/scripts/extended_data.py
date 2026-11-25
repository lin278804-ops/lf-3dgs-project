"""Extended Data Figs 1-4: controls that bound the main claims.

ED1  learned versus analytical angular lookup under paired training
ED2  rendering supervision (paired training and four-condition crossover)
ED3  calibrated plane sweep versus network across the 2 x 2 camera cross
ED4  two-frame displacement across five flow families (motion-adapted checkpoint)
"""
from __future__ import annotations

import collections

import numpy as np

from lfstyle import (C, ax_mm, boot_ci, load_json, mixed_row, new_figure, panel_label, read_csv, save,
                     strip, title)


def ed1():
    tr = read_csv("extended/learned_lookup_training.csv")
    fin = read_csv("extended/learned_lookup_final_fields.csv")
    fig = new_figure(64)
    cam_name = {"A": "0.80-mm camera", "B": "1.00-mm camera"}
    thr = {"A": 0.60, "B": 0.50}
    axes = []
    for k, cam in enumerate(("A", "B")):
        ax = ax_mm(fig, 13 + k * 45, 9, 34, 42)
        for arm, col in (("analytic", C["net"]), ("learned", C["alt"])):
            curves = []
            for seed in ("0", "1", "2"):
                rs = sorted([r for r in tr if r["arm"] == arm and r["camera"] == cam and r["seed"] == seed],
                            key=lambda r: int(r["step"]))
                xs = [int(r["step"]) / 1000 for r in rs]
                ys = [float(r["mean_f1"]) for r in rs]
                ax.plot(xs, ys, color=col, lw=0.5, alpha=0.4)
                curves.append(ys)
            ax.plot(xs, np.mean(curves, 0), color=col, lw=1.3,
                    label="Analytical projection" if arm == "analytic" else "Learned offsets")
        ax.axhline(thr[cam], color=C["mid"], lw=0.6, ls=(0, (3, 2)))
        ax.text(0.25, thr[cam] + 0.012, "Qualification threshold", fontsize=5.5, color=C["mid"], ha="left",
                va="bottom")
        ax.set_xlim(0, 8.4)
        ax.set_ylim(0, 0.72)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlabel("Successful updates (thousands)")
        if k == 0:
            ax.set_ylabel("Development F1")
            ax.legend(loc="lower right", fontsize=5.8)
        else:
            ax.set_yticklabels([])
        title(ax, cam_name[cam])
        panel_label(fig, ax, "ab"[k], dx=-10 if k == 0 else -4)
        axes.append(ax)
    diff = {}
    for k, cam in enumerate(("A", "B")):
        ax = ax_mm(fig, 113 + k * 34, 9, 26, 42)
        per_seed = []
        for j, seed in enumerate(("0", "1", "2")):
            an = {r["scene"]: float(r["f1"]) for r in fin if r["arm"] == "analytic" and r["camera"] == cam
                  and r["seed"] == seed}
            le = {r["scene"]: float(r["f1"]) for r in fin if r["arm"] == "learned" and r["camera"] == cam
                  and r["seed"] == seed}
            d = np.array([le[s] - an[s] for s in sorted(an)]) * 100
            per_seed.append(d)
            strip(ax, j, d, C["alt"], width=0.14, size=5, seed=j)
            ax.plot([j - 0.25, j + 0.25], [d.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
        allm = np.mean(per_seed, axis=0)
        lo, hi = boot_ci(allm)
        ax.errorbar([3.2], [allm.mean()], yerr=[[allm.mean() - lo], [hi - allm.mean()]], fmt="D", ms=3.5,
                    color=C["ink"], capsize=1.5, lw=0.8, zorder=5)
        ax.axhline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks([0, 1, 2, 3.2])
        ax.set_xticklabels(["1", "2", "3", "Mean"])
        ax.set_xlim(-0.6, 3.8)
        ax.set_ylim(-4, 8)
        ax.set_xlabel("Paired initialization")
        if k == 0:
            ax.set_ylabel("Learned − analytical F1\n(percentage points)")
        else:
            ax.set_yticklabels([])
        title(ax, cam_name[cam])
        panel_label(fig, ax, "cd"[k], dx=-11 if k == 0 else -4)
        axes.append(ax)
        diff[cam] = (allm.mean(), lo, hi)
    save(fig, "ed_fig1_learned_lookup", axes=axes, panel_ids=list("abcd"), row_groups=[list("abcd")],
         column_groups=[], exemptions=[mixed_row(list("abcd"))])
    print("ED1 learned-analytic pp", diff)


def ed2():
    pc = load_json("extended/render_loss_paired_comparison.json")["comparisons"]
    cross = read_csv("extended/render_loss_cross_fields.csv")
    fig = new_figure(62)
    # a: paired effect of the rendering loss (on - off), normalised to the 'off' value
    a = ax_mm(fig, 50, 9, 40, 40)
    items = [("static_best", "f1", "Detection F1"), ("static_best", "z_mae_mm", "Axial position error"),
             ("motion_best", "dz_mae_mm", "Axial displacement error"),
             ("motion_best", "epe3d_mm", "3D displacement error"), ("motion_best", "track_recall", "Correct-track recall")]
    better_if_higher = {"f1", "track_recall"}
    for i, (blk, key, lab) in enumerate(items):
        m = pc[blk][key]
        rel = 100 * m["paired_on_minus_off"] / m["off"]
        lo, hi = [100 * v / m["off"] for v in m["ci95"]]
        good = (rel > 0) == (key in better_if_higher)
        sig = not (lo <= 0 <= hi)
        col = (C["net"] if good else C["bad"]) if sig else C["mid"]
        y = len(items) - 1 - i
        a.errorbar([rel], [y], xerr=[[rel - lo], [hi - rel]], fmt="o", ms=3.5, color=col, capsize=1.5, lw=0.9)
        a.text(-27.5, y, lab, ha="right", va="center", fontsize=6.2)
    a.axvline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)))
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xlim(-26, 6)
    a.set_ylim(-0.6, len(items) - 0.4)
    a.set_xlabel("Change with rendering loss (% of value without)")
    title(a, "Paired training, one initialization")
    panel_label(fig, a, "a", dx=-44)
    # b, c: four-condition crossover
    arms = [("off_old", "Off", 0, C["bad"]), ("off_matched", "Off", 1, C["net"]), ("on_old", "On", 2.6, C["bad"]),
            ("on_matched", "On", 3.6, C["net"])]
    out = []
    for k, (key, ylab, ttl) in enumerate((("recall", "Reference particles recovered (%)", "Detection"),
                                           ("axial_mae_um", "Axial error, same 2,646 particles (µm)",
                                            "Depth accuracy"))):
        ax = ax_mm(fig, 106 + k * 40, 9, 28, 40)
        for arm, _, x, col in arms:
            v = np.array([float(r[key]) for r in cross if r["arm"] == arm]) * (100 if key == "recall" else 1)
            strip(ax, x, v, col, width=0.2, size=4, seed=int(x * 10))
            ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
        ax.set_xticks([0.5, 3.1])
        ax.set_xticklabels(["Rendering\nloss off", "Rendering\nloss on"])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 4.2)
        ax.set_ylabel(ylab)
        if key == "recall":
            ax.set_ylim(0, 80)
            ax.scatter([], [], s=6, color=C["bad"], label="Previous lookup")
            ax.scatter([], [], s=6, color=C["net"], label="Matched lookup")
            ax.legend(loc="upper left", fontsize=5.5, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25,
                      bbox_to_anchor=(0.0, 0.62))
        else:
            ax.set_ylim(110, 175)
        title(ax, ttl)
        panel_label(fig, ax, "bc"[k], dx=-12)
        out.append(ax)
    save(fig, "ed_fig2_render_loss", axes=[a] + out, panel_ids=list("abc"), row_groups=[list("abc")],
         column_groups=[], exemptions=[mixed_row(list("abc"))])


def ed3():
    ps = load_json("fig4/plane_sweep_vs_network.json")
    fig = new_figure(60)
    a = ax_mm(fig, 14, 9, 62, 38)
    b = ax_mm(fig, 100, 9, 62, 38)
    labels = [("AA", "0.80 / 0.80"), ("AB", "0.80 / 1.00"), ("BA", "1.00 / 0.80"), ("BB", "1.00 / 1.00")]
    for i, (k, lab) in enumerate(labels):
        for off, name, col in ((-0.19, "classical", C["classic"]), (0.19, "p7", C["net"])):
            m = ps[k][name]["f1"]
            a.bar(i + off, m["mean"], width=0.36, color=col, zorder=3)
            a.errorbar([i + off], [m["mean"]], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                       fmt="none", ecolor=C["ink"], capsize=1.2, lw=0.6, zorder=4)
            a.text(i + off, m["ci95"][1] + 0.02, f"{m['mean']:.2f}", ha="center", va="bottom", fontsize=5.8)
            e = ps[k]["equal_fp"]["p7_official"]["classical" if name == "classical" else "p7"]
            b.bar(i + off, 100 * e["mean"], width=0.36, color=col, zorder=3)
            b.text(i + off, 100 * e["mean"] + 2.5, f"{100 * e['mean']:.0f}", ha="center", va="bottom", fontsize=5.8)
    for ax, yl, ttl in ((a, "Detection F1", "Frozen operating points"),
                        (b, "Recall at the network's\nfalse-positive count (%)", "Equal false positives")):
        ax.set_xticks(range(4))
        ax.set_xticklabels([l for _, l in labels])
        ax.set_xlabel("Focal length of images / lookup (mm)")
        ax.set_ylabel(yl)
        title(ax, ttl)
    a.set_ylim(0, 1.0)
    b.set_ylim(0, 100)
    a.bar([-5], [0], color=C["classic"], label="Calibrated plane sweep")
    a.bar([-5], [0], color=C["net"], label="Network")
    a.set_xlim(-0.6, 3.6)
    b.set_xlim(-0.6, 3.6)
    a.legend(loc="upper right", fontsize=5.8, ncol=1, bbox_to_anchor=(1.0, 1.0))
    panel_label(fig, a, "a", dx=-11)
    panel_label(fig, b, "b", dx=-13)
    save(fig, "ed_fig3_plane_sweep", axes=[a, b], panel_ids=list("ab"), row_groups=[list("ab")], column_groups=[])


def ed4():
    rows = read_csv("fig5/flow_families_independent.csv")
    T = load_json("fig5/flow_families_temporal_models.json")
    fams = [("uniform", "Uniform"), ("shear", "Shear"), ("rotation", "Rotation"), ("lamb_oseen", "Lamb–Oseen\nvortex"),
            ("batchelor", "Batchelor\nvortex")]
    fig = new_figure(60)
    a = ax_mm(fig, 14, 9, 50, 36)
    b = ax_mm(fig, 80, 9, 50, 36)
    c = ax_mm(fig, 141, 9, 34, 36)
    for i, (key, lab) in enumerate(fams):
        sel = [r for r in rows if f"_{key}_snr" in r["pair_id"]]
        epe = np.array([float(r["epe3d_mm"]) for r in sel]) * 1000
        rec = np.array([float(r["track_recall"]) for r in sel]) * 100
        strip(a, i, epe, C["net"], width=0.2, size=5, seed=i)
        a.plot([i - 0.3, i + 0.3], [epe.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
        strip(b, i, rec, C["net"], width=0.2, size=5, seed=i)
        b.plot([i - 0.3, i + 0.3], [rec.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
    for ax, yl, ttl in ((a, "3D displacement error (µm)", "Independent reconstructions, linked"),
                        (b, "Correct tracks (% of true tracks)", "Track yield")):
        ax.set_xticks(range(5))
        ax.set_xticklabels([l for _, l in fams], fontsize=5.6)
        ax.set_ylabel(yl)
        ax.grid(axis="y", color=C["faint"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        title(ax, ttl)
    a.set_ylim(0, 180)
    b.set_ylim(0, 100)
    indep = np.mean([float(r["epe3d_mm"]) for r in rows]) * 1000
    smooth = T["diagnostic_comparison"]["control_summary"]["epe3d_mm"] * 1000
    joint = T["diagnostic_comparison"]["candidate_summary"]["epe3d_mm"] * 1000
    vals = [indep, smooth, joint]
    c.bar(range(3), vals, width=0.62, color=[C["net"], C["classic"], C["alt"]], zorder=3)
    for i, v in enumerate(vals):
        c.text(i, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
    c.set_xticks(range(3))
    c.set_xticklabels(["Linked", "Median\nsmoothed", "Learned\ntemporal"], fontsize=5.6)
    c.set_ylim(0, 180)
    c.set_yticklabels([])
    c.grid(axis="y", color=C["faint"], lw=0.5, zorder=0)
    c.set_axisbelow(True)
    title(c, "Post-processing")
    panel_label(fig, a, "a", dx=-11)
    panel_label(fig, b, "b", dx=-11)
    panel_label(fig, c, "c", dx=-5)
    save(fig, "ed_fig4_flow_families", axes=[a, b, c], panel_ids=list("abc"), row_groups=[list("abc")],
         column_groups=[], exemptions=[mixed_row(list("abc"))])
    print("ED4 EPE um", vals)


if __name__ == "__main__":
    ed1()
    ed2()
    ed3()
    ed4()
