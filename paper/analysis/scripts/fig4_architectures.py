"""Fig. 4 | How the aligned views are fused decides accuracy and what transfers to another camera.

Core conclusion: with the same encoder, decoder, losses, fields and schedule, a network that keeps
the projection-aligned views apart is the most accurate with half a million parameters; pooling
them transfers best from a single training camera; trained on four cameras and given each
calibration, the view-preserving network serves cameras absent from training.

Data: source_data/gap/results (sealed fields; single-stage training of 40 epochs for every network).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from lfstyle import C, DATA, ax_mm, new_figure, panel_label, save, title

RES = DATA / "gap" / "results"
MID = "#4F7FB8"
ARCH = {  # label, runs trained on the training camera, colour, marker, parameters (millions)
    "pool": ("Pooled", ["scratch_s0", "seed1", "seed2"], C["net_soft"], "s", 0.251),
    "poolcap": ("Pooled, added capacity", ["projlift_s0", "projlift_s1"], MID, "D", 9.389),
    "lift": ("Lifting, no parallax", ["lift_s0", "lift_s1"], C["alt"], "o", 9.650),
    "keep": ("View-preserving", ["viewlift_s0", "viewlift_s1"], C["net"], "o", 0.507),
}


def f1(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text())["summary"]["f1"] if p.exists() else None


def vals(runs, suffix):
    return [v for v in (f1(f"{r}__{suffix}") for r in runs) if v is not None]


def box(ax, x, y, w, h, text, fc, ec=None, fs=5.6, color=None, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=fc, ec=ec or fc, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"],
            fontweight="bold" if bold else "normal", linespacing=1.5 if "$" in text else 1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=C["mid"],
                                 shrinkA=0, shrinkB=0))


def main():
    fig = new_figure(104)
    out = {}

    # ------------------------------------------------------------------ a: three ways to form the volume
    a = ax_mm(fig, 4, 9, 66, 36)
    a.set_xlim(0, 66)
    a.set_ylim(0, 36)
    a.axis("off")
    rows = [(27.5, "Pooled", "Read each view at\nits projected position", "Mean and variance\nacross views", C["net_soft"]),
            (15.0, "View-preserving", "Read each view at\nits projected position", "Views kept apart; one\nfusion for all planes", C["net"]),
            (2.5, "Lifting", "Register views at\nthe focal plane", "Learned convolutions\nmap views to depth", C["alt"])]
    for y, name, t1, t2, col in rows:
        a.text(0.5, y + 7.6, name, fontsize=6.2, fontweight="bold", color=col, ha="left", va="bottom")
        box(a, 0.5, y, 10, 7, "32\nviews", C["faint"])
        arrow(a, 10.8, y + 3.5, 13.6, y + 3.5)
        box(a, 14, y, 20, 7, t1, C["faint"])
        arrow(a, 34.3, y + 3.5, 37.1, y + 3.5)
        box(a, 37.5, y, 19.5, 7, t2, "white", ec=col)
        arrow(a, 57.3, y + 3.5, 60.1, y + 3.5)
        box(a, 60.5, y, 5, 7, "3D", C["faint"])
    panel_label(fig, a, "a", dx=0)

    # ------------------------------------------------------------------ b: accuracy against size
    b = ax_mm(fig, 85, 10, 36, 34)
    lab_pos = {"pool": (1.35, 0.0, "left"), "poolcap": (0.78, -0.014, "right"), "lift": (0.78, 0.012, "right"), "keep": (1.35, 0.004, "left")}
    for key, (lab, runs, col, mk, par) in ARCH.items():
        v = vals(runs, "sealed_std")
        b.scatter([par] * len(v), v, s=6, color=col, alpha=0.5, linewidths=0, zorder=2)
        b.scatter([par], [np.mean(v)], s=22, color=col, marker=mk, linewidths=0, zorder=3)
        fx, dy, ha = lab_pos[key]
        b.text(par * fx, np.mean(v) + dy, f"{np.mean(v):.3f}", fontsize=5.8, color=col, ha=ha, va="center", fontweight="bold")
        out[f"sealed_{key}"] = dict(mean=float(np.mean(v)), runs=[float(x) for x in v], params_millions=par)
    w = f1("viewlift_w_s0__sealed_std")
    if w:
        b.scatter([1.099], [w], s=14, facecolors="white", edgecolors=C["net"], linewidths=0.8, zorder=3)
        out["sealed_keep_wide"] = float(w)
    b.set_xscale("log")
    b.set_xlim(0.13, 20)
    b.set_xticks([0.25, 1, 10])
    b.set_xticklabels(["0.25", "1", "10"])
    b.minorticks_off()
    b.set_ylim(0.76, 0.95)
    b.set_yticks([0.80, 0.85, 0.90, 0.95])
    b.set_xlabel("Parameters (millions)")
    b.set_ylabel("Detection F1, sealed fields")
    title(b, "Accuracy against size")
    panel_label(fig, b, "b", dx=-11)

    # ------------------------------------------------------------------ c: one training camera, other cameras
    c = ax_mm(fig, 137, 10, 38, 34)
    cams = [("0.9", "sealed_cam090", 0.90), ("1", "sealed_camB", 1.00), ("1.1", "sealed_cam110", 1.10)]
    for key in ("pool", "poolcap", "lift", "keep"):
        lab, runs, col, mk, _ = ARCH[key]
        runs = [r for r in runs if r != "scratch_s0"]
        xs, ys = [0.80], [np.mean(vals(runs, "sealed_std"))]
        for fl, ds, x in cams:
            v = vals(runs, f"{ds}__conv{fl}")
            if v:
                xs.append(x), ys.append(np.mean(v))
        c.plot(xs, ys, color=col, marker=mk, ms=3, lw=1.0, zorder=3 if key == "keep" else 2)
        out[f"transfer_{key}"] = dict(zip(map(str, xs), map(float, ys)))
    c.set_xticks([0.8, 0.9, 1.0, 1.1])
    c.set_xticklabels(["0.80\n(trained)", "0.90", "1.00", "1.10"])
    c.set_xlim(0.76, 1.14)
    c.set_ylim(0.25, 1.0)
    c.set_yticks([0.4, 0.6, 0.8, 1.0])
    c.set_xlabel("Microlens focal length (mm)")
    c.set_ylabel("Detection F1, output converted")
    title(c, "Trained on one camera")
    panel_label(fig, c, "c", dx=-11)

    # ------------------------------------------------------------------ d: trained on four cameras
    d = ax_mm(fig, 14, 62, 68, 32)
    # microlens focal length, dataset and result suffix, camera used in training
    sweep = [(0.70, "sealed_cam070__f0.7", False), (0.75, "sealed_cam075__f0.75", False), (0.80, "sealed_std", True),
             (0.85, "sealed_cam085__f0.85", False), (0.90, "sealed_cam090__f0.9", True), (0.95, "sealed_cam095__f0.95", False),
             (1.00, "sealed_camB__f1", True), (1.05, "sealed_cam105__f1.05", False), (1.10, "sealed_cam110__f1.1", True),
             (1.15, "sealed_cam115__f1.15", False), (1.20, "sealed_cam120__f1.2", False)]
    # cameras with another main-lens focal length F or object distance L: (name, F, L, f), plotted at the microlens
    # focal length that gives a training-type camera the same parallax per unit depth, f / M^2 with M = F / (L - F)
    other = [("sealed_F095", 95, 200, 0.8), ("sealed_F105", 105, 200, 0.8), ("sealed_L190", 100, 190, 0.8), ("sealed_L210", 100, 210, 0.8),
             ("sealed_F095_f095", 95, 200, 0.95), ("sealed_F105_f095", 105, 200, 0.95)]

    TOLD = {'sealed_F095': 0.97729, 'sealed_F105': 0.65488, 'sealed_L190': 0.648, 'sealed_L210': 0.968, 'sealed_F095_f095': 1.16053, 'sealed_F105_f095': 0.77767}   # focal lengths of equal parallax as supplied to the lifting baseline

    def other_tag(name, F, L, f, told=None):
        return (f"{name}" + (f"__f{(told if told else f):g}" if (told or f != 0.8) else "") + (f"__F{F:g}" if F != 100 else "")
                + (f"__L{L:g}" if L != 200 else "") + ("__feq" if told else ""))

    multi = [("View-preserving", ["mc_viewlift", "mc_viewlift_s1"], C["net"], "o", "-", False),
             ("Lifting, told the focal length", ["mc_liftcal", "mc_liftcal_s1"], C["alt"], "o", (0, (3, 1.5)), True),
             ("Lifting, no calibration", ["mc_lift"], C["alt_soft"], "o", "-", False),
             ("Pooled", ["mc_v3"], C["net_soft"], "s", "-", False), ("Pooled, added capacity", ["mc_projlift"], MID, "D", "-", False)]
    d.axvspan(0.80, 1.10, color=C["faint"], lw=0, zorder=0)
    d.text(0.95, 0.985, "Range of training cameras", ha="center", va="top", fontsize=5.3, color=C["mid"])
    for lab, runs, col, mk, ls, told in multi:
        xs = np.array([q[0] for q in sweep])
        ys = np.array([np.mean(vals(runs, q[1])) if vals(runs, q[1]) else np.nan for q in sweep])
        seen = np.array([q[2] for q in sweep])
        oth = []
        for name, F, L, f in other:
            feq = f / (F / (L - F)) ** 2
            v = vals(runs, other_tag(name, F, L, f, told=TOLD[name] if told else None))
            oth.append((feq, np.mean(v) if v else np.nan))
        oth = np.array(oth)
        out[f"multi_{lab}"] = dict(focal_mm=[float(v) for v in xs], values=[float(v) for v in ys], trained=[bool(v) for v in seen],
                                   mean=float(np.nanmean(ys[2:9])), unseen_mean=float(np.nanmean(ys[2:9][~seen[2:9]])),
                                   other_equivalent_focal_mm=[float(v) for v in oth[:, 0]], other_values=[float(v) for v in oth[:, 1]])
        if lab == "Pooled, added capacity":      # reported in Supplementary Tables 3 and 4; omitted from the panel for legibility
            continue
        top = 3 if lab == "View-preserving" else 2
        d.plot(xs, ys, color=col, lw=1.0, ls=ls, zorder=top, label=lab)
        d.scatter(xs[seen], ys[seen], s=11, color=col, marker=mk, linewidths=0, zorder=top + 2)
        d.scatter(xs[~seen], ys[~seen], s=11, facecolors="white", edgecolors=col, marker=mk, linewidths=0.8, zorder=top + 2)
        d.scatter(oth[:, 0], oth[:, 1], s=13, facecolors="white", edgecolors=col, marker="^", linewidths=0.8, zorder=top + 2)
    d.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    d.set_xticklabels(["0.70", "0.80", "0.90", "1.00", "1.10", "1.20"])
    d.set_xlim(0.615, 1.235)
    d.set_ylim(0.1, 1.0)
    d.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    d.set_xlabel("Microlens focal length (mm)")
    d.set_ylabel("Detection F1, own calibration")
    d.legend(loc="lower center", fontsize=5.3, ncol=1, handlelength=1.6, labelspacing=0.25, bbox_to_anchor=(0.56, -0.01))
    title(d, "Trained on four cameras")
    panel_label(fig, d, "d", dx=-10)

    # ------------------------------------------------------------------ e: density, noise, few fields
    e = ax_mm(fig, 100, 62, 75, 32)
    tests = [("Twice the\ndensity", "sealed_conc050", None), ("Three times\nthe density", "sealed_conc075", None),
             ("Noise, 30 dB\n(conditioned)", "sealed_std__snr30__cond", None),
             ("120 training\nfields", "sealed_std", {"pool": ["v3_n120"], "poolcap": ["projlift_n120"], "lift": ["lift_n120"], "keep": ["viewlift_n120"]})]
    order = ["pool", "poolcap", "lift", "keep"]
    for i, (lab, suffix, special) in enumerate(tests):
        for j, key in enumerate(order):
            name, runs, col, mk, _ = ARCH[key]
            runs = special[key] if special else [r for r in runs if r != "scratch_s0"]
            v = vals(runs, suffix)
            x = i + (j - 1.5) * 0.2
            if not v:          # run not evaluated yet: say so in the panel instead of leaving a silent gap
                e.text(x, 0.02, "in training", ha="center", va="bottom", fontsize=5.1, rotation=90, color=C["mid"])
                continue
            e.bar(x, np.mean(v), width=0.18, color=col, zorder=3)
            e.text(x, np.mean(v) + 0.015, f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=5.1, rotation=90)
            out[f"{suffix}_{'120' if special else 'full'}_{key}"] = float(np.mean(v))
    e.set_xticks(range(len(tests)))
    e.set_xticklabels([t[0] for t in tests], fontsize=5.6)
    e.tick_params(axis="x", length=0)
    e.set_xlim(-0.55, len(tests) - 0.45)
    e.set_ylim(0, 1.18)
    e.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    e.set_ylabel("Detection F1")
    for key in order:
        e.bar([-5], [0], color=ARCH[key][2], label=ARCH[key][0])
    e.legend(loc="upper left", fontsize=5.3, ncol=4, handlelength=0.8, handletextpad=0.4, columnspacing=0.8,
             bbox_to_anchor=(-0.01, 1.02), borderaxespad=0.0)
    title(e, "Density, noise and few training fields", pad=3.0)
    panel_label(fig, e, "e", dx=-10)

    save(fig, "fig4_architectures", axes=[b, c, d, e], panel_ids=list("bcde"),
         row_groups=[["b", "c"], ["d", "e"]], column_groups=[],
         exemptions=[{"panels": ["b", "c"], "checks": ["panel-width", "horizontal-gutter"], "reason": "row shared with a schematic"},
                     {"panels": ["d", "e"], "checks": ["panel-width", "horizontal-gutter"], "reason": "line plot and grouped bars"}])
    (DATA / "gap" / "fig4_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out)[:1500])


if __name__ == "__main__":
    main()
