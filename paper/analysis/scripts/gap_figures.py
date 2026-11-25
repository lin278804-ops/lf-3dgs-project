"""Extended Data Figs 5 and 6 and Supplementary Fig. 2: experiments on sealed fields.

ED5    baselines on 120 sealed fields (classical, per-exposure fit, learned lifting, network)
ED6    independence and robustness (training runs, density, diameter, sensor noise)
ED7    recorded pairs: axial displacement against a null reference, depth of tracked centres
Supp2  decoding convention of the offset head (lateral bias, plane-locked axial readings)

Inputs are the unmodified experiment records in source_data/gap/ and source_data/gapA/.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from lfstyle import C, DATA, ax_mm, boot_ci, image_axes, mixed_row, new_figure, panel_label, save, strip, title

GAP = DATA / "gap"
RES = GAP / "results"
DZ = 10.0 / 64


def res(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_field(name, key="f1"):
    r = res(name)
    return np.array([f[key] for f in r["fields"]]) if r else None


def errors(name):
    """Signed position errors (mm) of matched particles for one evaluation."""
    z = np.load(RES / f"{name}.npz")
    n = len([k for k in z.files if k.startswith("xyz_")])
    return np.concatenate([z[f"xyz_{i:04d}"][z[f"pred_idx_{i:04d}"]] - z[f"gt_{i:04d}"][z[f"gt_idx_{i:04d}"]] for i in range(n)])


# ---------------------------------------------------------------------------------------------- ED5
METHODS = [("backprojection__sealed_std", "Back-\nproj.", C["classic"]), ("planesweep__sealed_std", "Plane\nsweep", C["classic"]),
           ("mart32__sealed_std", "MART\n32", C["classic"]), ("mart484__sealed_std", "MART\n484", C["classic"]),
           ("mart484_up2__sealed_std", "MART\nfine", C["classic"]), ("fit__sealed_std", "Fit", C["fit"]),
           ("lift_s0__sealed_std", "Lifting", C["alt"]), ("p7__sealed_std", "Network", C["net"])]


def ed5():
    fig = new_figure(66)
    out = {}
    have = [(n, lab, col) for n, lab, col in METHODS if res(n)]
    # a: lateral detection against depth placement
    a = ax_mm(fig, 13, 10, 44, 38)
    place = {"backprojection__sealed_std": ("Backprojection", 5, 0), "planesweep__sealed_std": ("Sweep", -5, 3),
             "mart32__sealed_std": ("MART 32", -5, -6), "mart484__sealed_std": ("MART 484", 3, 7),
             "mart484_up2__sealed_std": ("MART fine", 7, 0), "fit__sealed_std": ("Fit", -5, 8),
             "lift_s0__sealed_std": ("Lifting", 6, -9), "p7__sealed_std": ("Network", -7, -10)}
    for n, lab, col in have:
        sm = res(n)["summary"]
        x, y = sm["lateral_f1"], sm["lateral_abs_dz_median_mm"]
        a.scatter([x], [y], s=16, color=col, zorder=3, linewidths=0)
        short, dx, dy = place[n]
        a.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.3,
                   ha="left" if dx > 0 else "right", va="center")
        out[n] = dict(f1=sm["f1"], lateral_f1=x, lateral_dz_median_mm=y)
    a.axhline(2 * DZ, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    a.text(0.615, 2 * DZ + 0.012, "Scoring gate", fontsize=5.3, color=C["mid"], ha="left", va="bottom")
    a.set_xlim(0.6, 1.1)
    a.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_ylim(0, 0.65)
    a.set_xlabel("F1 scored on lateral position only")
    a.set_ylabel("Median depth offset (mm)")
    title(a, "Found laterally, misplaced in depth")

    # b: longer training
    b = ax_mm(fig, 72, 10, 44, 38)
    chains = [("Pooled", ["seed1", "v3_long_a", "v3_long"], C["net_soft"], "s"),
              ("Pooled, added capacity", ["projlift_s0", "projlift_long_a", "projlift_long"], "#4F7FB8", "D"),
              ("Lifting", ["lift_s0", "lift_long_a", "lift_long"], C["alt"], "o")]
    for lab, runs, col, mk in chains:
        ys = [res(f"{r}__sealed_std")["summary"]["f1"] for r in runs if res(f"{r}__sealed_std")]
        b.plot([40, 80, 100][:len(ys)], ys, color=col, marker=mk, ms=3.2, lw=1.0, label=lab)
        out[f"long_{lab}"] = [float(v) for v in ys]
    k = res("viewlift_s0__sealed_std")
    if k:
        b.scatter([40], [k["summary"]["f1"]], s=16, color=C["net"], zorder=4, linewidths=0, label="View-preserving")
    b.scatter([220], [res("p7__sealed_std")["summary"]["f1"]], s=16, facecolors="white", edgecolors=C["net_soft"], marker="s",
              linewidths=0.9, zorder=4, label="Pooled, staged schedule")
    b.set_xticks([40, 80, 100, 220])
    b.set_xlim(20, 240)
    b.set_ylim(0.76, 0.95)
    b.set_xlabel("Training epochs")
    b.set_ylabel("Detection F1, sealed fields")
    b.legend(loc="lower right", fontsize=5.3, handlelength=1.4, labelspacing=0.25)
    title(b, "Longer training")

    # c: principal network on three cameras: old calibration, new lookup, converted output
    c = ax_mm(fig, 131, 10, 44, 38)
    cams = [("0.9", "sealed_cam090", "0.90"), ("1", "sealed_camB", "1.00"), ("1.1", "sealed_cam110", "1.10")]
    arms = [("As trained", "", C["bad"]), ("New lookup", "__f{}", C["net_soft"]), ("Output converted", "__conv{}", C["net"])]
    for i, (fl, ds, lab) in enumerate(cams):
        for j, (name, sfx, col) in enumerate(arms):
            v = res(f"p7__{ds}{sfx.format(fl)}")["summary"]["f1"]
            x = i + (j - 1) * 0.27
            c.bar(x, v, width=0.24, color=col, zorder=3)
            c.text(x, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
            out[f"principal_{lab}_{name}"] = float(v)
    c.set_xticks(range(3))
    c.set_xticklabels([q[2] for q in cams])
    c.tick_params(axis="x", length=0)
    c.set_xlim(-0.55, 2.55)
    c.set_ylim(0, 1.3)
    c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.set_xlabel("Microlens focal length of the camera (mm)")
    c.set_ylabel("Detection F1")
    for name, _, col in arms:
        c.bar([-5], [0], color=col, label=name)
    c.legend(loc="upper left", fontsize=5.3, handlelength=0.8, handletextpad=0.8, labelspacing=0.3, borderaxespad=0.0,
             bbox_to_anchor=(0.0, 1.0))
    title(c, "Where to apply a calibration")
    for ax, l in ((a, "a"), (b, "b"), (c, "c")):
        panel_label(fig, ax, l, dx=-10)
    save(fig, "ed_fig5_baselines", axes=[a, b, c], panel_ids=list("abc"), row_groups=[list("abc")], column_groups=[])
    print("ED5", json.dumps(out))
    return out


# ---------------------------------------------------------------------------------------------- ED6
def ed6():
    fig = new_figure(76)
    out = {}
    # a: independent training runs on sealed fields
    a = ax_mm(fig, 13, 10, 34, 38)
    runs = [("p7__sealed_std", "Principal", C["net"]), ("scratch_s0__sealed_std", "Run 1", C["net_soft"]),
            ("seed1__sealed_std", "Run 2", C["net_soft"]), ("seed2__sealed_std", "Run 3", C["net_soft"])]
    runs = [r for r in runs if res(r[0])]
    for i, (n, lab, col) in enumerate(runs):
        v = per_field(n)
        strip(a, i, v, col if i == 0 else C["net"], width=0.24, size=3, seed=i, alpha=0.55 if i == 0 else 0.3)
        a.plot([i - 0.34, i + 0.34], [v.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
        a.text(i, v.max() + 0.006, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=5.4)
        out[n] = dict(f1=float(v.mean()), ci95=boot_ci(v))
    a.set_xticks(range(len(runs)))
    a.set_xticklabels([r[1] for r in runs], fontsize=5.6)
    a.tick_params(axis="x", length=0)
    a.set_xlim(-0.6, len(runs) - 0.4)
    a.set_ylim(0.70, 0.95)
    a.set_ylabel("Detection F1 (120 sealed fields)")
    title(a, "Training runs")

    # b: seeding density
    b = ax_mm(fig, 60, 10, 30, 38)
    dens = [("sealed_std", 1), ("sealed_conc050", 2), ("sealed_conc075", 3)]
    series = [("p7", "", C["bad"], "o", "Frozen, 1,500 candidates"), ("p7", "__topk4000", C["net_soft"], "o", "Frozen, 4,000 candidates"),
              ("p7_mixconc", "", C["gold"], "s", "Tuned on mixed densities")]
    for run, sfx, col, mk, lab in series:
        xs, ys = [], []
        for ds, x in dens:
            r = res(f"{run}__{ds}{sfx}") or (res(f"{run}__{ds}") if ds == "sealed_std" and sfx else None)
            if r:
                xs.append(x), ys.append(r["summary"]["f1"])
        if len(xs) > 1:
            b.plot(xs, ys, color=col, marker=mk, ms=3, lw=0.9, label=lab)
            out[f"density_{run}{sfx}"] = dict(zip(map(str, xs), ys))
    b.set_xticks([1, 2, 3])
    b.set_xticklabels(["1×", "2×", "3×"])
    b.set_xlim(0.7, 3.3)
    b.set_ylim(0.3, 1.0)
    b.set_xlabel("Seeding density")
    b.set_ylabel("Detection F1")
    b.legend(loc="lower left", fontsize=5.1, handlelength=1.2, labelspacing=0.2, bbox_to_anchor=(-0.03, -0.02))
    title(b, "Density")

    # c: particle diameter
    c = ax_mm(fig, 103, 10, 24, 38)
    diam = [("sealed_diam30", 30), ("sealed_std", 50), ("sealed_diam70", 70)]
    for run, col, mk, lab in (("p7", C["bad"], "o", "Frozen"), ("p7_node", C["net"], "s", "Node offsets")):
        pts = [(x, res(f"{run}__{ds}")["summary"]["f1"]) for ds, x in diam if res(f"{run}__{ds}")]
        if len(pts) == 3:
            c.plot(*zip(*pts), color=col, marker=mk, ms=3, lw=0.9, label=lab)
            out[f"diameter_{run}"] = {str(x): y for x, y in pts}
    c.set_xticks([30, 50, 70])
    c.set_xlim(22, 78)
    c.set_ylim(0.3, 1.0)
    c.set_yticklabels([])
    c.legend(loc="lower left", fontsize=5.1, handlelength=1.2, labelspacing=0.2, bbox_to_anchor=(-0.03, -0.02))
    c.set_xlabel("Particle diameter (µm)")
    title(c, "Size")

    # d: sensor noise
    d = ax_mm(fig, 142, 10, 34, 38)
    snrs = [20, 25, 30, 35, 40, 50]
    for run, sfx, col, mk, lab in (("p7", "", C["bad"], "o", "Frozen network"), ("p7", "__cond", C["net"], "o", "Frozen, conditioned"),
                                   ("p7_aug", "", C["gold"], "s", "Tuned with noise"), ("lift_s0", "__cond", C["alt"], "^", "Lifting, conditioned"),
                                   ("fit", "", C["fit"], "D", "Gaussian fit")):
        pts = [(q, res(f"{run}__sealed_std__snr{q}{sfx}")["summary"]["f1"]) for q in snrs if res(f"{run}__sealed_std__snr{q}{sfx}")]
        clean = res(f"{run}__sealed_std{sfx}")
        if pts:
            d.plot(*zip(*pts), color=col, marker=mk, ms=3, lw=0.9, label=lab)
            if clean:
                d.plot([58], [clean["summary"]["f1"]], color=col, marker=mk, ms=3, clip_on=False)
            out[f"noise_{run}{sfx}"] = {str(q): y for q, y in pts} | ({"clean": clean["summary"]["f1"]} if clean else {})
    d.fill_between([28, 40], 0, 1.0, color=C["faint"], lw=0, zorder=0)
    d.text(34, 0.02, "Recorded\npairs", fontsize=5.0, color=C["mid"], ha="center", va="bottom")
    d.set_xticks([20, 30, 40, 50, 58])
    d.set_xticklabels(["20", "30", "40", "50", "None"])
    d.set_xlim(17, 60)
    d.set_ylim(0, 1.0)
    d.set_xlabel("Peak signal-to-noise ratio (dB)")
    d.set_ylabel("Detection F1")
    d.legend(loc="upper center", fontsize=5.2, handlelength=1.2, labelspacing=0.25, ncol=2, columnspacing=0.9,
             bbox_to_anchor=(0.42, -0.25))
    title(d, "Sensor noise")
    for ax, l, dx in ((a, "a", -10), (b, "b", -10), (c, "c", -4), (d, "d", -10)):
        panel_label(fig, ax, l, dx=dx)
    save(fig, "ed_fig6_robustness", axes=[a, b, c, d], panel_ids=list("abcd"), row_groups=[list("abcd")], column_groups=[],
         exemptions=[mixed_row(list("abcd"))])
    print("ED6", json.dumps(out))
    return out


# ---------------------------------------------------------------------------------------------- ED7
REAL = [("real__p7__heatmap", "Network", C["net"]), ("real__lift_s0__heatmap", "Lifting", C["alt"]), ("real__fit", "Fit", C["fit"])]


def real_tracks(name):
    """Per-track arrays of the recorded pairs: axial displacement and starting depth (mm)."""
    z = np.load(RES / f"{name}.npz")
    keys = sorted(k[:-5] for k in z.files if k.endswith("_xyz0"))
    dz = np.concatenate([z[k + "_xyz1"][:, 2] - z[k + "_xyz0"][:, 2] for k in keys])
    z0 = np.concatenate([z[k + "_xyz0"][:, 2] for k in keys])
    return dz, z0


def ed7():
    """Recorded pairs: axial displacement against a null reference, and the depth of tracked centres."""
    fig = new_figure(66)
    out = {}
    # a: mean axial displacement of each pair
    a = ax_mm(fig, 13, 10, 44, 38)
    bins = np.linspace(-0.3, 0.3, 41)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for name, lab, col in REAL:
        rows = [p for p in res(name)["pairs"] if p.get("tracks")]
        m = np.array([p["track_dz_mean_mm"] for p in rows])
        sd = np.array([p["track_dz_sd_mm"] for p in rows])
        n = np.array([p["tracks"] for p in rows])
        lat = np.array([np.hypot(p["track_row"], p["track_col"]) for p in rows])
        h, _ = np.histogram(m, bins)
        a.plot(cen, h, color=col, lw=1.0, drawstyle="steps-mid", label=lab)
        out[name] = dict(pairs=len(rows), mean_um=float(1000 * m.mean()), se_um=float(1000 * m.std(ddof=1) / np.sqrt(len(m))),
                         sd_over_pairs_um=float(1000 * m.std(ddof=1)), median_abs_um=float(1000 * np.median(np.abs(m))),
                         expected_from_track_scatter_um=float(1000 * np.median(sd / np.sqrt(n))),
                         track_sd_mm_median=float(np.median(sd)), lateral_px_median=float(np.median(lat)))
    a.axvline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    a.set_xlim(-0.3, 0.3)
    a.set_ylim(0, 48)
    a.set_xticks([-0.2, 0, 0.2])
    a.set_xlabel("Mean axial displacement of a pair (mm)")
    a.set_ylabel("Recorded pairs")
    a.legend(loc="upper left", fontsize=5.6, handlelength=1.2, labelspacing=0.25)
    title(a, "Axial displacement, reference zero")

    # b: single tracks, recorded against simulated stationary particles
    b = ax_mm(fig, 72, 10, 44, 38)
    bins = np.linspace(-0.9, 0.9, 73)
    cen = 0.5 * (bins[1:] + bins[:-1])
    dz, z0 = real_tracks("real__p7__heatmap")
    zs = np.load(RES / "motion__p7__heatmap.npz")
    sim = zs["meas"][zs["motion"] == list(zs["motions"]).index("still"), 2]
    for v, col, lab in ((sim, C["classic"], "Simulated, stationary"), (dz, C["net"], "Recorded")):
        h, _ = np.histogram(v, bins, density=True)
        b.plot(cen, h, color=col, lw=1.0, drawstyle="steps-mid", label=lab)
    out["tracks"] = dict(recorded_n=int(len(dz)), recorded_sd_mm=float(dz.std()), recorded_mad_sd_mm=float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
                         simulated_sd_mm=float(sim.std()))
    b.set_xlim(-0.9, 0.9)
    b.set_ylim(0, 12.5)
    b.set_xticks([-0.6, 0, 0.6])
    b.set_xlabel("Axial displacement of a track (mm)")
    b.set_ylabel("Density (per mm)")
    b.plot([-0.6, -0.6], [0, 1.2], color=C["mid"], lw=0.6)
    b.plot([0.6, 0.6], [0, 1.2], color=C["mid"], lw=0.6)
    b.text(0.63, 1.5, "Linking\ngate", fontsize=5.4, color=C["mid"], ha="left", va="bottom")
    b.legend(loc="upper right", fontsize=5.6, handlelength=1.2, labelspacing=0.25)
    title(b, "Single tracks")

    # c: depth of the tracked centres
    c = ax_mm(fig, 131, 10, 44, 38)
    bins = np.linspace(-5, 5, 65)
    cen = 0.5 * (bins[1:] + bins[:-1])
    top = 0
    for name, lab, col in REAL:
        _, z0 = real_tracks(name)
        h, _ = np.histogram(z0, bins, density=True)
        top = max(top, h.max())
        c.plot(cen, h, color=col, lw=1.0, drawstyle="steps-mid")
        q = np.percentile(z0, [5, 25, 50, 75, 95])
        out[name].update(tracks=int(len(z0)), depth_percentiles_mm=[float(v) for v in q])
    c.axvline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    c.set_xlim(-5, 5)
    c.set_ylim(0, 1.08 * top)
    c.text(-0.25, 1.04 * top, "Focal\nplane", fontsize=5.4, color=C["mid"], ha="right", va="top")
    c.set_xticks([-4, -2, 0, 2, 4])
    c.set_xlabel("Depth of tracked centres (mm)")
    c.set_ylabel("Density (per mm)")
    title(c, "Depth occupied by tracks")
    for ax, l in ((a, "a"), (b, "b"), (c, "c")):
        panel_label(fig, ax, l, dx=-10)
    save(fig, "ed_fig7_recorded_depth", axes=[a, b, c], panel_ids=list("abc"), row_groups=[list("abc")], column_groups=[])
    (GAP / "ed7_numbers.json").write_text(json.dumps(out, indent=1))
    print("ED7", json.dumps(out))
    return out


# ------------------------------------------------------------------------------- Supplementary Fig. 2
def supp2():
    """Decoding convention: lateral bias and plane-locked axial readings of the offset head."""
    fig = new_figure(62, 120)
    out = {}
    a = ax_mm(fig, 14, 10, 40, 38)
    bins = np.linspace(-90, 90, 61)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for name, col in (("p7__sealed_std", C["bad"]), ("p7__sealed_std__heatmap", C["net"])):
        e = errors(name)[:, 0] * 1000
        h, _ = np.histogram(e, bins, density=True)
        a.plot(cen, h * 1000, color=col, lw=1.0)
        a.plot([e.mean()] * 2, [0, 1.5], color=col, lw=0.9)
        out[name] = dict(bias_x_um=float(e.mean()), sd_x_um=float(e.std()), n=int(len(e)))
    a.axvline(0, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    a.text(-88, 15.5, "Occupancy\npeak", color=C["net"], fontsize=5.8, ha="left", va="center")
    a.text(46, 15.5, "Offset\nhead", color=C["bad"], fontsize=5.8, ha="left", va="center")
    a.set_xlim(-90, 90)
    a.set_ylim(0, 22)
    a.set_xticks([-80, -40, 0, 40, 80])
    a.set_xlabel("Lateral error along x (µm)")
    a.set_ylabel("Density (per mm)")
    title(a, "Position bias")

    b = ax_mm(fig, 72, 10, 40, 38)
    bins = np.linspace(-0.36, 0.46, 83)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for name, col in (("motion__p7", C["bad"]), ("motion__p7__heatmap", C["net"])):
        z = np.load(RES / f"{name}.npz")
        m = z["motion"] == list(z["motions"]).index("z_p005")
        h, _ = np.histogram(z["meas"][m, 2], bins, density=True)
        b.plot(cen, h, color=col, lw=0.9, drawstyle="steps-mid")
    for k in (-2, -1, 0, 1, 2):
        b.plot([k * DZ] * 2, [0, 2.2], color=C["mid"], lw=0.7)
    b.axvline(0.05, color=C["ink"], lw=0.7, ls=(0, (2, 1.5)))
    b.text(0.075, 43, "Imposed\n+0.05 mm", fontsize=5.4, ha="left", va="top")
    b.set_xlim(-0.36, 0.46)
    b.set_ylim(0, 45)
    b.set_xticks([-0.3, 0, 0.3])
    b.set_xlabel("Measured axial displacement (mm)")
    b.set_ylabel("Density (per mm)")
    title(b, "Sub-plane motion")
    panel_label(fig, a, "a", dx=-11)
    panel_label(fig, b, "b", dx=-11)
    save(fig, "supp_fig2_decoding", axes=[a, b], panel_ids=list("ab"), row_groups=[list("ab")], column_groups=[])
    print("SUPP2", json.dumps(out))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["ed5", "ed6", "ed7", "supp2"]
    for w in which:
        if w in globals():
            globals()[w]()
