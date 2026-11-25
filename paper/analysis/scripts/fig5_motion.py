"""Fig. 5 | Linking reconstructed Gaussian centres measures two-frame displacement.

Core conclusion: centres reconstructed independently in two exposures are linked into
three-component displacements; lateral and axial steps down to a third of a depth plane are
recovered on average, the plane-locked axial readings of the first decoder were a decoding
convention, and the remaining axial error is not set by the sampling of the depth planes.

Data: 24 sealed two-exposure fields of the training camera (source_data/gap/results/motion__*).
"""
from __future__ import annotations

import json

import numpy as np

from lfstyle import C, DATA, ax_mm, boot_ci, new_figure, panel_label, save, title

RES = DATA / "gap" / "results"
DZ = 10.0 / 64
PEAK, HEAD, FIT = "motion__p7__heatmap", "motion__p7", "motion__fit"
ROWS = [("x_p020", 0, 0.20, "Lateral, +0.20 mm"), ("z_p005", 2, 0.05, "Axial, +0.05 mm"), ("z_p020", 2, 0.20, "Axial, +0.20 mm"),
        ("z_m020", 2, -0.20, "Axial, −0.20 mm"), ("still", 2, 0.0, "Stationary (axial)")]


def load(name):
    return json.loads((RES / f"{name}.json").read_text()), np.load(RES / f"{name}.npz")


def main():
    (jp, zp), (jh, zh), (jf, _) = load(PEAK), load(HEAD), load(FIT)
    fig = new_figure(92)
    out = {}

    # ------------------------------------------------------------------ a
    a = ax_mm(fig, 30, 10, 44, 68)
    bins = np.linspace(-0.6, 0.6, 97)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for i, (mot, comp, step, lab) in enumerate(ROWS):
        base = (len(ROWS) - 1 - i) * 1.25
        mp = zp["meas"][zp["motion"] == list(zp["motions"]).index(mot), comp]
        mh = zh["meas"][zh["motion"] == list(zh["motions"]).index(mot), comp]
        hp, _ = np.histogram(mp, bins)
        hh, _ = np.histogram(mh, bins)
        a.fill_between(cen, base, base + hp / hp.max(), step="mid", color=C["net"], lw=0, zorder=3)
        a.plot(cen, base + np.minimum(hh / hp.max(), 1.12), color=C["bad"], lw=0.55, drawstyle="steps-mid", zorder=4)
        a.plot([-0.6, 0.6], [base, base], color=C["ink"], lw=0.5, zorder=5)
        if comp == 2:
            for kk in range(-3, 4):
                a.plot([kk * DZ] * 2, [base - 0.07, base], color=C["mid"], lw=0.6, zorder=5, clip_on=False)
        a.plot([step] * 2, [base, base + 1.1], color=C["ink"], lw=0.7, ls=(0, (2, 1.5)), zorder=6)
        a.text(-0.63, base + 0.42, lab, ha="right", va="center", fontsize=6.3)
        a.text(-0.63, base + 0.12, f"n = {len(mp):,}", ha="right", va="center", fontsize=5.5, color=C["mid"])
        out[mot] = dict(n_tracks=int(len(mp)), mean_measured_mm=float(mp.mean()), mean_measured_mm_head=float(mh.mean()))
    a.set_xlim(-0.6, 0.6)
    a.set_ylim(-0.02, len(ROWS) * 1.25 + 0.05)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    a.set_xlabel("Measured displacement (mm)")
    a.text(0.0, 1.035, "Imposed", transform=a.transAxes, ha="left", va="bottom", fontsize=5.6)
    a.plot([0.215, 0.29], [1.055, 1.055], transform=a.transAxes, color=C["ink"], lw=0.7, ls=(0, (2, 1.5)), clip_on=False)
    a.scatter([0.37], [1.055], transform=a.transAxes, marker="s", s=12, color=C["net"], clip_on=False, linewidths=0)
    a.text(0.40, 1.035, "Occupancy peak", transform=a.transAxes, ha="left", va="bottom", fontsize=5.6)
    a.plot([0.77, 0.83], [1.055, 1.055], transform=a.transAxes, color=C["bad"], lw=0.8, clip_on=False)
    a.text(0.85, 1.035, "Offset head", transform=a.transAxes, ha="left", va="bottom", fontsize=5.6)
    panel_label(fig, a, "a", dx=-27, dy=-5.5)

    # ------------------------------------------------------------------ b: share of the step recovered
    top_b, hb = 10, 26
    b = ax_mm(fig, 91, top_b, 33, hb)
    steps = [("Lateral\n0.20", ["x_p020"]), ("0.05", ["z_p005", "z_m005"]), ("0.10", ["z_p010", "z_m010"]),
             ("0.20", ["z_p020", "z_m020"]), ("0.40", ["z_p040", "z_m040"])]
    for k, (lab, mots) in enumerate(steps):
        for off, key, col in ((-0.2, "blind_transfer_all", C["mid"]), (0.2, "transfer", C["net"])):
            v = 100 * np.mean([[f[key] for f in jp["fields"][m]] for m in mots], 0)
            b.scatter(k + off + np.linspace(-0.11, 0.11, len(v)), v, s=4, color=col, alpha=0.6, linewidths=0, zorder=3)
            b.plot([k + off - 0.17, k + off + 0.17], [v.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
            name = ("lateral_" if k == 0 else "axial_") + lab.split()[-1]
            out[f"transfer_{name}_{'tracks' if key == 'transfer' else 'links'}"] = dict(mean=float(v.mean()), ci95=boot_ci(v, n=5000))
    b.axhline(100, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    b.set_xticks(range(5))
    b.set_xticklabels(["Lateral\n0.20", "Axial\n0.05", "Axial\n0.10", "Axial\n0.20", "Axial\n0.40"], fontsize=5.3)
    b.set_xlabel("Imposed step (mm)")
    b.set_xlim(-0.55, 4.55)
    b.set_ylim(60, 108)
    b.set_yticks([60, 80, 100])
    b.set_ylabel("Imposed step recovered (%)")
    b.scatter([], [], s=6, color=C["mid"], label="All links")
    b.scatter([], [], s=6, color=C["net"], label="Tracks with both ends found")
    b.legend(loc="lower left", fontsize=5.3, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    title(b, "Steps are recovered on average")
    panel_label(fig, b, "b", dx=-11)

    # ------------------------------------------------------------------ c: stationary pairs
    c = ax_mm(fig, 141, top_b, 34, hb)
    for off, j, col, lab in ((-0.2, jf, C["fit"], "Per-exposure fit"), (0.2, jp, C["net"], "Network")):
        sd = 1000 * np.array([f["sd_disp_mm"] for f in j["fields"]["still"]])
        for k in range(3):
            c.scatter(k + off + np.linspace(-0.1, 0.1, len(sd)), sd[:, k], s=4, color=col, alpha=0.6, linewidths=0, zorder=3)
            c.plot([k + off - 0.16, k + off + 0.16], [sd[:, k].mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
            c.text(k + off, sd[:, k].max() + 4, f"{sd[:, k].mean():.0f}", ha="center", va="bottom", fontsize=5.6, color=col,
                   fontweight="bold")
        out[f"still_sd_um_{lab.split()[0].lower()}"] = [float(v) for v in sd.mean(0)]
        c.scatter([], [], s=6, color=col, label=lab)
    c.set_xticks(range(3))
    c.set_xticklabels(["x", "y", "z"])
    c.set_xlim(-0.55, 2.55)
    c.set_ylim(0, 135)
    c.set_yticks([0, 40, 80, 120])
    c.set_ylabel("Apparent displacement of\nstationary particles, s.d. (µm)")
    c.legend(loc="upper left", fontsize=5.3, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    title(c, "Stationary-particle noise floor")
    panel_label(fig, c, "c", dx=-12)

    # ------------------------------------------------------------------ d: track yield
    top_d = 52
    d = ax_mm(fig, 91, top_d, 33, hb)
    mots = ["z_p020", "z_m020"]
    for off, j, col in ((-0.2, jf, C["fit"]), (0.2, jp, C["net"])):
        both = 100 * np.mean([[f["track_recall"] for f in j["fields"][m]] for m in mots], 0)
        link = 100 * np.mean([[f["blind_correct"] for f in j["fields"][m]] for m in mots], 0)
        for k, v in enumerate((both, link)):
            d.scatter(k + off + np.linspace(-0.1, 0.1, len(v)), v, s=4, color=col, alpha=0.6, linewidths=0, zorder=3)
            d.plot([k + off - 0.16, k + off + 0.16], [v.mean()] * 2, color=C["ink"], lw=1.0, zorder=4)
            d.text(k + off, v.min() - 4, f"{v.mean():.0f}", ha="center", va="top", fontsize=5.6, color=col, fontweight="bold")
        out[f"yield_{'fit' if col == C['fit'] else 'network'}"] = dict(both_ends=float(both.mean()), links_correct=float(link.mean()))
    d.set_xticks(range(2))
    d.set_xticklabels(["Both ends\ndetected", "Links to the\nsame particle"])
    d.set_xlim(-0.55, 1.55)
    d.set_ylim(40, 105)
    d.set_yticks([50, 75, 100])
    d.set_ylabel("Share (%)")
    title(d, "Endpoints limit yield")
    panel_label(fig, d, "d", dx=-11)

    # ------------------------------------------------------------------ e: axial error floor
    e = ax_mm(fig, 141, top_d, 34, hb)
    items = [("Fit", "fit__sealed_std", FIT, C["fit"]), ("Head", "p7__sealed_std", HEAD, C["bad"]),
             ("Peak", "p7__sealed_std__heatmap", PEAK, C["net"]),
             ("128\nplanes", "p7_node_d128__sealed_std", "motion__p7_node_d128", C["gold"]),
             ("Lifting", "lift_s0__sealed_std", "motion__lift_s0__heatmap", C["alt"])]
    for i, (lab, stat, mot, col) in enumerate(items):
        s = json.loads((RES / f"{stat}.json").read_text())["summary"]["err_sd_um"][2]
        m = 1000 * json.loads((RES / f"{mot}.json").read_text())["summary"]["still"]["sd_disp_mm"][2]
        e.bar(i - 0.2, s, width=0.36, color=col, zorder=3)
        e.bar(i + 0.2, m, width=0.36, color=col, alpha=0.45, zorder=3)
        e.text(i - 0.2, s - 5, f"{s:.0f}", ha="center", va="top", fontsize=5.2, rotation=90, color="white")
        e.text(i + 0.2, m - 5, f"{m:.0f}", ha="center", va="top", fontsize=5.2, rotation=90, color=C["ink"])
        out[f"floor_{lab.split()[0].lower()}"] = dict(position_sd_um=float(s), stationary_pair_sd_um=float(m))
    e.axhline(1000 * DZ, color=C["mid"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    e.text(4.5, 1000 * DZ + 3, "Plane spacing", fontsize=5.3, color=C["mid"], ha="right", va="bottom")
    e.bar([-5], [0], color=C["ink"], label="Position")
    e.bar([-5], [0], color=C["ink"], alpha=0.45, label="Stationary pair")
    e.legend(loc="upper left", fontsize=5.3, handlelength=0.8, labelspacing=0.2, ncol=2, columnspacing=0.8,
             bbox_to_anchor=(-0.03, 1.03))
    e.set_xticks(range(5))
    e.set_xticklabels([i[0] for i in items], fontsize=5.1)
    e.tick_params(axis="x", length=0)
    e.set_xlim(-0.6, 4.6)
    e.set_ylim(0, 215)
    e.set_yticks([0, 50, 100, 150, 200])
    e.set_ylabel("Axial s.d. (µm)")
    title(e, "Depth sampling is not the limit")
    panel_label(fig, e, "e", dx=-12)

    save(fig, "fig5_motion", axes=[a, b, c, d, e], panel_ids=list("abcde"),
         row_groups=[["b", "c"], ["d", "e"]], column_groups=[["b", "d"], ["c", "e"]])
    (DATA / "gap" / "fig5_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
