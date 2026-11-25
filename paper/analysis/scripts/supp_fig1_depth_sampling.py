"""Supplementary Fig. 1 | Depth is resolved to about one plane of the cost volume.

Re-analysis of archived predictions (training camera, 24 crossover fields) and
of the two-exposure links (1.00-mm camera, matched calibration).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from lfstyle import C, DATA, ax_mm, mixed_row, new_figure, panel_label, save, title

BIN = 10.0 / 64  # mm between depth planes


def main():
    frac = {"z": [], "y": [], "x": []}
    gz_f, pz_f = [], []
    for f in sorted(glob.glob(str(DATA / "fig4/camera_cross_predictions/AA/scene_*.npz"))):
        p = np.load(f)
        t = np.load(str(DATA / "fig4/camera_cross_truth" / os.path.basename(f)))
        for k, name in enumerate("zyx"):
            frac[name].append(p["vox"][:, k] % 1)
        gz, pz = t["gt_vox"][p["gt_idx"], 0], p["vox"][p["pred_idx"], 0]
        same = np.floor(gz) == np.floor(pz)
        gz_f.append(gz[same] % 1)
        pz_f.append(pz[same] % 1)
    gz_f, pz_f = np.concatenate(gz_f), np.concatenate(pz_f)

    dz = []
    for d in sorted(glob.glob(str(DATA / "fig5/motion_links/scene_*"))):
        for cond, sg in (("z_plus", 1), ("z_minus", -1)):
            z = np.load(os.path.join(d, f"{cond}.npz"))
            dz.append(sg * z["error_mm_B"][:, 2] + 0.20)
    dz = np.concatenate(dz)

    fig = new_figure(58)
    a = ax_mm(fig, 14, 9, 44, 36)
    bins = np.linspace(0, 1, 21)
    for name, col in (("x", C["classic"]), ("y", C["gold"]), ("z", C["net"])):
        v = np.concatenate(frac[name])
        h, _ = np.histogram(v, bins)
        a.step(bins[:-1], h / h.sum() * 20, where="post", color=col, lw=1.0, label=name)
    a.axhline(1.0, color=C["mid"], lw=0.6, ls=(0, (3, 2)))
    a.text(0.98, 1.08, "Uniform", fontsize=5.6, color=C["mid"], ha="right", va="bottom")
    a.set_xlim(0, 1)
    a.set_ylim(0, 11)
    a.set_xlabel("Predicted position within a voxel")
    a.set_ylabel("Density")
    a.legend(loc="upper right", fontsize=5.8, title="Axis", title_fontsize=5.8)
    title(a, "Predicted sub-voxel coordinates")
    panel_label(fig, a, "a", dx=-10)

    b = ax_mm(fig, 76, 9, 36, 36)
    h2, xe, ye = np.histogram2d(gz_f, pz_f, bins=20, range=[[0, 1], [0, 1]])
    b.imshow(h2.T, origin="lower", extent=[0, 1, 0, 1], cmap="Blues", aspect="equal")
    b.plot([0, 1], [0, 1], color=C["bad"], lw=0.7, ls=(0, (3, 2)))
    r = np.corrcoef(gz_f, pz_f)[0, 1]
    b.text(0.05, 0.93, f"r = {r:.2f}", transform=b.transAxes, ha="left", va="top", fontsize=6.3)
    b.set_xticks([0, 0.5, 1])
    b.set_yticks([0, 0.5, 1])
    b.set_xlabel("Reference depth within plane")
    b.set_ylabel("Predicted depth within plane")
    title(b, "Same-plane matches")
    panel_label(fig, b, "b", dx=-10)

    c = ax_mm(fig, 131, 9, 44, 36)
    k = dz / BIN
    hb = np.linspace(-1.5, 3.5, 101)
    h, _ = np.histogram(k, hb)
    c.fill_between(0.5 * (hb[1:] + hb[:-1]), 0, h / h.max(), step="mid", color=C["net"], lw=0)
    c.axvline(0.20 / BIN, color=C["ink"], lw=0.7, ls=(0, (2, 1.5)))
    c.text(0.20 / BIN + 0.06, 0.97, "Imposed\n1.28", fontsize=5.6, ha="left", va="top", linespacing=1.1)
    c.set_xlim(-1.5, 3.5)
    c.set_ylim(0, 1.05)
    c.set_xticks([-1, 0, 1, 2, 3])
    c.set_yticks([])
    c.spines["left"].set_visible(False)
    c.set_xlabel("Measured axial step (depth planes)")
    title(c, "Axial steps of 0.20 mm")
    panel_label(fig, c, "c", dx=-5)

    save(fig, "supp_fig1_depth_sampling", axes=[a, b, c], panel_ids=list("abc"), row_groups=[list("abc")],
         column_groups=[], exemptions=[mixed_row(list("abc"))])
    resid = np.abs(k - np.round(k))
    print("same-plane matches", len(gz_f), "r", r, "sd frac z", np.concatenate(frac["z"]).std(),
          "within 0.1 of integer", np.mean(resid < 0.1), "n links", len(dz))


if __name__ == "__main__":
    main()
