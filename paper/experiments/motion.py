"""Displacement measurement on sealed two-exposure fields with known rigid shifts.

  python motion.py --run p7 [--decode heatmap]
  python motion.py --run runs/p7_node
  python motion.py --fit                      # per-exposure fits from fit.py --data sealed_motion

Each exposure is reconstructed independently. Two readouts:
  identity-linked  detections matched to ground truth in each exposure (gate 2 voxels); a track is a
                   particle detected in both; displacement = difference of the two reconstructed centres.
                   Isolates the measurement from the linking.
  blind            detections of the two exposures linked one-to-one within 0.5 mm of the first
                   position (no ground truth); a link is correct if both ends match the same particle.
Writes results/motion__<run>[__decode].json and .npz (per-track arrays).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from evaluate import detect_heatmap, run_dir
from gapcommon import GAP, boot_ci, detect, get_cam, load_model, save_json, score

DZ = 10.0 / 64          # scoring plane spacing (mm)
LINK_MM = 0.5


def load(path):
    with np.load(path) as z:
        return torch.from_numpy(z["sai_stack"].astype(np.float32) / 255.0), z["particle_centers"].astype(np.float32), \
            (z["ids"] if "ids" in z.files else np.arange(len(z["particle_centers"]))), \
            (z["displacement_mm"] if "displacement_mm" in z.files else np.zeros(3, np.float32))


def blind_links(p0, p1):
    if len(p0) == 0 or len(p1) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.sqrt(((p0[:, None] - p1[None]) ** 2).sum(2))
    c = d.copy()
    c[d > LINK_MM] = 1e6
    ri, ci = linear_sum_assignment(c)
    k = d[ri, ci] <= LINK_MM
    return ri[k], ci[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--decode", choices=["head", "heatmap"], default="head")
    ap.add_argument("--data", default="sealed_motion")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cam_score = get_cam()
    folders = sorted(glob.glob(str(GAP / "data" / a.data / "field_*")))[: a.limit]
    if a.fit:
        name = "fit"

        def recon(path):
            f = GAP / "fits" / a.data / f"{Path(path).parent.name}__{Path(path).stem}.npz"
            return np.load(f)["fit_xyz"] if f.exists() else None
        folders = [f for f in folders if recon(f + "/frame0.npz") is not None]
    else:
        dev = "cuda:0"
        model, cfg = load_model(run_dir(a.run), dev)
        cam = get_cam(depth=int(cfg.get("grid_depth", 64)))
        name = (Path(a.run).name if a.run not in ("p7", "scratch_s0") else a.run) + \
               (f"__{a.decode}" if a.decode != "head" else "")

        def recon(path):
            sai = load(path)[0]
            if a.decode == "heatmap":
                return detect_heatmap(model, cam, sai, dev, int(cfg["topk"]))[0]
            return detect(model, cam, sai, dev, cam_score=cam_score)[0]

    tracks = {k: [] for k in ("motion", "field", "true", "meas", "z0", "pred_z0", "pred_z1")}
    per = {}
    motions = None
    for fi, folder in enumerate(folders):
        _, gt0, _, _ = load(folder + "/frame0.npz")
        p0 = recon(folder + "/frame0.npz")
        m0 = score(p0, gt0, cam_score)
        id_of_pred0 = dict(zip(m0["pred_idx"], m0["gt_idx"]))
        pred0_of_id = dict(zip(m0["gt_idx"], m0["pred_idx"]))
        motions = motions or sorted(Path(p).stem for p in glob.glob(folder + "/*.npz") if Path(p).stem != "frame0")
        for mi, mot in enumerate(motions):
            _, gt1, ids1, d = load(f"{folder}/{mot}.npz")
            p1 = recon(f"{folder}/{mot}.npz")
            if p1 is None:
                continue
            m1 = score(p1, gt1, cam_score)
            id_of_pred1 = {pi: int(ids1[gi]) for pi, gi in zip(m1["pred_idx"], m1["gt_idx"])}
            common = [(pred0_of_id[i], pi) for pi, i in id_of_pred1.items() if i in pred0_of_id]
            a0 = np.array([c[0] for c in common], int)
            a1 = np.array([c[1] for c in common], int)
            meas = p1[a1] - p0[a0]
            l0, l1 = blind_links(p0, p1)
            correct = np.array([id_of_pred0.get(i, -1) == id_of_pred1.get(j, -2) for i, j in zip(l0, l1)], bool)
            bl = p1[l1] - p0[l0]
            comp = int(np.argmax(np.abs(d))) if np.abs(d).max() > 0 else 2
            sign = np.sign(d[comp]) if d[comp] != 0 else 1.0
            mag = float(np.abs(d[comp]))
            rec = dict(field=Path(folder).name, n_gt_pairs=int(len(ids1)), n_tracks=int(len(common)),
                       track_recall=len(common) / max(len(ids1), 1),
                       mean_disp_mm=[float(v) for v in meas.mean(0)], median_disp_mm=[float(v) for v in np.median(meas, 0)],
                       sd_disp_mm=[float(v) for v in meas.std(0)],
                       transfer=float(sign * meas[:, comp].mean() / mag) if mag > 0 else None,
                       sign_correct=float((np.sign(meas[:, comp]) == sign).mean()) if mag > 0 else None,
                       on_plane=float((np.abs(meas[:, 2] / DZ - np.round(meas[:, 2] / DZ)) < 0.05).mean()),
                       zero_axial=float((np.abs(meas[:, 2]) < 0.05 * DZ).mean()),
                       n_blind=int(len(l0)), blind_correct=float(correct.mean()) if len(l0) else None,
                       blind_transfer_all=float(sign * bl[:, comp].mean() / mag) if mag > 0 and len(l0) else None,
                       blind_transfer_correct=float(sign * bl[correct, comp].mean() / mag) if mag > 0 and correct.any() else None)
            per.setdefault(mot, []).append(rec)
            tracks["motion"].append(np.full(len(common), mi, np.int16))
            tracks["field"].append(np.full(len(common), fi, np.int16))
            tracks["true"].append(np.tile(d, (len(common), 1)).astype(np.float32))
            tracks["meas"].append(meas.astype(np.float32))
            tracks["z0"].append(gt0[[id_of_pred0[i] for i in a0], 2])
            tracks["pred_z0"].append(p0[a0, 2])
            tracks["pred_z1"].append(p1[a1, 2])
    summary = {}
    for mot, rows in per.items():
        s = {"n_fields": len(rows)}
        for k in ("track_recall", "transfer", "sign_correct", "on_plane", "zero_axial", "blind_correct",
                  "blind_transfer_all", "blind_transfer_correct"):
            v = np.array([r[k] for r in rows if r[k] is not None], float)
            if len(v):
                s[k] = float(v.mean())
                s[k + "_ci95"] = boot_ci(v) if len(v) > 1 else [float(v[0])] * 2
        s["mean_disp_mm"] = [float(v) for v in np.mean([r["mean_disp_mm"] for r in rows], 0)]
        s["sd_disp_mm"] = [float(v) for v in np.mean([r["sd_disp_mm"] for r in rows], 0)]
        summary[mot] = s
        print(name, mot, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if "ci95" not in k},
              flush=True)
    save_json(GAP / "results" / f"motion__{name}.json", dict(run=name, motions=motions, summary=summary, fields=per))
    np.savez_compressed(GAP / "results" / f"motion__{name}.npz", motions=np.array(motions),
                        **{k: np.concatenate(v) for k, v in tracks.items()})


if __name__ == "__main__":
    main()
