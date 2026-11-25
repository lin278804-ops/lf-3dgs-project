"""Recorded perforated plate at 32 stage positions: where does the frozen network place the holes?

  python real_plate.py --run p7 [--decode heatmap] [--no-condition]

The plate has 100 holes (centre distance 1 mm) and was translated along the optical axis in 0.5-mm steps.
Each recording is conditioned like the flow recordings and passed once through the network with the
calibration of the training camera, so centres are returned in that camera's coordinates. For each stage
position the lateral similarity transform (scale, rotation, shift, axis flips) between the supplier's hole
coordinates and the detections is estimated from the detections themselves; a hole is found if a
detection lies within 0.4 hole spacings of its transformed position. Depth is never used for matching.
Writes results/plate__<run>.json and .npz.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate import detect_heatmap, run_dir
from gapcommon import GAP, condition, detect, get_cam, load_model, save_json

NOMINAL_PER_MM = 99.0 / 80.9          # expected lateral scale: training camera 99.0 um per pixel, plate 80.9 um


def similarity(src, dst):
    """Least-squares similarity transform dst ~ s * R @ src + t (2-D, proper rotation)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, d]) @ vt
    s = (sv * [1, d]).sum() / (a ** 2).sum()
    return s, r, md - s * r @ ms


def align(holes, det):
    """Best of the four axis-flip hypotheses: translation from the mode of pairwise differences, then
    iterated nearest-neighbour refinement of a similarity transform."""
    best = None
    tree = cKDTree(det)
    for fx in (1, -1):
        for fy in (1, -1):
            src = holes * [fx, fy]
            diff = (det[:, None] - NOMINAL_PER_MM * src[None]).reshape(-1, 2)
            h, ex, ey = np.histogram2d(diff[:, 0], diff[:, 1], bins=[np.arange(-12, 12.01, 0.15)] * 2)
            i, j = np.unravel_index(np.argmax(h), h.shape)
            s, r, t = NOMINAL_PER_MM, np.eye(2), np.array([ex[i] + 0.075, ey[j] + 0.075])
            for gate in (0.5, 0.4, 0.4, 0.4):
                pred = s * src @ r.T + t
                dist, k = tree.query(pred)
                ok = dist < gate * NOMINAL_PER_MM
                if ok.sum() < 8:
                    break
                s, r, t = similarity(src[ok], det[k[ok]])
            pred = s * src @ r.T + t
            dist, k = tree.query(pred)
            ok = dist < 0.4 * s
            if best is None or ok.sum() > best["ok"].sum():
                best = dict(ok=ok, k=k, dist=dist, scale=float(s), rot_deg=float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))),
                            flip=(fx, fy), pred=pred)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p7")
    ap.add_argument("--decode", choices=["head", "heatmap"], default="head")
    ap.add_argument("--no-condition", action="store_true")
    ap.add_argument("--data", default="realplate", help="directory under data/ (recorded plate or simulated plate)")
    ap.add_argument("--scale0", type=float, default=None, help="initial lateral scale (default: recorded plate)")
    a = ap.parse_args()
    dev = "cuda:0"
    model, cfg = load_model(run_dir(a.run), dev)
    cam = get_cam()
    mcam = get_cam(depth=int(cfg.get("grid_depth", 64)))
    name = (Path(a.run).name if a.run not in ("p7", "scratch_s0") else a.run) + (f"__{a.decode}" if a.decode != "head" else "") + \
           ("__raw" if a.no_condition else "")
    global NOMINAL_PER_MM
    if a.scale0:
        NOMINAL_PER_MM = a.scale0
    if a.data != "realplate":
        name += f"__{a.data}"
    rows, keep = [], {}
    for f in sorted(glob.glob(str(GAP / "data" / a.data / "*.npz")), key=lambda p: float(np.load(p)["stage_mm"])):
        z = np.load(f)
        x = torch.from_numpy(z["sai"].astype(np.float32) / 255.0)
        if not a.no_condition:
            x = condition(x.to(dev)).cpu()
        full = torch.zeros(484, *x.shape[1:])
        full[torch.as_tensor(z["view_idx"])] = x
        xyz = detect_heatmap(model, mcam, full, dev, int(cfg["topk"]))[0] if a.decode == "heatmap" else \
            detect(model, mcam, full, dev, cam_score=cam)[0]
        holes, stage = z["holes_mm"][:, :2].astype(float), float(z["stage_mm"])
        row = dict(stage_mm=stage, detections=int(len(xyz)))
        if len(xyz) >= 20:
            al = align(holes, xyz[:, :2].astype(float))
            ok, k = al["ok"], al["k"]
            zz = xyz[k[ok], 2]
            inside = (np.abs(xyz[:, 0]) < np.abs(al["pred"][:, 0]).max() + 0.6) & (np.abs(xyz[:, 1]) < np.abs(al["pred"][:, 1]).max() + 0.6)
            row.update(holes_found=int(ok.sum()), unique_detections=int(len(set(k[ok]))),
                       detections_on_plate=int(inside.sum()), scale_nominal_per_mm=al["scale"], rotation_deg=al["rot_deg"],
                       flip=list(al["flip"]), lateral_residual_um=float(1000 * np.sqrt((al["dist"][ok] ** 2).mean()) / al["scale"]),
                       z_median=float(np.median(zz)) if ok.any() else None, z_mean=float(zz.mean()) if ok.any() else None,
                       z_sd=float(zz.std()) if ok.any() else None,
                       z_mad_sd=float(1.4826 * np.median(np.abs(zz - np.median(zz)))) if ok.any() else None)
            keep[f"z_{stage:+.1f}"] = zz.astype(np.float32)
            keep[f"hole_{stage:+.1f}"] = np.flatnonzero(ok).astype(np.int16)
        rows.append(row)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    use = [r for r in rows if r.get("holes_found", 0) >= 50]
    summary = dict(run=name, planes=len(rows), planes_with_50_holes=len(use))
    if len(use) >= 4:
        st = np.array([r["stage_mm"] for r in use])
        zm = np.array([r["z_median"] for r in use])
        g, off = np.polyfit(st, zm, 1)
        summary.update(stage_range_mm=[float(st.min()), float(st.max())], depth_gain=float(g), depth_offset_mm=float(off),
                       depth_linearity_residual_mm=float(np.std(zm - (g * st + off))),
                       depth_sd_within_plane_mm_median=float(np.median([r["z_mad_sd"] for r in use])),
                       holes_found_median=float(np.median([r["holes_found"] for r in use])),
                       lateral_residual_um_median=float(np.median([r["lateral_residual_um"] for r in use])),
                       scale_nominal_per_mm_median=float(np.median([r["scale_nominal_per_mm"] for r in use])),
                       um_per_px_from_network=float(99.0 / np.median([r["scale_nominal_per_mm"] for r in use])))
    print("SUMMARY", summary, flush=True)
    save_json(GAP / "results" / f"plate__{name}.json", dict(summary=summary, planes=rows))
    np.savez_compressed(GAP / "results" / f"plate__{name}.npz", **keep)


if __name__ == "__main__":
    main()
