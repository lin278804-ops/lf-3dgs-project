"""Recorded double-frame light fields: lateral displacement from linked reconstructions against the
whole-field cross-correlation of the same two exposures.

  python real_pairs.py --run p7 [--decode heatmap] [--data realpairs]
  python real_pairs.py --fit                       # centres from fit.py --data realpairs_fit

Each exposure is reconstructed on its own in the coordinates of the training camera; centres are linked
one-to-one within 0.5 mm and projected into the central view, so that displacements are compared in
sub-aperture pixels and no object-space scale enters. The reference is the peak of the cross-correlation
of the central views (mean of the four innermost views) of the two exposures. Controls: the second
exposure taken from the next pair (no common particles), and the displacement of the linked centres of
an exposure with itself is zero by construction and is not reported.

Writes results/real__<run>.json (+ .npz with the linked centres of every pair).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from evaluate import detect_heatmap, run_dir
from gapcommon import GAP, detect, get_cam, load_model, save_json
from lf_3dgs.model_v3 import DifferentiableLFProjection

R = 12
INNER = [10 * 22 + 10, 10 * 22 + 11, 11 * 22 + 10, 11 * 22 + 11]


def central(sai):
    return sai[INNER].astype(np.float64).mean(0)


def xcorr_peak(a, b):
    a, b = a - a.mean(), b - b.mean()
    H, W = a.shape
    c = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(a, s=(2 * H, 2 * W))) * np.fft.rfft2(b, s=(2 * H, 2 * W)),
                                      s=(2 * H, 2 * W)))[H - R:H + R + 1, W - R:W + R + 1]
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    sub = []
    for axis, i in ((0, iy), (1, ix)):
        if 0 < i < 2 * R:
            lo, mid, hi = (c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) if axis == 0 else (c[iy, ix - 1], c[iy, ix], c[iy, ix + 1])
            den = lo - 2 * mid + hi
            sub.append(i - R + (0.5 * (lo - hi) / den if den < 0 else 0.0))
        else:
            sub.append(float(i - R))
    m = c.copy()
    m[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -np.inf
    return float(sub[0]), float(sub[1]), float(c[iy, ix] / max(m.max(), 1e-12))      # d_row, d_col, peak ratio


def link(p0, p1, gate=0.5):
    if len(p0) == 0 or len(p1) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.sqrt(((p0[:, None] - p1[None]) ** 2).sum(2))
    c = d.copy()
    c[d > gate] = 1e6
    ri, ci = linear_sum_assignment(c)
    k = d[ri, ci] <= gate
    return ri[k], ci[k]


def mode_displacement(q0, q1, z0, z1, half=8.0, step=0.1, sigma=0.3, dz=0.6):
    """Most frequent lateral displacement among all candidate pairings of two reconstructions.

    Every centre of the first exposure is paired with every centre of the second within +-half px laterally
    and dz mm axially; the displacement vectors are accumulated on a step-px grid, smoothed (sigma px), and
    the highest peak is returned with its height relative to the median level of the plane (pairings of
    unrelated particles form a flat background). No ground truth and no image correlation are used.
    """
    from scipy.ndimage import gaussian_filter
    if len(q0) == 0 or len(q1) == 0:
        return None
    d = q1[None] - q0[:, None]
    ok = (np.abs(d[..., 0]) <= half) & (np.abs(d[..., 1]) <= half) & (np.abs(z1[None] - z0[:, None]) <= dz)
    if ok.sum() < 5:
        return None
    n = int(round(2 * half / step)) + 1
    h, _, _ = np.histogram2d(d[ok][:, 0], d[ok][:, 1], bins=n, range=[[-half - step / 2, half + step / 2]] * 2)
    h = gaussian_filter(h, sigma / step)
    iy, ix = np.unravel_index(np.argmax(h), h.shape)
    centre = np.array([-half + iy * step, -half + ix * step])
    near = ok & (np.hypot(d[..., 0] - centre[0], d[..., 1] - centre[1]) <= 0.75)
    # one-to-one tracks among the pairings consistent with the mode
    cost = np.where(near, np.hypot(d[..., 0] - centre[0], d[..., 1] - centre[1]), 1e6)
    ri, ci = linear_sum_assignment(cost)
    k = cost[ri, ci] < 1e5
    ri, ci = ri[k], ci[k]
    return dict(mode_row=float(centre[0]), mode_col=float(centre[1]), peak_over_background=float(h[iy, ix] / max(np.median(h), 1e-9)),
                tracks=int(len(ri)), track_row=float(d[ri, ci, 0].mean()) if len(ri) else None,
                track_col=float(d[ri, ci, 1].mean()) if len(ri) else None,
                track_dz_mean_mm=float((z1[ci] - z0[ri]).mean()) if len(ri) else None,
                track_dz_sd_mm=float((z1[ci] - z0[ri]).std()) if len(ri) else None,
                track_sd_row=float(d[ri, ci, 0].std()) if len(ri) else None, track_sd_col=float(d[ri, ci, 1].std()) if len(ri) else None,
                idx0=ri, idx1=ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--decode", choices=["head", "heatmap"], default="head")
    ap.add_argument("--data", default="realpairs")
    ap.add_argument("--alpha", type=float, default=None, help="occupancy threshold (default: frozen 0.05)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cam = get_cam()
    proj = DifferentiableLFProjection(cam)
    s_idx, t_idx = torch.tensor([10, 10, 11, 11]), torch.tensor([10, 11, 10, 11])

    def to_px(xyz):
        u, v = proj(torch.as_tensor(xyz, dtype=torch.float32), s_idx, t_idx)
        return np.stack([u.mean(0).numpy(), v.mean(0).numpy()], 1)            # row, col

    files = sorted(glob.glob(str(GAP / "data" / a.data / "*.npz")))[: a.limit]
    if a.fit:
        name = "fit"

        def recon(pair, k, sai):
            f = GAP / "fits" / (a.data + "_fit") / f"{pair}__f{k}.npz"
            return np.load(f)["fit_xyz"] if f.exists() else None
    else:
        dev = "cuda:0"
        model, cfg = load_model(run_dir(a.run), dev)
        mcam = get_cam(depth=int(cfg.get("grid_depth", 64)))
        name = (Path(a.run).name if a.run not in ("p7", "scratch_s0") else a.run) + \
               (f"__{a.decode}" if a.decode != "head" else "") + (f"__a{a.alpha:g}" if a.alpha else "")

        def recon(pair, k, sai):
            x = torch.from_numpy(sai.astype(np.float32) / 255.0)
            if a.decode == "heatmap":
                return detect_heatmap(model, mcam, x, dev, int(cfg["topk"]))[0]
            return detect(model, mcam, x, dev, cam_score=cam, **({"alpha_thr": a.alpha} if a.alpha else {}))[0]

    if a.data not in ("realpairs_all", "realpairs_full"):      # other recordings keep their own result files
        name += f"__{a.data}"
    rows, keep, prev = [], {}, None
    for f in files:
        z = np.load(f)
        pair = str(z["pair_id"])
        sai = z["sai"]
        if sai.shape[1] != 484:                 # compact files: only the views the network reads
            full = np.zeros((2, 484) + sai.shape[2:], sai.dtype)
            full[:, z["view_idx"]] = sai
            sai = full
        p = [recon(pair, k, sai[k]) for k in (0, 1)]
        if p[0] is None or p[1] is None:
            continue
        ref = xcorr_peak(central(sai[0]), central(sai[1]))
        i, j = link(p[0], p[1])
        d = to_px(p[1][j]) - to_px(p[0][i]) if len(i) else np.zeros((0, 2))
        dz = p[1][j, 2] - p[0][i, 2] if len(i) else np.zeros(0)
        row = dict(pair=pair, group=str(z["group"]), n0=int(len(p[0])), n1=int(len(p[1])), links=int(len(i)),
                   ref_row=ref[0], ref_col=ref[1], ref_peak_ratio=ref[2],
                   med_row=float(np.median(d[:, 0])) if len(i) else None, med_col=float(np.median(d[:, 1])) if len(i) else None,
                   mean_row=float(d[:, 0].mean()) if len(i) else None, mean_col=float(d[:, 1].mean()) if len(i) else None,
                   sd_row=float(d[:, 0].std()) if len(i) else None, sd_col=float(d[:, 1].std()) if len(i) else None,
                   dz_median_mm=float(np.median(dz)) if len(i) else None, dz_sd_mm=float(dz.std()) if len(i) else None)
        md = mode_displacement(to_px(p[0]), to_px(p[1]), p[0][:, 2], p[1][:, 2])
        if md:
            i, j = md.pop("idx0"), md.pop("idx1")
            row.update(md)
        if prev is not None and prev[0] == row["group"]:        # mismatched control: previous first exposure, this second
            mc = mode_displacement(to_px(prev[1]), to_px(p[1]), prev[1][:, 2], p[1][:, 2])
            if mc:
                row.update(ctrl_peak_over_background=mc["peak_over_background"], ctrl_tracks=mc["tracks"],
                           ctrl_mode_row=mc["mode_row"], ctrl_mode_col=mc["mode_col"])
        prev = (row["group"], p[0])
        rows.append(row)
        keep[pair + "_xyz0"], keep[pair + "_xyz1"] = p[0][i].astype(np.float32), p[1][j].astype(np.float32)
        if len(rows) <= 6:
            print(name, pair, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                               if k in ("n0", "n1", "tracks", "ref_row", "ref_col", "track_row", "track_col", "peak_over_background",
                                        "ctrl_peak_over_background", "track_dz_sd_mm")}, flush=True)
    summary = dict(run=name, pairs=len(rows))
    for label, sel in (("all", [r for r in rows if r.get("tracks")]),
                       ("clear_reference", [r for r in rows if r.get("tracks") and r["ref_peak_ratio"] >= 2.0])):
        if len(sel) < 3:
            continue
        ref = np.array([[r["ref_row"], r["ref_col"]] for r in sel])
        est = np.array([[r["track_row"], r["track_col"]] for r in sel])
        nn = np.array([[r["med_row"], r["med_col"]] for r in sel])
        ctrl = np.array([r["ctrl_peak_over_background"] for r in sel if "ctrl_peak_over_background" in r])
        summary[label] = dict(
            pairs=len(sel), tracks_median=float(np.median([r["tracks"] for r in sel])),
            detections_median=float(np.median([0.5 * (r["n0"] + r["n1"]) for r in sel])),
            corr_row=float(np.corrcoef(ref[:, 0], est[:, 0])[0, 1]), corr_col=float(np.corrcoef(ref[:, 1], est[:, 1])[0, 1]),
            slope_row=float(np.polyfit(ref[:, 0], est[:, 0], 1)[0]), slope_col=float(np.polyfit(ref[:, 1], est[:, 1], 1)[0]),
            rms_difference_px=float(np.sqrt(((est - ref) ** 2).sum(1).mean())),
            median_difference_px=float(np.median(np.hypot(*(est - ref).T))),
            within_0p5px=float((np.hypot(*(est - ref).T) <= 0.5).mean()),
            rms_reference_px=float(np.sqrt((ref ** 2).sum(1).mean())),
            nearest_link_median_slope=[float(np.polyfit(ref[:, k], nn[:, k], 1)[0]) for k in (0, 1)],
            peak_over_background_median=float(np.median([r["peak_over_background"] for r in sel])),
            mismatched_peak_over_background_median=float(np.median(ctrl)) if len(ctrl) else None,
            track_dz_sd_mm_median=float(np.median([r["track_dz_sd_mm"] for r in sel])),
            track_dz_mean_mm_median_abs=float(np.median(np.abs([r["track_dz_mean_mm"] for r in sel]))))
    print("SUMMARY", summary, flush=True)
    save_json(GAP / "results" / f"real__{name}.json", dict(summary=summary, pairs=rows))
    np.savez_compressed(GAP / "results" / f"real__{name}.npz", **keep)


if __name__ == "__main__":
    main()
