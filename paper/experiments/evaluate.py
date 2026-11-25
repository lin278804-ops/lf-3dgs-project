"""Score a network on directories of static fields with the frozen rule
(occupancy >= 0.05, suppression 1.5 voxels, one-to-one Hungarian gate 2 voxels).

  python evaluate.py --run p7 --data sealed_std
  python evaluate.py --run p7 --data sealed_std --snr 40 30 20 --tag-suffix noise
  python evaluate.py --run p7 --data sealed_std --decode heatmap
  python evaluate.py --run runs/seed1 --data sealed_std ds9test

--decode head     centres as returned by the network (default)
--decode heatmap  centre = peak node + parabolic refinement of log-occupancy along each axis
                  (training-free; node convention of the occupancy label)
Writes results/<run>__<data>[__snrXX][__decode].json (summary, per-field metrics) and .npz (predictions).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gapcommon import (ALPHA_THR, DS9, GAP, NMS_VOX, P7, SCRATCH_S0, add_noise, boot_ci, condition, detect, get_cam, load_field,
                       load_model, nms_gaussians, save_json, score, score_lateral, to_vox)

DATASETS = {"ds9test": DS9 / "test", "ds9val": DS9 / "val"}
BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def data_dir(name):
    return Path(DATASETS[name]) if name in DATASETS else GAP / "data" / name


def run_dir(name):
    if name == "p7":
        return P7
    if name == "scratch_s0":
        return SCRATCH_S0
    return Path(name) if os.path.isabs(name) else GAP / name


@torch.no_grad()
def detect_heatmap(model, cam, sai01, device, topk):
    V = cam.px_per_mla
    lf = sai01.to(device).reshape(1, V, V, cam.num_mla_y, cam.num_mla_x)
    out = model(lf, out_hw=(cam.grid_shape[1], cam.grid_shape[2]), cam_params=cam)
    logit = out["volume"]["alpha_logits"][0].float()                       # (D, H, W)
    D, H, W = logit.shape
    pooled = F.max_pool3d(logit[None, None], 3, 1, 1)[0, 0]
    scored = logit.masked_fill(logit < pooled - 1e-6, -1e4).reshape(-1)
    val, idx = torch.topk(scored, min(topk, scored.numel()))
    keep = torch.sigmoid(val) >= ALPHA_THR
    val, idx = val[keep], idx[keep]
    d, h, w = idx // (H * W), (idx % (H * W)) // W, idx % W
    lp = F.logsigmoid(logit)
    pad = F.pad(lp[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c0 = pad[d + 1, h + 1, w + 1]
    delta = []
    for dz, dy, dx in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        lo = pad[d + 1 - dz, h + 1 - dy, w + 1 - dx]
        hi = pad[d + 1 + dz, h + 1 + dy, w + 1 + dx]
        den = (lo - 2 * c0 + hi).clamp(max=-1e-6)
        delta.append((0.5 * (lo - hi) / den).clamp(-0.5, 0.5))
    vox = torch.stack([d.float() + delta[0], h.float() + delta[1], w.float() + delta[2]], 1).cpu().numpy()
    alpha = torch.sigmoid(val).cpu().numpy()
    xr, yr, zr = cam.x_range, cam.y_range, cam.z_range
    xyz = np.stack([xr[0] + vox[:, 2] / W * (xr[1] - xr[0]), yr[0] + vox[:, 1] / H * (yr[1] - yr[0]),
                    zr[0] + vox[:, 0] / D * (zr[1] - zr[0])], 1).astype(np.float32)
    vox = to_vox(xyz)
    if len(vox) > 1:
        k = nms_gaussians(vox, alpha, merge_radius=NMS_VOX)
        xyz, vox, alpha = xyz[k], vox[k], alpha[k]
    return xyz, vox.astype(np.float32), alpha.astype(np.float32)


def convert_output(xyz, cam_from, cam_to, device, views=None):
    """Map centres decoded under one calibration to the points with the same projections under another
    (least squares over the lookup views; Gauss-Newton from the decoded centre)."""
    from lf_3dgs.model_v3 import DifferentiableLFProjection
    if len(xyz) == 0:
        return xyz
    views = torch.linspace(0, 483, 32).long() if views is None else views
    s_idx, t_idx = (views // 22).to(device), (views % 22).to(device)
    pa, pb = DifferentiableLFProjection(cam_from).to(device), DifferentiableLFProjection(cam_to).to(device)
    x0 = torch.as_tensor(xyz, dtype=torch.float64, device=device)
    with torch.no_grad():
        ua, va = pa(x0.float(), s_idx, t_idx)
        target = torch.stack([ua, va], -1).double()                      # (K, N, 2)
    x = x0.clone()
    for _ in range(8):
        with torch.no_grad():
            u, v = pb(x.float(), s_idx, t_idx)
        r = torch.stack([u, v], -1).double() - target                    # (K, N, 2)
        # numerical Jacobian (projection is smooth and cheap): (K, N, 2, 3)
        eps = 1e-3
        Jn = []
        for a in range(3):
            d = torch.zeros(3, dtype=torch.float64, device=device)
            d[a] = eps
            with torch.no_grad():
                up, vp = pb((x + d).float(), s_idx, t_idx)
                um, vm = pb((x - d).float(), s_idx, t_idx)
            Jn.append(torch.stack([up - um, vp - vm], -1).double() / (2 * eps))
        Jn = torch.stack(Jn, -1).permute(1, 0, 2, 3).reshape(len(x), -1, 3)   # (N, 2K, 3)
        rr = r.detach().permute(1, 0, 2).reshape(len(x), -1, 1)
        H = Jn.transpose(1, 2) @ Jn + 1e-9 * torch.eye(3, dtype=torch.float64, device=device)
        x = x - torch.linalg.solve(H, Jn.transpose(1, 2) @ rr)[:, :, 0]
    return x.cpu().numpy().astype(np.float32)


def rectify(sai, cam_true, scale, device):
    """Rescale every view about the image of the optical axis by `scale` (bilinear), so that the lateral
    magnification equals that of the training cameras. The lookup must then use pitch / scale."""
    from lf_3dgs.model_v3 import DifferentiableLFProjection
    c = torch.tensor([10], device=device)
    with torch.no_grad():
        u0, v0 = DifferentiableLFProjection(cam_true).to(device)(torch.zeros(1, 3, device=device), c, c)
    H, W = sai.shape[-2:]
    r = u0.reshape(()) + (torch.arange(H, device=device).float() - u0.reshape(())) / scale
    q = v0.reshape(()) + (torch.arange(W, device=device).float() - v0.reshape(())) / scale
    rr, qq = torch.meshgrid(r, q, indexing="ij")
    grid = torch.stack([2.0 * qq / (W - 1) - 1.0, 2.0 * rr / (H - 1) - 1.0], -1)[None]
    return F.grid_sample(sai.to(device).float()[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def summarize(rows, preds):
    f1 = np.array([r["f1"] for r in rows])
    out = {"n_fields": len(rows), "f1": float(f1.mean()), "f1_ci95": boot_ci(f1), "f1_sd": float(f1.std(ddof=1)),
           "precision": float(np.mean([r["precision"] for r in rows])),
           "recall": float(np.mean([r["recall"] for r in rows])),
           "lateral_f1": float(np.mean([r["lateral_f1"] for r in rows])),
           "lateral_recall": float(np.mean([r["lateral_recall"] for r in rows])),
           "lateral_abs_dz_median_mm": float(np.mean([r["lateral_abs_dz_median_mm"] for r in rows
                                                      if r["lateral_abs_dz_median_mm"] is not None])),
           "tp": int(sum(r["tp"] for r in rows)), "fp": int(sum(r["fp"] for r in rows)),
           "fn": int(sum(r["fn"] for r in rows))}
    err = np.concatenate([p["err_mm"] for p in preds]) if preds else np.zeros((0, 3))
    if len(err):
        out.update(err_sd_um=[float(v) for v in 1000 * err.std(0)], err_bias_um=[float(v) for v in 1000 * err.mean(0)],
                   err_mae_um=[float(v) for v in 1000 * np.abs(err).mean(0)],
                   err3d_mm=float(np.linalg.norm(err, axis=1).mean()),
                   axial_mae_um_field_mean=float(np.mean([1000 * np.abs(p["err_mm"][:, 2]).mean() for p in preds
                                                          if len(p["err_mm"])])))
    band = np.zeros((len(preds), 5, 2))
    for i, p in enumerate(preds):
        az = np.abs(p["gt"][:, 2])
        hit = np.zeros(len(az), bool)
        hit[p["gt_idx"]] = True
        for k, (lo, hi) in enumerate(BANDS):
            m = (az >= lo) & (az < hi)
            band[i, k] = [hit[m].sum(), m.sum()]
    out["recall_by_depth_band"] = [float(v) for v in (band[:, :, 0] / np.maximum(band[:, :, 1], 1)).mean(0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best.pt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--snr", nargs="*", type=float, default=[])
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--decode", choices=["head", "heatmap"], default="head")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--f-lookup", type=float, default=0.8)
    ap.add_argument("--F-lookup", type=float, default=100.0, help="main-lens focal length in the lookup (mm)")
    ap.add_argument("--L-lookup", type=float, default=200.0, help="object distance in the lookup (mm)")
    ap.add_argument("--rectify", action="store_true",
                    help="rescale the views to the lateral magnification of the training cameras (1.0) before the network")
    ap.add_argument("--convert-to", type=float, default=None,
                    help="microlens focal length of the camera that took the data; decoded centres are mapped to it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--views", type=int, default=None, help="number of views in the lookup at inference")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--condition", action="store_true", help="apply the input conditioning used for recorded frames")
    ap.add_argument("--intervene", choices=["reverse"], default=None,
                    help="reverse: each view is read at the addresses of its mirror view (images and weights unchanged)")
    ap.add_argument("--blur", type=float, default=3.0, help="background scale of the conditioning (px); tuning only")
    ap.add_argument("--floor", type=float, default=3.0, help="noise floor of the conditioning (robust s.d.); tuning only")
    a = ap.parse_args()

    dev = "cuda:0"
    rd = run_dir(a.run)
    model, cfg = load_model(rd, dev, a.which, topk=a.topk)
    if a.views:
        model.n_views_cost = int(a.views)
    cam = get_cam(a.f_lookup, depth=int(cfg.get("grid_depth", 64)))
    model.cam_f = a.f_lookup                      # used only by the calibration-conditioned lifting baseline
    cam.f_main, cam.u_obj = float(a.F_lookup), float(a.L_lookup)
    cam_true, rect_scale = None, 1.0
    if a.rectify:
        cam_true = get_cam(a.f_lookup, depth=int(cfg.get("grid_depth", 64)))
        cam_true.f_main, cam_true.u_obj = float(a.F_lookup), float(a.L_lookup)
        rect_scale = 1.0 / cam.magnification
        cam.pitch = cam.pitch / rect_scale        # projected coordinates scale about the optical axis like the images
    if a.intervene == "reverse":
        original = model._build_cost_volume

        def reversed_lookup(feat_down, s_idx, t_idx, cam_params, cv_shape):
            return original(feat_down, 21 - s_idx, 21 - t_idx, cam_params, cv_shape)
        model._build_cost_volume = reversed_lookup
    cam_score = get_cam()
    name = Path(a.run).name if a.run not in ("p7", "scratch_s0") else a.run
    gen = torch.Generator(device=dev)
    for ds in a.data:
        files = sorted(glob.glob(str(data_dir(ds) / "*.npz")))[: a.limit]
        for snr in ([None] + list(a.snr)):
            tag = f"{name}__{ds}" + (f"__snr{snr:g}" if snr is not None else "") + \
                  (f"__{a.decode}" if a.decode != "head" else "") + (f"__topk{a.topk}" if a.topk else "") + (f"__views{a.views}" if a.views else "") + \
                  (f"__f{a.f_lookup:g}" if a.f_lookup != 0.8 else "") + (f"__conv{a.convert_to:g}" if a.convert_to else "") + \
                  (f"__F{a.F_lookup:g}" if a.F_lookup != 100.0 else "") + (f"__L{a.L_lookup:g}" if a.L_lookup != 200.0 else "") + \
                  ("__rect" if a.rectify else "") + \
                  ("__cond" if a.condition else "") + (f"__{a.intervene}" if a.intervene else "") + \
                  (f"__{a.tag_suffix}" if a.tag_suffix else "")
            rows, preds, t0 = [], [], time.time()
            for i, f in enumerate(files):
                sai, gt = load_field(f)
                if snr is not None or a.gain != 1.0:
                    gen.manual_seed(1000 + i)
                    sai = add_noise(sai[None].to(dev), snr, gen, gain=a.gain)[0]
                if a.rectify:
                    sai = rectify(sai, cam_true, rect_scale, dev)
                if a.condition:
                    sai = condition(sai.to(dev), k=a.floor, blur=a.blur)
                if a.decode == "heatmap":
                    xyz, vox, alpha = detect_heatmap(model, cam, sai, dev, int(cfg["topk"]))
                else:
                    xyz, vox, alpha = detect(model, cam, sai, dev, cam_score=cam_score)
                if a.convert_to:
                    xyz = convert_output(xyz, cam_score, get_cam(a.convert_to), dev)
                m = score(xyz, gt, cam_score)
                lat = score_lateral(xyz, gt, cam_score)
                rows.append({k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")} |
                            {"file": Path(f).name, "n_pred": len(xyz), "lateral_f1": lat["lateral_f1"],
                             "lateral_recall": lat["lateral_recall"],
                             "lateral_abs_dz_median_mm": float(np.median(lat["abs_dz_mm"])) if len(lat["abs_dz_mm"]) else None})
                preds.append(dict(xyz=xyz, alpha=alpha, gt=gt, pred_idx=m["pred_idx"], gt_idx=m["gt_idx"],
                                  err_mm=m["err_mm"]))
            summary = summarize(rows, preds)
            summary.update(run=str(rd), which=a.which, data=ds, snr_db=snr, decode=a.decode, topk=int(cfg["topk"]),
                           f_lookup=a.f_lookup, F_lookup=a.F_lookup, L_lookup=a.L_lookup, rectified=bool(a.rectify),
                           rectify_scale=float(rect_scale), seconds=time.time() - t0)
            save_json(GAP / "results" / f"{tag}.json", dict(summary=summary, fields=rows))
            np.savez_compressed(GAP / "results" / f"{tag}.npz",
                                **{f"{k}_{i:04d}": p[k] for i, p in enumerate(preds) for k in
                                   ("xyz", "alpha", "gt", "pred_idx", "gt_idx")})
            print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()
                        if k in ("n_fields", "f1", "precision", "recall", "err_sd_um", "err_bias_um",
                                 "recall_by_depth_band")}, flush=True)


if __name__ == "__main__":
    main()
