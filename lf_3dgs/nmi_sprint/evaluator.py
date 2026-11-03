"""Unified Hungarian detection evaluator for the NMI-minimum sprint.

Replaces greedy matching in profile_d4_errors for paper-facing metrics.
All Stage-2 / baseline / OOD evaluations must call this module.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from lf_3dgs.camera import LFCameraParams
from lf_3dgs.metrics import nms_gaussians
from lf_3dgs.model_v3 import LF3DGSNetV3


MATCH_RADIUS_VOX = 2.0
NMS_RADIUS_VOX = 1.5
ALPHA_THR = 0.05


def hungarian_match(
    pred: np.ndarray,
    gt: np.ndarray,
    match_radius: float,
) -> Dict:
    """One-to-one Hungarian matching with radius gate.

    Returns tp/fp/fn, matched error vectors (pred-gt), and index pairs.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    empty = np.empty((0, 3), dtype=np.float32)
    if pred.size == 0 and gt.size == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "errors": empty,
                "pred_idx": [], "gt_idx": [], "fp_idx": [], "fn_idx": list(range(0))}
    if pred.size == 0:
        return {"tp": 0, "fp": 0, "fn": len(gt), "errors": empty,
                "pred_idx": [], "gt_idx": [], "fp_idx": [],
                "fn_idx": list(range(len(gt)))}
    if gt.size == 0:
        return {"tp": 0, "fp": len(pred), "fn": 0, "errors": empty,
                "pred_idx": [], "gt_idx": [],
                "fp_idx": list(range(len(pred))), "fn_idx": []}

    d = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(axis=2))
    cost = d.copy()
    cost[d > match_radius] = 1e6
    ri, ci = linear_sum_assignment(cost)
    keep = d[ri, ci] <= match_radius
    ri, ci = ri[keep], ci[keep]
    errors = pred[ri] - gt[ci]
    used_p, used_g = set(map(int, ri)), set(map(int, ci))
    fp_idx = [i for i in range(len(pred)) if i not in used_p]
    fn_idx = [i for i in range(len(gt)) if i not in used_g]
    return {
        "tp": int(len(ri)),
        "fp": int(len(fp_idx)),
        "fn": int(len(fn_idx)),
        "errors": errors.astype(np.float32),
        "pred_idx": list(map(int, ri)),
        "gt_idx": list(map(int, ci)),
        "fp_idx": fp_idx,
        "fn_idx": fn_idx,
    }


def metrics_from_match(m: Dict) -> Dict[str, float]:
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    err = m["errors"]
    out = {
        "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "n_pred": float(tp + fp), "n_gt": float(tp + fn),
    }
    if len(err) > 0:
        dist = np.sqrt((err ** 2).sum(1))
        out.update({
            "mean_dist": float(dist.mean()),
            "bias_x": float(err[:, 0].mean()),
            "bias_y": float(err[:, 1].mean()),
            "bias_z": float(err[:, 2].mean()),
            "sigma_x": float(err[:, 0].std(ddof=0)),
            "sigma_y": float(err[:, 1].std(ddof=0)),
            "sigma_z": float(err[:, 2].std(ddof=0)),
        })
    else:
        out.update({
            "mean_dist": 0.0,
            "bias_x": 0.0, "bias_y": 0.0, "bias_z": 0.0,
            "sigma_x": 0.0, "sigma_y": 0.0, "sigma_z": 0.0,
        })
    return out


def load_v3_model(ckpt_path: str, device: str, topk: int = 1500,
                  base_channels: int = 32, n_views_cost: int = 32,
                  cam: Optional[LFCameraParams] = None):
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    cam = cam if cam is not None else LFCameraParams()
    model = LF3DGSNetV3(
        out_shape=tuple(cam.grid_shape),
        base_channels=int(cfg.get("base_channels", base_channels)),
        topk=int(cfg.get("topk", topk)),
        n_views_cost=int(cfg.get("n_views_cost", n_views_cost)),
        encode_native=bool(cfg.get("encode_native", True)),
        extract_mode=str(cfg.get("extract_mode", "peak")),
        bound_offset=bool(cfg.get("bound_offset", True)),
        view_sample=str(cfg.get("view_sample", "uniform")),
        skip_mode=str(cfg.get("skip_mode", "broadcast")),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cam


@torch.no_grad()
def infer_detections(model, cam: LFCameraParams, sai_stack: np.ndarray,
                     device: str, use_nms: bool = True,
                     alpha_thr: float = ALPHA_THR,
                     nms_radius: float = NMS_RADIUS_VOX
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pred_phys (N,3), pred_vox (N,3), alpha (N,)."""
    sai = sai_stack.astype(np.float32)
    if sai.max() > 1.0:
        sai = sai / 255.0
    V = U = cam.px_per_mla
    H, W = cam.num_mla_y, cam.num_mla_x
    t = torch.from_numpy(sai).unsqueeze(0).to(device)
    lf4d = t.reshape(1, V, U, H, W)
    out = model(lf4d, out_hw=(cam.grid_shape[1], cam.grid_shape[2]), cam_params=cam)
    mu = out["mu"][0].detach().cpu()
    alpha = torch.sigmoid(out["logit_alpha"][0].detach().cpu()).numpy()
    keep = alpha >= alpha_thr
    mu_a = mu[keep]
    alp_a = alpha[keep]
    if len(mu_a) == 0:
        z = np.empty((0, 3), dtype=np.float32)
        return z, z, np.empty((0,), dtype=np.float32)
    vox = cam.physical_to_voxel(mu_a).numpy().astype(np.float32)
    if use_nms and len(vox) > 1:
        nms_k = nms_gaussians(vox, alp_a, merge_radius=nms_radius)
        vox = vox[nms_k]
        alp_a = alp_a[nms_k]
        mu_a = mu_a.numpy()[nms_k]
    else:
        mu_a = mu_a.numpy()
    return mu_a.astype(np.float32), vox, alp_a.astype(np.float32)


def evaluate_split(
    ckpt: str,
    test_dir: str,
    device: str = "cuda:0",
    use_nms: bool = True,
    match_radius_vox: float = MATCH_RADIUS_VOX,
    output_path: Optional[str] = None,
    tag: str = "",
    cam: Optional[LFCameraParams] = None,
) -> Dict:
    """Evaluate a checkpoint on a directory of .npz samples (Hungarian)."""
    model, cam = load_v3_model(ckpt, device, cam=cam)
    files = []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(test_dir, f)) as z:
                if "sai_stack" in z.files and "particle_centers" in z.files:
                    files.append(f)
        except Exception:
            continue
    D, H, W = cam.grid_shape
    vx = (cam.x_range[1] - cam.x_range[0]) / W
    vy = (cam.y_range[1] - cam.y_range[0]) / H
    vz = (cam.z_range[1] - cam.z_range[0]) / D

    per_sample = []
    all_err_mm = []
    t0 = time.time()
    for i, fname in enumerate(files):
        data = np.load(os.path.join(test_dir, fname), allow_pickle=True)
        sai = data["sai_stack"].astype(np.float32)
        gt = data["particle_centers"].astype(np.float32)
        pred_phys, pred_vox, _ = infer_detections(
            model, cam, sai, device, use_nms=use_nms)
        gt_vox = (cam.physical_to_voxel(torch.from_numpy(gt)).numpy()
                  if len(gt) else np.empty((0, 3), dtype=np.float32))
        m = hungarian_match(pred_vox, gt_vox, match_radius_vox)
        # convert voxel errors -> mm
        if len(m["errors"]) > 0:
            ev = m["errors"]
            err_mm = np.column_stack([
                ev[:, 2] * vx, ev[:, 1] * vy, ev[:, 0] * vz,
            ]).astype(np.float32)
        else:
            err_mm = np.empty((0, 3), dtype=np.float32)
        m_mm = dict(m)
        m_mm["errors"] = err_mm
        sm = metrics_from_match(m_mm)
        sm["file"] = fname
        per_sample.append(sm)
        if len(err_mm):
            all_err_mm.append(err_mm)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {fname}: F1={sm['f1']:.4f}")

    f1s = [s["f1"] for s in per_sample]
    ps = [s["precision"] for s in per_sample]
    rs = [s["recall"] for s in per_sample]
    summary = {
        "tag": tag,
        "ckpt": os.path.abspath(ckpt),
        "test_dir": os.path.abspath(test_dir),
        "matcher": "hungarian",
        "match_radius_voxel": match_radius_vox,
        "use_nms": use_nms,
        "nms_radius_voxel": NMS_RADIUS_VOX,
        "n_samples": len(files),
        "mean_precision": float(np.mean(ps)),
        "mean_recall": float(np.mean(rs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "elapsed_sec": round(time.time() - t0, 2),
        "per_sample": per_sample,
    }
    if all_err_mm:
        e = np.vstack(all_err_mm)
        summary.update({
            "bias_x": float(e[:, 0].mean()),
            "bias_y": float(e[:, 1].mean()),
            "bias_z": float(e[:, 2].mean()),
            "sigma_x": float(e[:, 0].std(ddof=0)),
            "sigma_y": float(e[:, 1].std(ddof=0)),
            "sigma_z": float(e[:, 2].std(ddof=0)),
            "fp_rate": float(sum(s["fp"] for s in per_sample)
                             / max(sum(s["n_pred"] for s in per_sample), 1)),
            "fn_rate": float(sum(s["fn"] for s in per_sample)
                             / max(sum(s["n_gt"] for s in per_sample), 1)),
        })
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] saved -> {output_path}")
    print(f"[eval] F1={summary['mean_f1']:.4f} "
          f"P={summary['mean_precision']:.4f} R={summary['mean_recall']:.4f} "
          f"n={summary['n_samples']}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-nms", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base)
    evaluate_split(
        ckpt=args.ckpt, test_dir=args.test_dir, device=args.device,
        use_nms=not args.no_nms, output_path=args.output, tag=args.tag,
    )


if __name__ == "__main__":
    main()
