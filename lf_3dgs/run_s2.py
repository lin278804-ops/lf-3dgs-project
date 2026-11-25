"""
LF-3DGS Stage 2: Experiment runner.

Supports all experiment phases:
  Phase 0: Sanity checks (SC0 smoke test, SC1 single overfit, SC2 small overfit)
  Phase 1: Baselines (B1 single-conc, B2 full dataset, B3 eval old model)
  Phase 2: Hyperparameter sweeps (H1 topk, H2 loss, H3 capacity, H4 training)

Usage:
    python -m lf_3dgs.run_s2 --phase SC0 --device cuda:0
    python -m lf_3dgs.run_s2 --phase all_sanity --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .camera import LFCameraParams
from .dataset import LF3DGSDataset
from .losses import (
    chamfer_distance, charbonnier_loss, hungarian_matching_loss,
    l1_loss, mse_loss, volume_peak_hardneg_loss,
)
from .metrics import evaluate_gaussians
from .model import LF3DGSNet
from .model_v2 import LF3DGSNetV2
from .model_v3 import LF3DGSNetV3
from .renderer import GaussianSplatRenderer

DATASET4_ROOT = "output/lightfield_dataset4"
DATASET6_ROOT = "output/lightfield_dataset6"
DATASET7_ROOT = "output/lightfield_dataset7"
DATASET8_ROOT = "output/lightfield_dataset8"
DATASET8_CONC005_ROOT = "output/lightfield_dataset8_conc005"
TEACHER6_ROOT = "output/stage1_teachers_ds6"
TEACHER7_ROOT = "output/stage1_teachers_ds7"
TEACHER8_ROOT = "output/stage1_teachers_ds8"
OUTPUT_ROOT = "output/lf3dgs_s2"

VOLUME_CFG = {
    "x_range": [-10.0, 10.0],
    "y_range": [-8.0, 8.0],
    "z_range": [-5.0, 5.0],
    "grid_shape": [64, 128, 128],
}

BASE_S2_CFG = {
    "base_channels": 32,
    "depth_bins": 64,
    "topk": 3500,
    "epochs": 200,
    "batch_size": 1,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "lr_scheduler": "none",
    "render_loss_weight": 1.0,
    "center_loss_weight": 0.0,
    "center_loss_type": "chamfer",
    "render_loss_type": "charbonnier",
    "volume_alpha_weight": 1.0,
    "views_per_step": 16,
    "log_every": 10,
}


# ---- Volume-level alpha supervision helpers ----

def build_gt_occupancy(centers, cam, device, mode="hard", sigma=1.0):
    """Build a (D, H, W) occupancy grid from GT particle centres.

    mode="hard": one-hot voxel (legacy).
    mode="gaussian": CenterNet-style soft peak, σ in voxels.
    """
    D, H, W = cam.grid_shape
    occ = torch.zeros(D, H, W, device=device)
    if centers.shape[0] == 0:
        return occ
    centers = centers.to(device)
    vox = cam.physical_to_voxel(centers)  # (N, 3) continuous [z, y, x]
    if mode != "gaussian":
        wi = vox[:, 2].long().clamp(0, W - 1)
        hi = vox[:, 1].long().clamp(0, H - 1)
        di = vox[:, 0].long().clamp(0, D - 1)
        occ[di, hi, wi] = 1.0
        return occ

    r = max(1, int(math.ceil(3.0 * float(sigma))))
    zz = torch.arange(D, device=device, dtype=torch.float32)
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    inv = 1.0 / (2.0 * float(sigma) ** 2)
    for n in range(vox.shape[0]):
        z, y, x = vox[n]
        zi, yi, xi = int(z.item()), int(y.item()), int(x.item())
        z0, z1 = max(0, zi - r), min(D, zi + r + 1)
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)
        x0, x1 = max(0, xi - r), min(W, xi + r + 1)
        zg = zz[z0:z1][:, None, None]
        yg = yy[y0:y1][None, :, None]
        xg = xx[x0:x1][None, None, :]
        g = torch.exp(-((zg - z) ** 2 + (yg - y) ** 2 + (xg - x) ** 2) * inv)
        patch = occ[z0:z1, y0:y1, x0:x1]
        occ[z0:z1, y0:y1, x0:x1] = torch.maximum(patch, g)
    return occ.clamp(0.0, 1.0)


def centernet_focal_loss(logits, targets, alpha=2.0, beta=4.0):
    """CornerNet / CenterNet focal loss for soft Gaussian occupancy."""
    prob = torch.sigmoid(logits)
    pos_mask = targets >= 0.9
    neg_mask = ~pos_mask
    pos_loss = ((1.0 - prob) ** alpha) * F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(targets), reduction="none")
    neg_w = (1.0 - targets) ** beta
    neg_loss = neg_w * (prob ** alpha) * F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(targets), reduction="none")
    loss = (pos_loss * pos_mask.float() + neg_loss * neg_mask.float()).sum()
    return loss / pos_mask.float().sum().clamp(min=1.0)


def offset_peak_l1(pred_off, centers, cam):
    """L1 on (dx,dy,dz) only at GT-occupied voxels. pred_off: (D,H,W,3)."""
    if centers.shape[0] == 0:
        return pred_off.new_tensor(0.0)
    D, H, W, _ = pred_off.shape
    vox = cam.physical_to_voxel(centers.to(pred_off.device))
    zi = vox[:, 0].floor().long().clamp(0, D - 1)
    yi = vox[:, 1].floor().long().clamp(0, H - 1)
    xi = vox[:, 2].floor().long().clamp(0, W - 1)
    target = torch.stack([
        vox[:, 2] - (xi.float() + 0.5),
        vox[:, 1] - (yi.float() + 0.5),
        vox[:, 0] - (zi.float() + 0.5),
    ], dim=-1).clamp(-0.5, 0.5)
    pred = pred_off[zi, yi, xi]
    return (pred - target).abs().mean()


def focal_bce_loss(logits, targets, gamma=2.0, pos_weight=50.0):
    """Focal BCE loss for extremely imbalanced occupancy grids.

    Args:
        logits: raw alpha logit volume (D, H, W)
        targets: binary or soft occupancy (D, H, W)
        gamma: focal modulation exponent (higher = more focus on hard examples)
        pos_weight: scalar weight multiplier for positive voxels
    """
    weight = torch.ones_like(targets)
    weight[targets > 0.5] = pos_weight
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, reduction="none")
    prob = torch.sigmoid(logits)
    pt = targets * prob + (1 - targets) * (1 - prob)
    focal = ((1 - pt) ** gamma) * bce
    return focal.mean()


def build_soft_teacher_volume(mu_np, logit_alpha_np, cam, device,
                               sigma_vox=1.5, alpha_thr=0.05):
    """Smooth (0..1) teacher occupancy volume from Stage 1 Gaussians."""
    D, H, W = cam.grid_shape
    alpha_np = 1.0 / (1.0 + np.exp(-logit_alpha_np.astype(np.float64)))
    active = alpha_np > alpha_thr
    if active.sum() == 0:
        return torch.zeros(D, H, W, device=device)
    mu_t = torch.tensor(mu_np[active], dtype=torch.float32)
    alpha_t = torch.tensor(alpha_np[active], dtype=torch.float32)
    vox = cam.physical_to_voxel(mu_t)
    z_idx = vox[:, 0].long().clamp(0, D - 1)
    y_idx = vox[:, 1].long().clamp(0, H - 1)
    x_idx = vox[:, 2].long().clamp(0, W - 1)
    vol = torch.zeros(D, H, W)
    for n in range(len(z_idx)):
        zi, yi, xi = z_idx[n].item(), y_idx[n].item(), x_idx[n].item()
        vol[zi, yi, xi] = max(vol[zi, yi, xi].item(), alpha_t[n].item())
    vol = vol.unsqueeze(0).unsqueeze(0)
    k = max(1, int(sigma_vox * 3))
    coords = torch.arange(-k, k + 1, dtype=torch.float32)
    kern = torch.exp(-coords ** 2 / (2.0 * sigma_vox ** 2))
    kern = kern / kern.sum()
    vol = F.conv3d(vol, kern.view(1, 1, -1, 1, 1), padding=(k, 0, 0))
    vol = F.conv3d(vol, kern.view(1, 1, 1, -1, 1), padding=(0, k, 0))
    vol = F.conv3d(vol, kern.view(1, 1, 1, 1, -1), padding=(0, 0, k))
    return vol.squeeze(0).squeeze(0).clamp(0.0, 1.0).to(device)


def teacher_bce_loss(logits, soft_targets, pos_weight=50.0):
    """Weighted BCE between predicted alpha logits and soft teacher targets."""
    weight = torch.ones_like(soft_targets)
    weight[soft_targets > 0.1] = pos_weight
    return F.binary_cross_entropy_with_logits(logits, soft_targets, weight=weight)


def load_teacher(sample_path, teacher_dir):
    """Load Stage 1 teacher (mu, logit_alpha) for a sample, or (None, None)."""
    if not teacher_dir:
        return None, None
    name = os.path.splitext(os.path.basename(sample_path))[0]
    tpath = os.path.join(teacher_dir, f"{name}.npz")
    if not os.path.exists(tpath):
        return None, None
    try:
        t = np.load(tpath)
        return t["mu"], t["logit_alpha"]
    except Exception:
        return None, None


# ---- Camera / data helpers ----

def get_cam(vol=None):
    if vol is None:
        vol = VOLUME_CFG
    cam = LFCameraParams()
    cam.x_range = tuple(vol["x_range"])
    cam.y_range = tuple(vol["y_range"])
    cam.z_range = tuple(vol["z_range"])
    cam.grid_shape = tuple(vol["grid_shape"])
    return cam


def get_base_dir():
    return os.path.dirname(os.path.dirname(__file__))


def collate_fn(batch):
    sai_list, center_list, meta_list, path_list = [], [], [], []
    for sai, centers, meta, path in batch:
        sai_list.append(torch.from_numpy(sai))
        center_list.append(torch.from_numpy(centers))
        meta_list.append(meta)
        path_list.append(path)
    return torch.stack(sai_list, 0), center_list, meta_list, path_list


def filter_dataset_by_conc(dataset, target_conc, tol=0.005):
    indices = []
    for i in range(len(dataset)):
        _, _, meta, _ = dataset[i]
        if abs(meta.get("ppm_concentration", -1) - target_conc) < tol:
            indices.append(i)
    return Subset(dataset, indices)


def make_loaders(train_ds, val_ds, bs=1):
    nw = int(os.environ.get("LF_NUM_WORKERS", "8"))
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                    collate_fn=collate_fn, num_workers=nw,
                    pin_memory=True,
                    persistent_workers=(nw > 0),
                    prefetch_factor=(4 if nw > 0 else None))
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                    collate_fn=collate_fn, num_workers=max(2, nw // 2),
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=4)
    return tl, vl


# ---- Training loop ----

def train_one_epoch(model, renderer, loader, optimizer, cam, cfg, device,
                    scaler=None, teacher_dir=None):
    """
    Train one epoch.

    Args:
        scaler:      torch.cuda.amp.GradScaler for AMP (or None for fp32)
        teacher_dir: directory with Stage-1 teacher .npz labels (or None)
    """
    model.train()
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in mod.parameters()):
                mod.eval()
    total_sum, render_sum, center_sum, vol_sum, teacher_sum, off_sum, hn_sum = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_batches = 0
    K = int(cfg.get("views_per_step", 16))
    P = cam.px_per_mla
    render_w = float(cfg.get("render_loss_weight", 1.0))
    center_w = float(cfg.get("center_loss_weight", 0.0))
    vol_w = float(cfg.get("volume_alpha_weight", 1.0))
    teacher_w = float(cfg.get("teacher_loss_weight", 0.0))
    center_type = str(cfg.get("center_loss_type", "chamfer"))
    render_type = str(cfg.get("render_loss_type", "charbonnier"))
    render_fn = {"charbonnier": charbonnier_loss, "l1": l1_loss,
                 "mse": mse_loss}.get(render_type, charbonnier_loss)
    focal_gamma = float(cfg.get("focal_bce_gamma", 2.0))
    focal_pos_w = float(cfg.get("focal_bce_pos_weight", 50.0))
    occ_mode = str(cfg.get("occupancy_label", "gaussian"))
    occ_sigma = float(cfg.get("occupancy_sigma", 1.0))
    vol_loss_type = str(cfg.get("volume_loss_type", "centernet"))
    offset_w = float(cfg.get("offset_loss_weight", 1.0))
    count_w = float(cfg.get("count_loss_weight", 0.0))
    hn_w = float(cfg.get("hn_loss_weight", 0.0))
    hn_topk = int(cfg.get("hn_topk", 512))
    hn_radius = float(cfg.get("hn_match_radius_vox", 2.0))
    hn_target_logit = cfg.get("hn_target_logit", None)
    if hn_target_logit is not None:
        hn_target_logit = float(hn_target_logit)
    use_amp = scaler is not None

    for sai_batch, center_list, meta_list, path_list in loader:
        sai_batch = sai_batch.to(device)
        B = sai_batch.shape[0]
        C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
        V = U = int(math.sqrt(C))
        lf4d = sai_batch.reshape(B, V, U, H, W)

        out_hw = (cam.grid_shape[1], cam.grid_shape[2])

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(lf4d, out_hw=out_hw, cam_params=cam)

            # Render loss
            loss_render = torch.tensor(0.0, device=device)
            s_idx, t_idx = renderer.sample_views(K, P, device)
            flat_idx = renderer.sai_index(s_idx, t_idx, P)
            for i in range(B):
                rendered = renderer(outputs["mu"][i], outputs["log_sigma"][i],
                                    outputs["logit_alpha"][i], s_idx, t_idx)
                gt_views = sai_batch[i, flat_idx]
                g_max = gt_views.max().detach().clamp(min=1e-6)
                loss_render += render_fn(rendered / g_max, gt_views / g_max)
            loss_render = loss_render / B

            # Volume-level alpha supervision
            loss_vol = torch.tensor(0.0, device=device)
            if vol_w > 0 and "volume" in outputs:
                alpha_logits_vol = outputs["volume"]["alpha_logits"]
                for i in range(B):
                    gt_occ = build_gt_occupancy(
                        center_list[i].to(device), cam, device,
                        mode=occ_mode, sigma=occ_sigma)
                    if vol_loss_type == "centernet" and occ_mode == "gaussian":
                        loss_vol += centernet_focal_loss(
                            alpha_logits_vol[i], gt_occ,
                            alpha=focal_gamma, beta=4.0)
                    else:
                        loss_vol += focal_bce_loss(
                            alpha_logits_vol[i], gt_occ,
                            gamma=focal_gamma, pos_weight=focal_pos_w)
                loss_vol = loss_vol / B

            loss_off = torch.tensor(0.0, device=device)
            if offset_w > 0 and "volume" in outputs:
                pred_off = outputs["volume"]["offsets"]
                for i in range(B):
                    loss_off += offset_peak_l1(
                        pred_off[i], center_list[i].to(device), cam)
                loss_off = loss_off / B

            # Teacher supervision (soft alpha targets from Stage 1)
            loss_teacher = torch.tensor(0.0, device=device)
            if teacher_w > 0 and teacher_dir and "volume" in outputs:
                alpha_logits_vol = outputs["volume"]["alpha_logits"]
                n_teacher = 0
                for i in range(B):
                    t_mu, t_la = load_teacher(path_list[i], teacher_dir)
                    if t_mu is None:
                        continue
                    soft_vol = build_soft_teacher_volume(t_mu, t_la, cam, device)
                    loss_teacher += teacher_bce_loss(alpha_logits_vol[i], soft_vol)
                    n_teacher += 1
                if n_teacher > 0:
                    loss_teacher = loss_teacher / n_teacher

            # Volume local-max hard negatives (not just extracted top-K).
            loss_hn = torch.tensor(0.0, device=device)
            if hn_w > 0 and "volume" in outputs:
                alpha_logits_vol = outputs["volume"]["alpha_logits"]
                for i in range(B):
                    loss_hn = loss_hn + volume_peak_hardneg_loss(
                        alpha_logits_vol[i], center_list[i].to(device), cam,
                        match_radius_vox=hn_radius, topk_neg=hn_topk,
                        target_logit=hn_target_logit)
                loss_hn = loss_hn / max(B, 1)

            # Soft cardinality on extracted peaks: extras should carry low α
            # so the frozen official cut (α=0.05) can drop them.
            loss_count = torch.tensor(0.0, device=device)
            if count_w > 0:
                for i in range(B):
                    a = torch.sigmoid(outputs["logit_alpha"][i].float())
                    n_gt = max(float(center_list[i].shape[0]), 1.0)
                    loss_count = loss_count + (a.sum() - n_gt).abs() / n_gt
                loss_count = loss_count / max(B, 1)

            # Centre matching loss (Chamfer or Hungarian on extracted top-K)
            loss_center = torch.tensor(0.0, device=device)
            hung_pos_w = float(cfg.get("hung_pos_weight", 0.1))
            hung_a_w = float(cfg.get("hung_alpha_weight", 1.0))
            hung_r = float(cfg.get("hung_match_radius_vox", 2.0))
            if center_type == "hungarian" or (center_w > 0 and center_type != "hungarian"):
                for i in range(B):
                    mu_i = outputs["mu"][i]
                    gt_i = center_list[i].to(device)
                    if gt_i.shape[0] == 0 or mu_i.shape[0] == 0:
                        continue
                    if center_type == "hungarian":
                        # Same space/radius as official evaluator (voxel, r=2.0):
                        # matched peaks → high α, unmatched extras → α→0 so the
                        # frozen α=0.05 cut can drop them.
                        pred_v = cam.physical_to_voxel(mu_i)
                        gt_v = cam.physical_to_voxel(gt_i)
                        hung = hungarian_matching_loss(
                            pred_v, outputs["logit_alpha"][i], gt_v,
                            match_radius=hung_r)
                        loss_center = loss_center + (
                            hung_pos_w * hung["position"]
                            + hung_a_w * hung["alpha"])
                    else:
                        loss_center = loss_center + chamfer_distance(mu_i, gt_i)
                loss_center = loss_center / max(B, 1)

            center_term = (loss_center if center_type == "hungarian"
                           else center_w * loss_center)
            loss = (render_w * loss_render
                    + center_term
                    + vol_w * loss_vol
                    + teacher_w * loss_teacher
                    + offset_w * loss_off
                    + count_w * loss_count
                    + hn_w * loss_hn)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_sum += loss.item()
        render_sum += loss_render.item()
        center_sum += loss_center.item()
        vol_sum += loss_vol.item()
        teacher_sum += loss_teacher.item()
        off_sum += loss_off.item()
        hn_sum += loss_hn.item()
        n_batches += 1

    return {"loss": total_sum / max(n_batches, 1),
            "render_loss": render_sum / max(n_batches, 1),
            "center_loss": center_sum / max(n_batches, 1),
            "vol_loss": vol_sum / max(n_batches, 1),
            "teacher_loss": teacher_sum / max(n_batches, 1),
            "offset_loss": off_sum / max(n_batches, 1),
            "hn_loss": hn_sum / max(n_batches, 1)}


@torch.no_grad()
def validate_model(model, loader, cam, device,
                   alpha_threshold=None, use_nms=False, nms_radius_voxel=1.5):
    """Evaluate model on a data loader.

    Optional NMS / alpha threshold post-processing:
      alpha_threshold  – override the default alpha cut-off in evaluate_gaussians
      use_nms          – apply voxel-space NMS after thresholding
      nms_radius_voxel – NMS suppression radius (voxels)
    """
    model.eval()
    all_m = []
    eval_kwargs = {}
    if alpha_threshold is not None:
        eval_kwargs["alpha_threshold"] = alpha_threshold
    if use_nms:
        eval_kwargs["use_nms"] = True
        eval_kwargs["nms_radius_voxel"] = nms_radius_voxel
    for sai_batch, center_list, meta_list, _ in loader:
        sai_batch = sai_batch.to(device)
        B = sai_batch.shape[0]
        C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
        V = U = int(math.sqrt(C))
        lf4d = sai_batch.reshape(B, V, U, H, W)
        out_hw = (cam.grid_shape[1], cam.grid_shape[2])
        outputs = model(lf4d, out_hw=out_hw, cam_params=cam)
        for i in range(B):
            m = evaluate_gaussians(outputs["mu"][i], outputs["logit_alpha"][i],
                                   center_list[i], cam, **eval_kwargs)
            all_m.append(m)
    if not all_m:
        return {"precision": 0, "recall": 0, "f1": 0, "mean_dist": 99}
    return {k: float(np.mean([m[k] for m in all_m]))
            for k in ("precision", "recall", "f1", "mean_dist")}


def run_training(cfg, train_loader, val_loader, output_dir, device, tag="",
                 model_cls="v1", teacher_dir=None, pretrained_ckpt=None,
                 cam=None, ckpt_select="val_f1", trainable_substr=None):
    dev = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    cam = cam if cam is not None else get_cam()
    topk = int(cfg.get("topk", 3500))
    bc = int(cfg.get("base_channels", 32))
    n_views_cost = int(cfg.get("n_views_cost", 16))
    if model_cls == "v3":
        model = LF3DGSNetV3(
            out_shape=tuple(cam.grid_shape),
            base_channels=bc, topk=topk,
            n_views_cost=n_views_cost,
            encode_native=bool(cfg.get("encode_native", True)),
            extract_mode=str(cfg.get("extract_mode", "peak")),
            bound_offset=bool(cfg.get("bound_offset", True)),
            view_sample=str(cfg.get("view_sample", "uniform")),
            skip_mode=str(cfg.get("skip_mode", "broadcast")),
        ).to(dev)
    elif model_cls == "v2":
        model = LF3DGSNetV2(out_shape=tuple(cam.grid_shape),
                             base_channels=bc, topk=topk,
                             n_views_cost=n_views_cost).to(dev)
    else:
        model = LF3DGSNet(out_shape=tuple(cam.grid_shape),
                           base_channels=bc, topk=topk).to(dev)
    if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
        state = torch.load(pretrained_ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[S2] Loaded pretrained weights from {pretrained_ckpt}")
        if missing:
            print(f"[S2]   missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[S2]   unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    renderer = GaussianSplatRenderer(cam).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt_params = list(model.parameters())
    if trainable_substr:
        opt_params = []
        for name, param in model.named_parameters():
            param.requires_grad = trainable_substr in name
            if param.requires_grad:
                opt_params.append(param)
        n_train = sum(p.numel() for p in opt_params)
        print(f"[S2] trainable '{trainable_substr}': {n_train:,} / {n_params:,}")
        if not opt_params:
            raise RuntimeError(f"no parameters match {trainable_substr}")
    if bool(cfg.get("freeze_bn", False)):
        n_bn = 0
        for mod in model.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in mod.parameters():
                    param.requires_grad = False
                    n_bn += param.numel()
        opt_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[S2] BN frozen ({n_bn:,} affine params, running stats stay in eval)")
        if not opt_params:
            raise RuntimeError("freeze_bn left no trainable parameters")
    lr = float(cfg.get("lr", 5e-4))
    opt = torch.optim.AdamW(opt_params, lr=lr,
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", 200))
    sched_type = str(cfg.get("lr_scheduler", "none")).lower()
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.01)
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(cfg.get("lr_step_size", 80)), gamma=0.5)
    else:
        scheduler = None

    use_amp = bool(cfg.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    log_every = int(cfg.get("log_every", 10))

    print(f"\n{'='*60}")
    print(f"[S2] Experiment: {tag}")
    print(f"[S2] model={model_cls.upper()}, train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}")
    print(f"[S2] topk={topk}, ch={bc}, params={n_params:,}, nv={n_views_cost}")
    print(f"[S2] epochs={epochs}, lr={lr}, sched={sched_type}, amp={use_amp}")
    print(f"[S2] render={cfg.get('render_loss_type','charb')}, "
          f"cw={cfg.get('center_loss_weight',0.0)}, "
          f"center={cfg.get('center_loss_type','chamfer')}, "
          f"hung_a={cfg.get('hung_alpha_weight',0.0)}, "
          f"vol_w={cfg.get('volume_alpha_weight',1.0)}, "
          f"count_w={cfg.get('count_loss_weight',0.0)}, "
          f"hn_w={cfg.get('hn_loss_weight',0.0)}, "
          f"skip={cfg.get('skip_mode','broadcast')}, "
          f"teacher_w={cfg.get('teacher_loss_weight',0.0)}")
    if teacher_dir:
        print(f"[S2] teacher_dir={teacher_dir}")
    print(f"[S2] Output: {output_dir}")
    print(f"{'='*60}")

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    best_f1 = -1.0
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tm = train_one_epoch(model, renderer, train_loader, opt, cam, cfg, dev,
                             scaler=scaler, teacher_dir=teacher_dir)
        vm = validate_model(model, val_loader, cam, dev)
        cur_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0
        entry = {"epoch": ep, "elapsed": elapsed, "lr": cur_lr}
        entry.update({f"train_{k}": v for k, v in tm.items()})
        entry.update({f"val_{k}": v for k, v in vm.items()})
        history.append(entry)
        if ckpt_select == "last":
            is_best = ep == epochs
        else:
            is_best = vm.get("f1", 0) > best_f1
        if is_best:
            best_f1 = vm.get("f1", 0) if ckpt_select != "last" else vm.get("f1", 0)
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
        if ep % log_every == 0 or ep == 1 or ep == epochs or is_best:
            tl_str = f" tch {tm.get('teacher_loss', 0):.4f}" if teacher_dir else ""
            print(f"  Ep {ep:4d}/{epochs} | loss {tm['loss']:.4f} "
                  f"(rend {tm['render_loss']:.4f} vol {tm['vol_loss']:.4f}"
                  f"{tl_str} ctr {tm['center_loss']:.4f}) | "
                  f"val F1 {vm['f1']:.3f} P {vm['precision']:.3f} "
                  f"R {vm['recall']:.3f} | lr {cur_lr:.1e} | "
                  f"{'*BEST*' if is_best else ''} {elapsed:.1f}s")
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[S2] {tag} done. Best F1={best_f1:.3f} | {total_time:.0f}s")
    return {"best_f1": best_f1, "history": history, "total_time": total_time}


# ============================================================
# Phase 0
# ============================================================

def run_SC0(device):
    print("\n" + "=" * 60)
    print("[SC0] Pipeline Smoke Test")
    print("=" * 60)
    cam = get_cam()
    dev = torch.device(device)
    base_dir = get_base_dir()
    data_root = os.path.join(base_dir, DATASET4_ROOT)
    ds = LF3DGSDataset(data_root, split="train")
    sai_np, centers_np, meta, path = ds[0]
    sai = torch.from_numpy(sai_np).unsqueeze(0).to(dev)
    print(f"  Sample: {os.path.basename(path)}")
    print(f"  SAI shape: {sai.shape}, particles: {centers_np.shape[0]}")

    topk = 1500
    model = LF3DGSNet(out_shape=tuple(cam.grid_shape), topk=topk).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}, topk={topk}")

    B, C, H, W = sai.shape
    V = U = int(math.sqrt(C))
    lf4d = sai.reshape(B, V, U, H, W)
    outputs = model(lf4d, out_hw=(cam.grid_shape[1], cam.grid_shape[2]),
                    cam_params=cam)
    mu = outputs["mu"]
    assert mu.shape == (1, topk, 3), f"mu shape mismatch: {mu.shape}"
    assert outputs["logit_alpha"].shape == (1, topk)
    print("  [PASS] Output shapes correct")

    renderer = GaussianSplatRenderer(cam).to(dev)
    P = cam.px_per_mla
    s_idx, t_idx = renderer.sample_views(4, P, dev)
    rendered = renderer(mu[0], outputs["log_sigma"][0],
                        outputs["logit_alpha"][0], s_idx, t_idx)
    print(f"  Rendered: {rendered.shape}, range "
          f"[{rendered.min().item():.4f}, {rendered.max().item():.4f}]")
    print("  [PASS] Rendering works")

    gt_c = torch.from_numpy(centers_np).to(dev)
    loss_ch = chamfer_distance(mu[0], gt_c)
    flat_idx = renderer.sai_index(s_idx, t_idx, P)
    gt_v = sai[0, flat_idx]
    gm = gt_v.max().detach().clamp(min=1e-6)
    loss_r = charbonnier_loss(rendered / gm, gt_v / gm)
    gt_occ = build_gt_occupancy(gt_c, cam, dev)
    loss_vol = focal_bce_loss(outputs["volume"]["alpha_logits"][0], gt_occ)
    (loss_r + loss_vol + 0.5 * loss_ch).backward()
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    print(f"  Loss: render={loss_r.item():.4f}, vol_alpha={loss_vol.item():.4f}, "
          f"chamfer={loss_ch.item():.4f}")
    assert not has_nan, "NaN in gradients!"
    print("  [PASS] Backward clean")
    print("\n[SC0] ALL CHECKS PASSED")
    return True


def run_SC1(device):
    base_dir = get_base_dir()
    ds = LF3DGSDataset(os.path.join(base_dir, DATASET4_ROOT), split="train")
    target_idx = 0
    for i in range(len(ds)):
        _, _, meta, _ = ds[i]
        if (abs(meta.get("ppm_concentration", 0) - 0.025) < 0.005
                and abs(meta.get("particle_diameter_um", 0) - 50) < 5):
            target_idx = i
            break
    sub = Subset(ds, [target_idx])
    tl, vl = make_loaders(sub, sub)
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 1500, "lr": 1e-3, "epochs": 500,
                "center_loss_weight": 0.0,
                "volume_alpha_weight": 1.0,
                "render_loss_weight": 1.0,
                "views_per_step": 32,
                "log_every": 25})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "SC1_v2"),
                        device, tag="SC1: single-sample overfit v2")


def run_SC2(device):
    base_dir = get_base_dir()
    ds = LF3DGSDataset(os.path.join(base_dir, DATASET4_ROOT), split="train")
    sub = filter_dataset_by_conc(ds, 0.025)
    n = min(5, len(sub))
    sub5 = Subset(sub, list(range(n)))
    tl, vl = make_loaders(sub5, sub5)
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 1500, "lr": 1e-3, "epochs": 300,
                "center_loss_weight": 0.0,
                "volume_alpha_weight": 1.0,
                "render_loss_weight": 1.0,
                "views_per_step": 32,
                "log_every": 15})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "SC2_v2"),
                        device, tag="SC2: 5-sample overfit v2")


# ============================================================
# Phase 1
# ============================================================

def run_B1(device):
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    train_ds = LF3DGSDataset(dr, split="train")
    val_ds = LF3DGSDataset(dr, split="val")
    ts = filter_dataset_by_conc(train_ds, 0.025)
    vs = filter_dataset_by_conc(val_ds, 0.025)
    if len(vs) == 0:
        vs = val_ds
    tl, vl = make_loaders(ts, vs)
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 1500, "epochs": 200})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "B1_single_conc_0p025"),
                        device, tag="B1: 0.025ppm baseline")


def run_B2(device):
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    tl, vl = make_loaders(LF3DGSDataset(dr, split="train"),
                          LF3DGSDataset(dr, split="val"))
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 3500, "epochs": 200})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "B2_full_dataset4"),
                        device, tag="B2: full dataset4 baseline")


def run_B3(device):
    print("\n" + "=" * 60)
    print("[B3] Evaluate v3 model on dataset4 test set")
    print("=" * 60)
    base_dir = get_base_dir()
    cam = get_cam()
    dev = torch.device(device)
    mp = os.path.join(base_dir, "output/lf3dgs_v3_stage2/best.pt")
    if not os.path.exists(mp):
        print(f"  [SKIP] v3 model not found: {mp}")
        return None
    dr = os.path.join(base_dir, DATASET4_ROOT)
    test_loader = DataLoader(
        LF3DGSDataset(dr, split="test"), batch_size=1,
        shuffle=False, collate_fn=collate_fn, num_workers=0)
    model = LF3DGSNet(out_shape=tuple(cam.grid_shape),
                       base_channels=32, topk=500).to(dev)
    try:
        model.load_state_dict(
            torch.load(mp, map_location=dev, weights_only=True))
    except Exception as e:
        print(f"  [SKIP] Failed to load: {e}")
        return None
    metrics = validate_model(model, test_loader, cam, dev)
    print(f"  v3: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} "
          f"R={metrics['recall']:.3f} dist={metrics['mean_dist']:.2f}")
    od = os.path.join(base_dir, OUTPUT_ROOT, "B3_v3_eval")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ============================================================
# Phase 1.5: Cost Volume Architecture Experiments
# ============================================================

def run_CV0(device):
    """Smoke test for LF3DGSNetV2 (cost volume model)."""
    print("\n" + "=" * 60)
    print("[CV0] Cost Volume Pipeline Smoke Test")
    print("=" * 60)
    cam = get_cam()
    dev = torch.device(device)
    base_dir = get_base_dir()
    data_root = os.path.join(base_dir, DATASET4_ROOT)
    ds = LF3DGSDataset(data_root, split="train")
    sai_np, centers_np, meta, path = ds[0]
    sai = torch.from_numpy(sai_np).unsqueeze(0).to(dev)
    print(f"  Sample: {os.path.basename(path)}")
    print(f"  SAI shape: {sai.shape}, particles: {centers_np.shape[0]}")

    topk = 1500
    model = LF3DGSNetV2(out_shape=tuple(cam.grid_shape), topk=topk,
                         n_views_cost=16).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  V2 params: {n_params:,}, topk={topk}")

    B, C, H, W = sai.shape
    V = U = int(math.sqrt(C))
    lf4d = sai.reshape(B, V, U, H, W)
    outputs = model(lf4d, out_hw=(cam.grid_shape[1], cam.grid_shape[2]),
                    cam_params=cam)
    mu = outputs["mu"]
    assert mu.shape == (1, topk, 3), f"mu shape mismatch: {mu.shape}"
    assert outputs["logit_alpha"].shape == (1, topk)
    print("  [PASS] Output shapes correct")

    renderer = GaussianSplatRenderer(cam).to(dev)
    P = cam.px_per_mla
    s_idx, t_idx = renderer.sample_views(4, P, dev)
    rendered = renderer(mu[0], outputs["log_sigma"][0],
                        outputs["logit_alpha"][0], s_idx, t_idx)
    print(f"  Rendered: {rendered.shape}, range "
          f"[{rendered.min().item():.4f}, {rendered.max().item():.4f}]")
    print("  [PASS] Rendering works")

    gt_c = torch.from_numpy(centers_np).to(dev)
    loss_ch = chamfer_distance(mu[0], gt_c)
    flat_idx = renderer.sai_index(s_idx, t_idx, P)
    gt_v = sai[0, flat_idx]
    gm = gt_v.max().detach().clamp(min=1e-6)
    loss_r = charbonnier_loss(rendered / gm, gt_v / gm)
    gt_occ = build_gt_occupancy(gt_c, cam, dev)
    loss_vol = focal_bce_loss(outputs["volume"]["alpha_logits"][0], gt_occ)
    (loss_r + loss_vol + 0.5 * loss_ch).backward()
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    print(f"  Loss: render={loss_r.item():.4f}, vol_alpha={loss_vol.item():.4f}, "
          f"chamfer={loss_ch.item():.4f}")
    assert not has_nan, "NaN in gradients!"
    print("  [PASS] Backward clean")

    peak_mb = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"  Peak GPU mem: {peak_mb:.0f} MB")
    print("\n[CV0] ALL CHECKS PASSED")
    return True


def run_CV2(device):
    """Single-sample overfit test for V2 (cost volume). Compares against V1's SC1."""
    base_dir = get_base_dir()
    ds = LF3DGSDataset(os.path.join(base_dir, DATASET4_ROOT), split="train")
    target_idx = 0
    for i in range(len(ds)):
        _, _, meta, _ = ds[i]
        if (abs(meta.get("ppm_concentration", 0) - 0.025) < 0.005
                and abs(meta.get("particle_diameter_um", 0) - 50) < 5):
            target_idx = i
            break
    sub = Subset(ds, [target_idx])
    tl, vl = make_loaders(sub, sub)
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 1500, "lr": 1e-3, "epochs": 500,
                "center_loss_weight": 0.0,
                "volume_alpha_weight": 1.0,
                "render_loss_weight": 1.0,
                "views_per_step": 32,
                "n_views_cost": 16,
                "log_every": 25})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "CV2_single_overfit"),
                        device, tag="CV2: V2 single-sample overfit",
                        model_cls="v2")


def run_CV3_B1(device):
    """Cost Volume baseline on single-concentration (same conditions as B1)."""
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    train_ds = LF3DGSDataset(dr, split="train")
    val_ds = LF3DGSDataset(dr, split="val")
    ts = filter_dataset_by_conc(train_ds, 0.025)
    vs = filter_dataset_by_conc(val_ds, 0.025)
    if len(vs) == 0:
        vs = val_ds
    tl, vl = make_loaders(ts, vs)
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 1500, "epochs": 200, "n_views_cost": 16})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "CV3_B1_cost_vol"),
                        device, tag="CV3-B1: V2 single-conc baseline",
                        model_cls="v2")


def run_CV3_B2(device):
    """Cost Volume baseline on full dataset (same conditions as B2)."""
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    tl, vl = make_loaders(LF3DGSDataset(dr, split="train"),
                          LF3DGSDataset(dr, split="val"))
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({"topk": 3500, "epochs": 200, "n_views_cost": 16})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "CV3_B2_cost_vol"),
                        device, tag="CV3-B2: V2 full-dataset baseline",
                        model_cls="v2")


# ============================================================
# Phase 2
# ============================================================

def _run_sweep_fast(experiments, device, sweep_name, model_cls="v2"):
    """Fast sweep on single-concentration 0.025ppm subset (22 train, 3 val)."""
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    train_ds = LF3DGSDataset(dr, split="train")
    val_ds = LF3DGSDataset(dr, split="val")
    ts = filter_dataset_by_conc(train_ds, 0.025)
    vs = filter_dataset_by_conc(val_ds, 0.025)
    if len(vs) == 0:
        vs = val_ds
    tl, vl = make_loaders(ts, vs)
    results = {}
    for exp in experiments:
        name = exp["name"]
        cfg = copy.deepcopy(BASE_S2_CFG)
        cfg.update(exp.get("overrides", {}))
        mcls = exp.get("model_cls", model_cls)
        od = os.path.join(base_dir, OUTPUT_ROOT, sweep_name, name)
        res = run_training(cfg, tl, vl, od, device,
                           tag=f"{sweep_name}/{name}", model_cls=mcls)
        results[name] = {"best_f1": res["best_f1"],
                         "total_time": res["total_time"]}
    sd = os.path.join(base_dir, OUTPUT_ROOT, sweep_name)
    with open(os.path.join(sd, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{sweep_name}] Sweep done.")
    for n, r in results.items():
        print(f"  {n}: F1={r['best_f1']:.3f}")
    return results


def _run_sweep(experiments, device, sweep_name, model_cls="v2"):
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    tl, vl = make_loaders(LF3DGSDataset(dr, split="train"),
                          LF3DGSDataset(dr, split="val"))
    results = {}
    for exp in experiments:
        name = exp["name"]
        cfg = copy.deepcopy(BASE_S2_CFG)
        cfg.update(exp.get("overrides", {}))
        mcls = exp.get("model_cls", model_cls)
        od = os.path.join(base_dir, OUTPUT_ROOT, sweep_name, name)
        res = run_training(cfg, tl, vl, od, device,
                           tag=f"{sweep_name}/{name}", model_cls=mcls)
        results[name] = {"best_f1": res["best_f1"],
                         "total_time": res["total_time"]}
    sd = os.path.join(base_dir, OUTPUT_ROOT, sweep_name)
    with open(os.path.join(sd, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{sweep_name}] Sweep done.")
    for n, r in results.items():
        print(f"  {n}: F1={r['best_f1']:.3f}")
    return results


def run_H1(device):
    return _run_sweep_fast([
        {"name": f"topk_{k}", "overrides": {"topk": k, "epochs": 100,
                                             "n_views_cost": 16}}
        for k in [500, 1000, 1500, 2500, 3500]
    ], device, "H1_topk")


def run_H2(device):
    exps = []
    base = {"n_views_cost": 16, "epochs": 100}
    for cw in [0.0, 0.1, 0.5, 1.0]:
        exps.append({"name": f"chamfer_cw{cw}",
                     "overrides": {**base, "center_loss_weight": cw,
                                   "center_loss_type": "chamfer"}})
    for rt in ["l1", "mse"]:
        exps.append({"name": f"render_{rt}",
                     "overrides": {**base, "render_loss_type": rt}})
    return _run_sweep_fast(exps, device, "H2_loss")


def run_H3(device):
    exps = []
    for ch in [16, 32, 64]:
        exps.append({"name": f"v2_ch{ch}",
                     "overrides": {"base_channels": ch, "epochs": 100,
                                   "n_views_cost": 16},
                     "model_cls": "v2"})
    exps.append({"name": "v1_ch32",
                 "overrides": {"base_channels": 32, "epochs": 100},
                 "model_cls": "v1"})
    exps.append({"name": "v1_ch64",
                 "overrides": {"base_channels": 64, "epochs": 100},
                 "model_cls": "v1"})
    return _run_sweep_fast(exps, device, "H3_capacity")


def run_H4(device):
    exps = []
    base = {"n_views_cost": 16, "epochs": 100}
    for lv in [1e-4, 5e-4, 1e-3, 2e-3]:
        exps.append({"name": f"lr_{lv}",
                     "overrides": {**base, "lr": lv}})
    exps.append({"name": "cosine_sched",
                 "overrides": {**base, "lr_scheduler": "cosine"}})
    for vpk in [8, 16, 32]:
        exps.append({"name": f"views_{vpk}",
                     "overrides": {**base, "views_per_step": vpk}})
    return _run_sweep_fast(exps, device, "H4_training")


def run_H5(device):
    """View count sweep for cost volume construction (V2 only)."""
    return _run_sweep_fast([
        {"name": f"nv_{nv}", "overrides": {"topk": 1500, "epochs": 100,
                                            "n_views_cost": nv}}
        for nv in [4, 8, 16, 32, 64]
    ], device, "H5_nviews")


# ============================================================
# Phase A: V3 Architecture Experiments
# ============================================================

def _get_v3_base_cfg():
    """Best config from Phase 2 sweeps, adapted for V3."""
    cfg = copy.deepcopy(BASE_S2_CFG)
    cfg.update({
        "topk": 1500,
        "n_views_cost": 32,
        "lr": 5e-4,
        "lr_scheduler": "cosine",
        "render_loss_type": "charbonnier",
        "center_loss_weight": 0.1,
        "center_loss_type": "chamfer",
        "volume_alpha_weight": 1.0,
        "teacher_loss_weight": 0.0,
        "use_amp": False,
        "encode_native": True,
        "extract_mode": "peak",
        "bound_offset": True,
        "view_sample": "uniform",
        "occupancy_label": "gaussian",
        "occupancy_sigma": 1.0,
        "volume_loss_type": "centernet",
        "offset_loss_weight": 1.0,
    })
    return cfg


def run_V3_A2(device):
    """V3 single-sample overfit test. Target: F1 > 0.5 (V2 scored 0.329)."""
    base_dir = get_base_dir()
    ds = LF3DGSDataset(os.path.join(base_dir, DATASET4_ROOT), split="train")
    target_idx = 0
    for i in range(len(ds)):
        _, _, meta, _ = ds[i]
        if (abs(meta.get("ppm_concentration", 0) - 0.025) < 0.005
                and abs(meta.get("particle_diameter_um", 0) - 50) < 5):
            target_idx = i
            break
    sub = Subset(ds, [target_idx])
    tl, vl = make_loaders(sub, sub)
    cfg = _get_v3_base_cfg()
    cfg.update({"epochs": 500, "lr": 1e-3, "views_per_step": 32,
                "n_views_cost": 32, "log_every": 25})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "V3_A2_single_overfit"),
                        device, tag="V3-A2: single-sample overfit", model_cls="v3")


def run_V3_A3(device):
    """V3 baseline on single-concentration (same as CV3_B1). Target F1 > 0.40."""
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    train_ds = LF3DGSDataset(dr, split="train")
    val_ds = LF3DGSDataset(dr, split="val")
    ts = filter_dataset_by_conc(train_ds, 0.025)
    vs = filter_dataset_by_conc(val_ds, 0.025)
    if len(vs) == 0:
        vs = val_ds
    tl, vl = make_loaders(ts, vs)
    cfg = _get_v3_base_cfg()
    cfg.update({"epochs": 200})
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "V3_A3_baseline"),
                        device, tag="V3-A3: baseline (22 samples)", model_cls="v3")


# ============================================================
# Phase B: Large Dataset + Teacher Supervision
# ============================================================

def run_V3_B4(device):
    """V3 + dataset6 (240 train) + Stage-1 teacher supervision. Target F1 > 0.55."""
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)

    if not os.path.isdir(ds6_root):
        print(f"[V3_B4] dataset6 not found at {ds6_root}. Run B1 first.")
        return {"best_f1": 0.0}

    train_ds = LF3DGSDataset(ds6_root, split="train")
    val_ds = LF3DGSDataset(ds6_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    use_teacher = os.path.isdir(teacher_dir)
    cfg = _get_v3_base_cfg()
    cfg.update({
        "epochs": 300,
        "n_views_cost": 32,
        "lr_scheduler": "cosine",
        "teacher_loss_weight": 1.0 if use_teacher else 0.0,
        "log_every": 20,
    })
    print(f"[V3_B4] teacher_dir={'found' if use_teacher else 'NOT FOUND'}")
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "V3_B4_large_teacher"),
                        device, tag="V3-B4: dataset6 + teacher",
                        model_cls="v3",
                        teacher_dir=teacher_dir if use_teacher else None)


# ============================================================
# Phase C: Quick Sweep + Final Training
# ============================================================

def run_V3_sweep(device):
    """Quick hyperparameter sweep for V3 on dataset4 single-conc subset."""
    base_dir = get_base_dir()
    dr = os.path.join(base_dir, DATASET4_ROOT)
    train_ds = LF3DGSDataset(dr, split="train")
    val_ds = LF3DGSDataset(dr, split="val")
    ts = filter_dataset_by_conc(train_ds, 0.025)
    vs = filter_dataset_by_conc(val_ds, 0.025)
    if len(vs) == 0:
        vs = val_ds
    tl, vl = make_loaders(ts, vs)

    base = _get_v3_base_cfg()
    base["epochs"] = 100

    exps = []
    for topk in [1000, 1500, 2500]:
        exps.append({"name": f"topk_{topk}",
                     "overrides": {**base, "topk": topk}})
    for nv in [8, 16, 32]:
        exps.append({"name": f"nv_{nv}",
                     "overrides": {**base, "n_views_cost": nv}})

    results = {}
    for exp in exps:
        name = exp["name"]
        cfg = copy.deepcopy(BASE_S2_CFG)
        cfg.update(exp["overrides"])
        od = os.path.join(base_dir, OUTPUT_ROOT, "V3_sweep", name)
        res = run_training(cfg, tl, vl, od, device,
                           tag=f"V3_sweep/{name}", model_cls="v3")
        results[name] = {"best_f1": res["best_f1"], "total_time": res["total_time"]}
    sd = os.path.join(base_dir, OUTPUT_ROOT, "V3_sweep")
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[V3_sweep] done:")
    for n, r in results.items():
        print(f"  {n}: F1={r['best_f1']:.3f}")
    return results


def run_V3_final(device):
    """Final V3 training with best config from C1 sweep."""
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)

    if not os.path.isdir(ds6_root):
        print(f"[V3_final] dataset6 not found, falling back to dataset4")
        dr = os.path.join(base_dir, DATASET4_ROOT)
        train_ds = LF3DGSDataset(dr, split="train")
        val_ds = LF3DGSDataset(dr, split="val")
        ts = filter_dataset_by_conc(train_ds, 0.025)
        vs = filter_dataset_by_conc(val_ds, 0.025)
        if len(vs) == 0:
            vs = val_ds
        tl, vl = make_loaders(ts, vs)
        t_dir = None
    else:
        train_ds = LF3DGSDataset(ds6_root, split="train")
        val_ds = LF3DGSDataset(ds6_root, split="val")
        tl, vl = make_loaders(train_ds, val_ds)
        t_dir = teacher_dir if os.path.isdir(teacher_dir) else None

    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 1500,
        "n_views_cost": 32,
        "lr_scheduler": "cosine",
        "teacher_loss_weight": 1.0 if t_dir else 0.0,
        "epochs": 400,
        "log_every": 20,
    })
    return run_training(cfg, tl, vl,
                        os.path.join(base_dir, OUTPUT_ROOT, "V3_final"),
                        device, tag="V3-final: full training",
                        model_cls="v3", teacher_dir=t_dir)


# ============================================================
# CLI
# ============================================================

def run_V3_eval_test(device, ckpt_path=None):
    """Evaluate best V3 model on dataset6 test set. Saves results to JSON."""
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    dev = torch.device(device)

    if ckpt_path is None:
        # Try V3_final first, fall back to V3_B4
        for candidate in ["V3_final", "V3_B4_large_teacher"]:
            ckpt_path = os.path.join(base_dir, OUTPUT_ROOT, candidate, "best.pt")
            if os.path.exists(ckpt_path):
                print(f"[eval_test] Loading checkpoint: {ckpt_path}")
                break

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")

    cam = get_cam()
    topk = 1500
    model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                        base_channels=32, topk=topk, n_views_cost=32).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    test_ds = LF3DGSDataset(ds6_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    per_sample = []
    with torch.no_grad():
        for sai_batch, center_list, meta_list, fp_list in test_loader:
            sai_batch = sai_batch.to(dev)
            B = sai_batch.shape[0]
            C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
            V = U = int(math.sqrt(C))
            lf4d = sai_batch.reshape(B, V, U, H, W)
            out_hw = (cam.grid_shape[1], cam.grid_shape[2])
            outputs = model(lf4d, out_hw=out_hw, cam_params=cam)
            for i in range(B):
                m = evaluate_gaussians(outputs["mu"][i], outputs["logit_alpha"][i],
                                       center_list[i], cam)
                try:
                    sname = os.path.basename(str(fp_list[i]))
                except Exception:
                    sname = f"sample_{len(per_sample)}"
                m["sample"] = sname
                per_sample.append(m)

    agg = {k: float(np.mean([m[k] for m in per_sample]))
           for k in ("precision", "recall", "f1", "mean_dist")}
    print(f"\n[V3 Test Set Evaluation]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Samples: {len(per_sample)}")
    print(f"  Precision: {agg['precision']:.4f}")
    print(f"  Recall:    {agg['recall']:.4f}")
    print(f"  F1:        {agg['f1']:.4f}")
    print(f"  Mean dist: {agg['mean_dist']:.4f}")

    result = {"checkpoint": ckpt_path, "n_samples": len(per_sample),
              "aggregate": agg, "per_sample": per_sample}
    save_path = os.path.join(os.path.dirname(ckpt_path), "test_eval.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved: {save_path}")
    return result


# ============================================================
# Phase D: Precision Improvement
# ============================================================

def _load_v3_best_checkpoint(base_dir, device, topk=1500, n_views_cost=32,
                              base_channels=32):
    """Load the best V3 checkpoint (V3_final or V3_B4 fallback)."""
    cam = get_cam()
    dev = torch.device(device)
    model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                        base_channels=base_channels, topk=topk,
                        n_views_cost=n_views_cost).to(dev)
    for candidate in ["V3_final", "V3_B4_large_teacher"]:
        ckpt_path = os.path.join(base_dir, OUTPUT_ROOT, candidate, "best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=dev)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state)
            print(f"[D1] Loaded checkpoint: {ckpt_path}")
            return model, cam, dev, ckpt_path
    raise FileNotFoundError(
        "No V3 checkpoint found. Run V3_final or V3_B4 first.")


def run_D1_threshold_sweep(device):
    """D1: Inference-time alpha threshold + NMS sweep on test set.

    Zero training cost — sweeps evaluate the existing best.pt checkpoint under
    all combinations of alpha_threshold x {use_nms, nms_radius_voxel}.
    Saves a full JSON grid and prints the configuration that maximises F1 while
    keeping Recall >= 0.90.
    """
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    if not os.path.isdir(ds6_root):
        print("[D1] dataset6 not found. Run V3_B4 first.")
        return {}

    model, cam, dev, ckpt_path = _load_v3_best_checkpoint(base_dir, device)
    model.eval()

    test_ds = LF3DGSDataset(ds6_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    alpha_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    nms_configs = [
        {"use_nms": False, "nms_radius_voxel": 1.5},
        {"use_nms": True,  "nms_radius_voxel": 1.0},
        {"use_nms": True,  "nms_radius_voxel": 1.5},
        {"use_nms": True,  "nms_radius_voxel": 2.0},
    ]

    # Collect raw model outputs once
    all_outputs = []
    with torch.no_grad():
        for sai_batch, center_list, meta_list, fp_list in test_loader:
            sai_batch = sai_batch.to(dev)
            B = sai_batch.shape[0]
            C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
            V = U = int(math.sqrt(C))
            lf4d = sai_batch.reshape(B, V, U, H, W)
            out_hw = (cam.grid_shape[1], cam.grid_shape[2])
            outputs = model(lf4d, out_hw=out_hw, cam_params=cam)
            for i in range(B):
                all_outputs.append({
                    "mu": outputs["mu"][i].cpu(),
                    "logit_alpha": outputs["logit_alpha"][i].cpu(),
                    "gt_centers": center_list[i].cpu(),
                })

    results = {}
    print(f"\n{'='*70}")
    print("[D1] Threshold + NMS Sweep on Test Set")
    print(f"{'='*70}")
    print(f"{'alpha_thr':>10} {'use_nms':>8} {'nms_r':>6} | "
          f"{'P':>7} {'R':>7} {'F1':>7} {'dist':>7}")
    print("-" * 70)

    best_f1 = -1.0
    best_cfg = {}          # max F1 overall (no recall constraint)
    best_cfg_hr = {}       # max F1 with Recall >= 0.90
    best_f1_hr = -1.0
    for nms_cfg in nms_configs:
        for alpha_thr in alpha_thresholds:
            key = f"thr{alpha_thr}_nms{nms_cfg['use_nms']}_r{nms_cfg['nms_radius_voxel']}"
            per_m = []
            for item in all_outputs:
                m = evaluate_gaussians(
                    item["mu"], item["logit_alpha"], item["gt_centers"], cam,
                    alpha_threshold=alpha_thr,
                    use_nms=nms_cfg["use_nms"],
                    nms_radius_voxel=nms_cfg["nms_radius_voxel"],
                )
                per_m.append(m)
            agg = {k: float(np.mean([m[k] for m in per_m]))
                   for k in ("precision", "recall", "f1", "mean_dist")}
            results[key] = {"alpha_threshold": alpha_thr, **nms_cfg, **agg}
            print(f"  {alpha_thr:>8.2f} {str(nms_cfg['use_nms']):>8} "
                  f"{nms_cfg['nms_radius_voxel']:>6.1f} | "
                  f"{agg['precision']:>7.4f} {agg['recall']:>7.4f} "
                  f"{agg['f1']:>7.4f} {agg['mean_dist']:>7.4f}")
            if agg["f1"] > best_f1:
                best_f1 = agg["f1"]
                best_cfg = {"alpha_threshold": alpha_thr, **nms_cfg, **agg}
            if agg["f1"] > best_f1_hr and agg["recall"] >= 0.90:
                best_f1_hr = agg["f1"]
                best_cfg_hr = {"alpha_threshold": alpha_thr, **nms_cfg, **agg}

    print(f"\n[D1] Best config (max F1, no recall constraint):")
    print(f"  alpha_threshold={best_cfg.get('alpha_threshold')}, "
          f"use_nms={best_cfg.get('use_nms')}, "
          f"nms_radius={best_cfg.get('nms_radius_voxel')}")
    print(f"  => P={best_cfg.get('precision',0):.4f} "
          f"R={best_cfg.get('recall',0):.4f} "
          f"F1={best_cfg.get('f1',0):.4f}")
    if best_cfg_hr:
        print(f"\n[D1] Best config (max F1 with Recall>=0.90):")
        print(f"  alpha_threshold={best_cfg_hr.get('alpha_threshold')}, "
              f"use_nms={best_cfg_hr.get('use_nms')}, "
              f"nms_radius={best_cfg_hr.get('nms_radius_voxel')}")
        print(f"  => P={best_cfg_hr.get('precision',0):.4f} "
              f"R={best_cfg_hr.get('recall',0):.4f} "
              f"F1={best_cfg_hr.get('f1',0):.4f}")
    else:
        print(f"\n[D1] No config achieves Recall >= 0.90 (NMS reduces recall).")

    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D1_threshold_sweep")
    os.makedirs(out_dir, exist_ok=True)
    full_result = {
        "checkpoint": ckpt_path,
        "n_test_samples": len(all_outputs),
        "best_cfg": best_cfg,            # max-F1 config (used by D4)
        "best_cfg_high_recall": best_cfg_hr,  # max-F1 with Recall>=0.90
        "grid": results,
    }
    with open(os.path.join(out_dir, "threshold_sweep.json"), "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"[D1] Results saved: {out_dir}/threshold_sweep.json")
    return full_result


def _run_ds6_sweep(experiments, device, sweep_name, teacher_dir=None):
    """Run a hyperparameter sweep on dataset6 (240 train / 30 val, 100 epochs)."""
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    if not os.path.isdir(ds6_root):
        print(f"[{sweep_name}] dataset6 not found at {ds6_root}.")
        return {}
    train_ds = LF3DGSDataset(ds6_root, split="train")
    val_ds = LF3DGSDataset(ds6_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)
    results = {}
    for exp in experiments:
        name = exp["name"]
        cfg = _get_v3_base_cfg()
        cfg.update(exp.get("overrides", {}))
        od = os.path.join(base_dir, OUTPUT_ROOT, sweep_name, name)
        res = run_training(cfg, tl, vl, od, device,
                           tag=f"{sweep_name}/{name}", model_cls="v3",
                           teacher_dir=teacher_dir)
        results[name] = {"best_f1": res["best_f1"],
                         "total_time": res["total_time"]}
    sd = os.path.join(base_dir, OUTPUT_ROOT, sweep_name)
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{sweep_name}] done:")
    for n, r in results.items():
        print(f"  {n}: F1={r['best_f1']:.3f}")
    return results


def run_D2_gamma_sweep(device):
    """D2: Sweep focal BCE gamma and pos_weight on dataset6 (100 epochs each).

    Sweeps gamma in [1.0, 2.0, 3.0, 5.0] × pos_weight in [20, 50, 100].
    Modifies the volume alpha supervision loss; all other hyperparameters are
    kept at the V3 best config (topk=1500, n_views=32, teacher_w=1.0).
    """
    base_dir = get_base_dir()
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)
    t_dir = teacher_dir if os.path.isdir(teacher_dir) else None
    base_override = {
        "epochs": 100,
        "teacher_loss_weight": 1.0 if t_dir else 0.0,
        "log_every": 10,
    }
    exps = []
    for gamma in [1.0, 2.0, 3.0, 5.0]:
        for pos_w in [20, 50, 100]:
            exps.append({
                "name": f"gamma{gamma}_pw{pos_w}",
                "overrides": {**base_override,
                               "focal_bce_gamma": gamma,
                               "focal_bce_pos_weight": float(pos_w)},
            })
    return _run_ds6_sweep(exps, device, "D2_gamma_sweep", teacher_dir=t_dir)


def run_D3_teacher_sweep(device):
    """D3: Sweep teacher_loss_weight on dataset6 (100 epochs each).

    Tests teacher_loss_weight in [0.3, 0.5, 1.0, 2.0, 3.0] to find the
    optimal balance between teacher distillation and render/GT losses.
    """
    base_dir = get_base_dir()
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)
    t_dir = teacher_dir if os.path.isdir(teacher_dir) else None
    if not t_dir:
        print("[D3] Teacher labels not found. Run generate_teacher_labels.py first.")
    base_override = {"epochs": 100, "log_every": 10}
    exps = [
        {"name": f"tw{tw}",
         "overrides": {**base_override, "teacher_loss_weight": tw}}
        for tw in [0.3, 0.5, 1.0, 2.0, 3.0]
    ]
    return _run_ds6_sweep(exps, device, "D3_teacher_sweep", teacher_dir=t_dir)


def run_D4_full_train(device):
    """D4: Full training with best config from D1-D3 + AMP (400 epochs).

    Reads D1/D2/D3 sweep summaries if available to auto-select best hyperparams;
    falls back to known-good defaults otherwise.  Runs final evaluation on the
    test set and saves results.
    """
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)

    if not os.path.isdir(ds6_root):
        print("[D4] dataset6 not found.")
        return {}

    # --- Auto-read best gamma/pos_weight from D2 sweep ---
    best_gamma = 2.0
    best_pos_w = 50.0
    d2_summary = os.path.join(base_dir, OUTPUT_ROOT,
                               "D2_gamma_sweep", "sweep_summary.json")
    if os.path.exists(d2_summary):
        with open(d2_summary) as f:
            d2_res = json.load(f)
        best_d2_name = max(d2_res, key=lambda k: d2_res[k].get("best_f1", 0))
        parts = best_d2_name.replace("gamma", "").replace("_pw", " ").split()
        try:
            best_gamma = float(parts[0])
            best_pos_w = float(parts[1])
            print(f"[D4] Auto-selected from D2: gamma={best_gamma}, pos_weight={best_pos_w}")
        except Exception:
            pass

    # --- Auto-read best teacher_loss_weight from D3 sweep ---
    best_teacher_w = 1.0
    d3_summary = os.path.join(base_dir, OUTPUT_ROOT,
                               "D3_teacher_sweep", "sweep_summary.json")
    if os.path.exists(d3_summary):
        with open(d3_summary) as f:
            d3_res = json.load(f)
        best_d3_name = max(d3_res, key=lambda k: d3_res[k].get("best_f1", 0))
        try:
            best_teacher_w = float(best_d3_name.replace("tw", ""))
            print(f"[D4] Auto-selected from D3: teacher_loss_weight={best_teacher_w}")
        except Exception:
            pass

    # --- Read best alpha_threshold + NMS config from D1 sweep ---
    best_alpha_thr = 0.05
    best_use_nms = False
    best_nms_radius = 1.5
    d1_result = os.path.join(base_dir, OUTPUT_ROOT,
                              "D1_threshold_sweep", "threshold_sweep.json")
    if os.path.exists(d1_result):
        with open(d1_result) as f:
            d1_data = json.load(f)
        bc = d1_data.get("best_cfg", {})
        best_alpha_thr = bc.get("alpha_threshold", 0.05)
        best_use_nms = bool(bc.get("use_nms", False))
        best_nms_radius = float(bc.get("nms_radius_voxel", 1.5))
        print(f"[D4] Auto-selected from D1: alpha_thr={best_alpha_thr}, "
              f"use_nms={best_use_nms}, nms_r={best_nms_radius}")

    t_dir = teacher_dir if os.path.isdir(teacher_dir) else None
    train_ds = LF3DGSDataset(ds6_root, split="train")
    val_ds = LF3DGSDataset(ds6_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    cfg = _get_v3_base_cfg()
    cfg.update({
        "epochs": 400,
        "teacher_loss_weight": best_teacher_w if t_dir else 0.0,
        "focal_bce_gamma": best_gamma,
        "focal_bce_pos_weight": best_pos_w,
        "use_amp": True,
        "log_every": 20,
    })
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train")
    print(f"[D4] Config: gamma={best_gamma}, pos_w={best_pos_w}, "
          f"teacher_w={cfg['teacher_loss_weight']}, AMP=True, epochs=400")
    result = run_training(cfg, tl, vl, out_dir, device,
                          tag="D4: full train (best config + AMP)",
                          model_cls="v3", teacher_dir=t_dir)

    # Final test set evaluation with best alpha threshold
    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        dev = torch.device(device)
        cam = get_cam()
        model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                             base_channels=32, topk=1500,
                             n_views_cost=32).to(dev)
        state = torch.load(ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state)
        test_ds = LF3DGSDataset(ds6_root, split="test")
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
        test_m = validate_model(model, test_loader, cam, dev,
                                alpha_threshold=best_alpha_thr,
                                use_nms=best_use_nms,
                                nms_radius_voxel=best_nms_radius)
        print(f"\n[D4] Test set (alpha_thr={best_alpha_thr}, "
              f"use_nms={best_use_nms}, nms_r={best_nms_radius}):")
        print(f"  P={test_m['precision']:.4f} R={test_m['recall']:.4f} "
              f"F1={test_m['f1']:.4f} dist={test_m['mean_dist']:.4f}")
        result["test_metrics"] = test_m
        with open(os.path.join(out_dir, "test_eval.json"), "w") as f:
            json.dump({"alpha_threshold": best_alpha_thr,
                       "use_nms": best_use_nms,
                       "nms_radius_voxel": best_nms_radius,
                       **test_m}, f, indent=2)
    return result


def run_D4_tw3(device):
    """D4-v2: Full training with teacher_w=3.0 (D3 best) + AMP, 400 epochs.

    Identical to D4 except teacher_loss_weight is fixed at 3.0 regardless of
    D3 sweep result, allowing a direct ablation of the teacher weight choice.
    """
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER6_ROOT)

    if not os.path.isdir(ds6_root):
        print("[D4_tw3] dataset6 not found.")
        return {}

    # Reuse D1 NMS config
    best_alpha_thr = 0.05
    best_use_nms = True
    best_nms_radius = 1.5
    d1_result = os.path.join(base_dir, OUTPUT_ROOT,
                              "D1_threshold_sweep", "threshold_sweep.json")
    if os.path.exists(d1_result):
        with open(d1_result) as f:
            d1_data = json.load(f)
        bc = d1_data.get("best_cfg", {})
        best_alpha_thr = bc.get("alpha_threshold", 0.05)
        best_use_nms = bool(bc.get("use_nms", True))
        best_nms_radius = float(bc.get("nms_radius_voxel", 1.5))

    t_dir = teacher_dir if os.path.isdir(teacher_dir) else None
    train_ds = LF3DGSDataset(ds6_root, split="train")
    val_ds = LF3DGSDataset(ds6_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    cfg = _get_v3_base_cfg()
    cfg.update({
        "epochs": 400,
        "teacher_loss_weight": 3.0 if t_dir else 0.0,
        "focal_bce_gamma": 2.0,
        "focal_bce_pos_weight": 50.0,
        "use_amp": True,
        "log_every": 20,
    })
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D4_tw3")
    print(f"[D4_tw3] teacher_w=3.0, AMP=True, epochs=400")
    result = run_training(cfg, tl, vl, out_dir, device,
                          tag="D4-tw3: teacher_w=3.0 ablation",
                          model_cls="v3", teacher_dir=t_dir)

    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        dev = torch.device(device)
        cam = get_cam()
        model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                             base_channels=32, topk=1500,
                             n_views_cost=32).to(dev)
        state = torch.load(ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state)
        test_ds = LF3DGSDataset(ds6_root, split="test")
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
        nms_cfgs = [
            dict(use_nms=False, nms_radius_voxel=1.5),
            dict(use_nms=True,  nms_radius_voxel=1.5),
        ]
        nms_results = {}
        for nc in nms_cfgs:
            m = validate_model(model, test_loader, cam, dev,
                               alpha_threshold=best_alpha_thr, **nc)
            label = f"nms={'True' if nc['use_nms'] else 'False'} r={nc['nms_radius_voxel']}"
            nms_results[label] = {k: m[k] for k in ("precision", "recall", "f1", "mean_dist")}
            print(f"  [{label}] P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f}")
        with open(os.path.join(out_dir, "test_nms_eval.json"), "w") as f:
            json.dump({"checkpoint": ckpt, "nms_eval": nms_results}, f, indent=2)
    return result


def run_D4_no_teacher(device):
    """D4-v3: Ablation — no teacher supervision (teacher_w=0), 400 epochs + AMP.

    Quantifies the contribution of Stage-1 teacher labels to the final F1.
    All other hyperparameters are identical to D4.
    """
    base_dir = get_base_dir()
    ds6_root = os.path.join(base_dir, DATASET6_ROOT)

    if not os.path.isdir(ds6_root):
        print("[D4_no_teacher] dataset6 not found.")
        return {}

    best_alpha_thr = 0.05
    best_use_nms = True
    best_nms_radius = 1.5
    d1_result = os.path.join(base_dir, OUTPUT_ROOT,
                              "D1_threshold_sweep", "threshold_sweep.json")
    if os.path.exists(d1_result):
        with open(d1_result) as f:
            d1_data = json.load(f)
        bc = d1_data.get("best_cfg", {})
        best_alpha_thr = bc.get("alpha_threshold", 0.05)
        best_use_nms = bool(bc.get("use_nms", True))
        best_nms_radius = float(bc.get("nms_radius_voxel", 1.5))

    train_ds = LF3DGSDataset(ds6_root, split="train")
    val_ds = LF3DGSDataset(ds6_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    cfg = _get_v3_base_cfg()
    cfg.update({
        "epochs": 400,
        "teacher_loss_weight": 0.0,
        "focal_bce_gamma": 2.0,
        "focal_bce_pos_weight": 50.0,
        "use_amp": True,
        "log_every": 20,
    })
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D4_no_teacher")
    print("[D4_no_teacher] teacher_w=0.0, AMP=True, epochs=400")
    result = run_training(cfg, tl, vl, out_dir, device,
                          tag="D4-no-teacher: ablation (no Stage-1 supervision)",
                          model_cls="v3", teacher_dir=None)

    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        dev = torch.device(device)
        cam = get_cam()
        model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                             base_channels=32, topk=1500,
                             n_views_cost=32).to(dev)
        state = torch.load(ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state)
        test_ds = LF3DGSDataset(ds6_root, split="test")
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
        nms_cfgs = [
            dict(use_nms=False, nms_radius_voxel=1.5),
            dict(use_nms=True,  nms_radius_voxel=1.5),
        ]
        nms_results = {}
        for nc in nms_cfgs:
            m = validate_model(model, test_loader, cam, dev,
                               alpha_threshold=best_alpha_thr, **nc)
            label = f"nms={'True' if nc['use_nms'] else 'False'} r={nc['nms_radius_voxel']}"
            nms_results[label] = {k: m[k] for k in ("precision", "recall", "f1", "mean_dist")}
            print(f"  [{label}] P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f}")
        with open(os.path.join(out_dir, "test_nms_eval.json"), "w") as f:
            json.dump({"checkpoint": ckpt, "nms_eval": nms_results}, f, indent=2)
    return result


# ============================================================
# Phase E: Multi-Concentration Generalization
# ============================================================

def run_E3_multiconc(device):
    """E3: Train V3 on multi-concentration dataset7 (0.025/0.05/0.075ppm).

    Requires dataset7 (output/lightfield_dataset7) with 360/45/45 split and
    optionally teacher labels in output/stage1_teachers_ds7.  Uses topk=4000
    to cover the highest concentration (~3234 particles at 0.075ppm).
    """
    base_dir = get_base_dir()
    ds7_root = os.path.join(base_dir, DATASET7_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER7_ROOT)

    if not os.path.isdir(ds7_root):
        print(f"[E3] dataset7 not found at {ds7_root}.")
        print("     Run scripts/prepare_dataset7.sh to generate it first.")
        return {}

    use_teacher = os.path.isdir(teacher_dir)
    train_ds = LF3DGSDataset(ds7_root, split="train")
    val_ds = LF3DGSDataset(ds7_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    # Auto-read best hyperparams from D4 if available
    best_gamma = 2.0
    best_pos_w = 50.0
    best_teacher_w = 1.0
    d4_cfg = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train", "config.json")
    if os.path.exists(d4_cfg):
        with open(d4_cfg) as f:
            d4 = json.load(f)
        best_gamma = float(d4.get("focal_bce_gamma", 2.0))
        best_pos_w = float(d4.get("focal_bce_pos_weight", 50.0))
        best_teacher_w = float(d4.get("teacher_loss_weight", 1.0))
        print(f"[E3] Loaded D4 config: gamma={best_gamma}, pos_w={best_pos_w}, "
              f"teacher_w={best_teacher_w}")

    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 4000,
        "epochs": 400,
        "teacher_loss_weight": best_teacher_w if use_teacher else 0.0,
        "focal_bce_gamma": best_gamma,
        "focal_bce_pos_weight": best_pos_w,
        "use_amp": True,
        "log_every": 20,
    })
    print(f"[E3] teacher={'found' if use_teacher else 'NOT FOUND'}, "
          f"train={len(train_ds)}, val={len(val_ds)}")
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "E3_multiconc")
    return run_training(cfg, tl, vl, out_dir, device,
                        tag="E3: multi-conc (0.025/0.05/0.075ppm)",
                        model_cls="v3",
                        teacher_dir=teacher_dir if use_teacher else None)


def run_E4_eval_conc(device, ckpt_path=None):
    """E4: Per-concentration evaluation of the E3 model on test set.

    Groups test samples by ppm_concentration and reports P/R/F1 per group,
    producing both a console table and a JSON report.
    """
    base_dir = get_base_dir()
    ds7_root = os.path.join(base_dir, DATASET7_ROOT)
    dev = torch.device(device)

    if not os.path.isdir(ds7_root):
        print(f"[E4] dataset7 not found.")
        return {}

    if ckpt_path is None:
        ckpt_path = os.path.join(base_dir, OUTPUT_ROOT, "E3_multiconc", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[E4] Checkpoint not found: {ckpt_path}")
        return {}

    cam = get_cam()
    model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                        base_channels=32, topk=4000, n_views_cost=32).to(dev)
    state = torch.load(ckpt_path, map_location=dev)
    state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    model.eval()

    # Load best alpha threshold from D1 (fallback 0.05)
    alpha_thr = 0.05
    d1_result = os.path.join(base_dir, OUTPUT_ROOT,
                              "D1_threshold_sweep", "threshold_sweep.json")
    if os.path.exists(d1_result):
        with open(d1_result) as f:
            alpha_thr = json.load(f).get("best_cfg", {}).get("alpha_threshold", 0.05)

    test_ds = LF3DGSDataset(ds7_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    per_sample = []
    with torch.no_grad():
        for sai_batch, center_list, meta_list, fp_list in test_loader:
            sai_batch = sai_batch.to(dev)
            B = sai_batch.shape[0]
            C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
            V = U = int(math.sqrt(C))
            lf4d = sai_batch.reshape(B, V, U, H, W)
            out_hw = (cam.grid_shape[1], cam.grid_shape[2])
            outputs = model(lf4d, out_hw=out_hw, cam_params=cam)
            for i in range(B):
                m = evaluate_gaussians(
                    outputs["mu"][i], outputs["logit_alpha"][i],
                    center_list[i], cam, alpha_threshold=alpha_thr)
                m["ppm"] = float(meta_list[i].get("ppm_concentration", -1))
                m["diameter_um"] = float(meta_list[i].get("particle_diameter_um", -1))
                try:
                    m["sample"] = os.path.basename(str(fp_list[i]))
                except Exception:
                    m["sample"] = f"sample_{len(per_sample)}"
                per_sample.append(m)

    # Group by concentration
    conc_groups: Dict[float, List] = {}
    for m in per_sample:
        c = round(m["ppm"], 4)
        conc_groups.setdefault(c, []).append(m)

    print(f"\n[E4] Per-Concentration Evaluation  (alpha_thr={alpha_thr})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'Conc(ppm)':>12} {'N':>5} {'P':>7} {'R':>7} {'F1':>7} {'dist':>7}")
    print("-" * 55)
    per_conc_agg = {}
    for conc in sorted(conc_groups.keys()):
        ms = conc_groups[conc]
        agg = {k: float(np.mean([m[k] for m in ms]))
               for k in ("precision", "recall", "f1", "mean_dist")}
        per_conc_agg[conc] = {"n": len(ms), **agg}
        print(f"  {conc:>10.4f} {len(ms):>5d} "
              f"{agg['precision']:>7.4f} {agg['recall']:>7.4f} "
              f"{agg['f1']:>7.4f} {agg['mean_dist']:>7.4f}")

    overall = {k: float(np.mean([m[k] for m in per_sample]))
               for k in ("precision", "recall", "f1", "mean_dist")}
    print("-" * 55)
    print(f"  {'Overall':>10} {len(per_sample):>5d} "
          f"{overall['precision']:>7.4f} {overall['recall']:>7.4f} "
          f"{overall['f1']:>7.4f} {overall['mean_dist']:>7.4f}")

    result = {
        "checkpoint": ckpt_path,
        "alpha_threshold": alpha_thr,
        "overall": overall,
        "per_concentration": per_conc_agg,
        "per_sample": per_sample,
    }
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "E3_multiconc")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_per_conc.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[E4] Results saved: {out_dir}/eval_per_conc.json")
    return result


# ============================================================
# Phase F: Multi-Particle-Size Generalization
# ============================================================

def run_F3_multisize(device):
    """F3: Train V3 on multi-size dataset8 (30/50/70um particles, 0.025ppm).

    Requires dataset8 (output/lightfield_dataset8) with 360/45/45 split and
    optionally teacher labels in output/stage1_teachers_ds8.  Uses topk=1500
    since concentration is fixed at 0.025ppm (~1078 particles).
    """
    base_dir = get_base_dir()
    ds8_root = os.path.join(base_dir, DATASET8_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER8_ROOT)

    if not os.path.isdir(ds8_root):
        print(f"[F3] dataset8 not found at {ds8_root}.")
        print("     Run scripts/prepare_dataset8.sh to generate it first.")
        return {}

    use_teacher = os.path.isdir(teacher_dir)
    train_ds = LF3DGSDataset(ds8_root, split="train")
    val_ds = LF3DGSDataset(ds8_root, split="val")
    tl, vl = make_loaders(train_ds, val_ds)

    # Auto-read best hyperparams from D4 if available
    best_gamma = 2.0
    best_pos_w = 50.0
    best_teacher_w = 1.0
    d4_cfg = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train", "config.json")
    if os.path.exists(d4_cfg):
        with open(d4_cfg) as f:
            d4 = json.load(f)
        best_gamma = float(d4.get("focal_bce_gamma", 2.0))
        best_pos_w = float(d4.get("focal_bce_pos_weight", 50.0))
        best_teacher_w = float(d4.get("teacher_loss_weight", 1.0))
        print(f"[F3] Loaded D4 config: gamma={best_gamma}, pos_w={best_pos_w}, "
              f"teacher_w={best_teacher_w}")

    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 1500,
        "epochs": 400,
        "teacher_loss_weight": best_teacher_w if use_teacher else 0.0,
        "focal_bce_gamma": best_gamma,
        "focal_bce_pos_weight": best_pos_w,
        "use_amp": True,
        "log_every": 20,
    })
    print(f"[F3] teacher={'found' if use_teacher else 'NOT FOUND'}, "
          f"train={len(train_ds)}, val={len(val_ds)}")
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "F3_multisize")
    return run_training(cfg, tl, vl, out_dir, device,
                        tag="F3: multi-size (30/50/70um)",
                        model_cls="v3",
                        teacher_dir=teacher_dir if use_teacher else None)


def run_F4_eval_size(device, ckpt_path=None):
    """F4: Per-particle-size evaluation of the F3 model on test set.

    Groups test samples by particle_diameter_um and reports P/R/F1 per group.
    """
    base_dir = get_base_dir()
    ds8_root = os.path.join(base_dir, DATASET8_ROOT)
    dev = torch.device(device)

    if not os.path.isdir(ds8_root):
        print(f"[F4] dataset8 not found.")
        return {}

    if ckpt_path is None:
        ckpt_path = os.path.join(base_dir, OUTPUT_ROOT, "F3_multisize", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[F4] Checkpoint not found: {ckpt_path}")
        return {}

    cam = get_cam()
    model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                        base_channels=32, topk=1500, n_views_cost=32).to(dev)
    state = torch.load(ckpt_path, map_location=dev)
    state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    model.eval()

    # Load best alpha threshold from D1 (fallback 0.05)
    alpha_thr = 0.05
    d1_result = os.path.join(base_dir, OUTPUT_ROOT,
                              "D1_threshold_sweep", "threshold_sweep.json")
    if os.path.exists(d1_result):
        with open(d1_result) as f:
            alpha_thr = json.load(f).get("best_cfg", {}).get("alpha_threshold", 0.05)

    test_ds = LF3DGSDataset(ds8_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    per_sample = []
    with torch.no_grad():
        for sai_batch, center_list, meta_list, fp_list in test_loader:
            sai_batch = sai_batch.to(dev)
            B = sai_batch.shape[0]
            C, H, W = sai_batch.shape[1], sai_batch.shape[2], sai_batch.shape[3]
            V = U = int(math.sqrt(C))
            lf4d = sai_batch.reshape(B, V, U, H, W)
            out_hw = (cam.grid_shape[1], cam.grid_shape[2])
            outputs = model(lf4d, out_hw=out_hw, cam_params=cam)
            for i in range(B):
                m = evaluate_gaussians(
                    outputs["mu"][i], outputs["logit_alpha"][i],
                    center_list[i], cam, alpha_threshold=alpha_thr)
                m["ppm"] = float(meta_list[i].get("ppm_concentration", -1))
                m["diameter_um"] = float(meta_list[i].get("particle_diameter_um", -1))
                try:
                    m["sample"] = os.path.basename(str(fp_list[i]))
                except Exception:
                    m["sample"] = f"sample_{len(per_sample)}"
                per_sample.append(m)

    # Group by diameter
    size_groups: Dict[float, List] = {}
    for m in per_sample:
        d = round(m["diameter_um"], 1)
        size_groups.setdefault(d, []).append(m)

    print(f"\n[F4] Per-Particle-Size Evaluation  (alpha_thr={alpha_thr})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'Diam(um)':>10} {'N':>5} {'P':>7} {'R':>7} {'F1':>7} {'dist':>7}")
    print("-" * 55)
    per_size_agg = {}
    for diam in sorted(size_groups.keys()):
        ms = size_groups[diam]
        agg = {k: float(np.mean([m[k] for m in ms]))
               for k in ("precision", "recall", "f1", "mean_dist")}
        per_size_agg[diam] = {"n": len(ms), **agg}
        print(f"  {diam:>8.1f} {len(ms):>5d} "
              f"{agg['precision']:>7.4f} {agg['recall']:>7.4f} "
              f"{agg['f1']:>7.4f} {agg['mean_dist']:>7.4f}")

    overall = {k: float(np.mean([m[k] for m in per_sample]))
               for k in ("precision", "recall", "f1", "mean_dist")}
    print("-" * 55)
    print(f"  {'Overall':>8} {len(per_sample):>5d} "
          f"{overall['precision']:>7.4f} {overall['recall']:>7.4f} "
          f"{overall['f1']:>7.4f} {overall['mean_dist']:>7.4f}")

    result = {
        "checkpoint": ckpt_path,
        "alpha_threshold": alpha_thr,
        "overall": overall,
        "per_size": per_size_agg,
        "per_sample": per_sample,
    }
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "F3_multisize")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_per_size.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[F4] Results saved: {out_dir}/eval_per_size.json")
    return result


# ============================================================
# Plan B: 0.05ppm dedicated D4 model training
# ============================================================

def run_D4_conc005(device):
    """B1: Train a dedicated D4-style model on 0.05ppm data only from dataset7.

    Configuration:
    - Data: dataset7 filtered to ppm_concentration == 0.05 (~2157 particles)
    - topk = 2500 (covers 2157 + 16% margin)
    - Teacher labels: from TEACHER7_ROOT if available
    - Epochs: 400, AMP enabled
    - Best hyperparams auto-read from D4_full_train config.json
    - Checkpoint saved to output/D4_conc005/best.pt
    """
    base_dir = get_base_dir()
    ds7_root = os.path.join(base_dir, DATASET7_ROOT)
    teacher_dir = os.path.join(base_dir, TEACHER7_ROOT)

    if not os.path.isdir(ds7_root):
        print(f"[D4_conc005] dataset7 not found at {ds7_root}. "
              "Generate it first.")
        return {}

    # Filter to 0.05ppm only
    full_train = LF3DGSDataset(ds7_root, split="train")
    full_val   = LF3DGSDataset(ds7_root, split="val")
    train_ds = filter_dataset_by_conc(full_train, 0.05)
    val_ds   = filter_dataset_by_conc(full_val,   0.05)
    print(f"[D4_conc005] 0.05ppm splits: train={len(train_ds)}, val={len(val_ds)}")

    if len(train_ds) == 0:
        print("[D4_conc005] No 0.05ppm samples found.")
        return {}

    tl, vl = make_loaders(train_ds, val_ds)

    # Auto-read best hyperparams from D4 config
    best_gamma    = 2.0
    best_pos_w    = 50.0
    best_teacher_w = 1.0
    d4_cfg_path = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train", "config.json")
    if os.path.exists(d4_cfg_path):
        with open(d4_cfg_path) as f:
            d4 = json.load(f)
        best_gamma     = float(d4.get("focal_bce_gamma",       2.0))
        best_pos_w     = float(d4.get("focal_bce_pos_weight",  50.0))
        best_teacher_w = float(d4.get("teacher_loss_weight",   1.0))
        print(f"[D4_conc005] Loaded D4 config: gamma={best_gamma}, "
              f"pos_w={best_pos_w}, teacher_w={best_teacher_w}")

    use_teacher = os.path.isdir(teacher_dir)
    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk":                  2500,    # covers 2157 + 16% margin
        "epochs":                400,
        "teacher_loss_weight":   best_teacher_w if use_teacher else 0.0,
        "focal_bce_gamma":       best_gamma,
        "focal_bce_pos_weight":  best_pos_w,
        "use_amp":               True,
        "log_every":             20,
    })

    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D4_conc005")
    print(f"[D4_conc005] topk=2500, gamma={best_gamma}, pos_w={best_pos_w}, "
          f"teacher_w={cfg['teacher_loss_weight']}, AMP=True, epochs=400")
    result = run_training(cfg, tl, vl, out_dir, device,
                          tag="D4_conc005: 0.05ppm dedicated (topk=2500, 400ep)",
                          model_cls="v3",
                          teacher_dir=teacher_dir if use_teacher else None)

    # Evaluate on 0.05ppm test samples
    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        dev = torch.device(device)
        cam = get_cam()
        model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                             base_channels=32, topk=2500,
                             n_views_cost=32).to(dev)
        state = torch.load(ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state)
        test_ds_full = LF3DGSDataset(ds7_root, split="test")
        test_ds = filter_dataset_by_conc(test_ds_full, 0.05)
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0)
            test_m = validate_model(model, test_loader, cam, dev,
                                    alpha_threshold=0.05, use_nms=False)
            print(f"\n[D4_conc005] Test (0.05ppm, {len(test_ds)} samples): "
                  f"P={test_m['precision']:.4f} R={test_m['recall']:.4f} "
                  f"F1={test_m['f1']:.4f} dist={test_m['mean_dist']:.4f}")
            result["test_metrics_conc005"] = test_m
            with open(os.path.join(out_dir, "test_eval_conc005.json"), "w") as f:
                json.dump({"concentration": 0.05, **test_m}, f, indent=2)
    return result




# ============================================================
# D4_conc005_ds8: 0.05ppm dedicated D4 model on large dataset8_conc005
# ============================================================

def run_D4_conc005_ds8(device):
    """Train dedicated D4-style model on 0.05ppm data from dataset8_conc005.

    - Data: dataset8_conc005 (3360 train / 420 val / 420 test)
    - Diameters: 20/50/80 um @ 0.05 ppm
    - topk=3500, epochs=400, AMP, no Stage-1 teachers
    - Hyperparams auto-read from D4_full_train config.json
    """
    base_dir = get_base_dir()
    ds_root = os.path.join(base_dir, DATASET8_CONC005_ROOT)

    if not os.path.isdir(ds_root):
        print(f"[D4_conc005_ds8] dataset not found: {ds_root}")
        return {}
    for split in ("train", "val", "test"):
        if not os.path.isdir(os.path.join(ds_root, split)):
            print(f"[D4_conc005_ds8] missing split: {split}")
            return {}

    # Limit to ~720 files per split to match D4_full_train scale (240 files/split)
    # ds8_conc005 has 3 diameters so 720 = 240 x3; each 3-diam group = 1 sample
    train_ds = LF3DGSDataset(ds_root, split="train", limit=720)
    val_ds   = LF3DGSDataset(ds_root, split="val", limit=90)
    print(f"[D4_conc005_ds8] train={len(train_ds)}, val={len(val_ds)} (limited)")
    tl, vl = make_loaders(train_ds, val_ds)

    best_gamma, best_pos_w = 2.0, 50.0
    d4_cfg_path = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train", "config.json")
    if os.path.exists(d4_cfg_path):
        with open(d4_cfg_path) as f:
            d4 = json.load(f)
        best_gamma = float(d4.get("focal_bce_gamma", 2.0))
        best_pos_w = float(d4.get("focal_bce_pos_weight", 50.0))
        print(f"[D4_conc005_ds8] D4 config: gamma={best_gamma}, pos_w={best_pos_w}")

    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 3500, "epochs": 400, "teacher_loss_weight": 0.0,
        "focal_bce_gamma": best_gamma, "focal_bce_pos_weight": best_pos_w,
        "use_amp": True, "log_every": 20,
    })
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D4_conc005_ds8")
    print(f"[D4_conc005_ds8] topk=3500, gamma={best_gamma}, pos_w={best_pos_w}, AMP=True, ep=400")
    result = run_training(cfg, tl, vl, out_dir, device,
                          tag="D4_conc005_ds8: 0.05ppm large dataset (ds8)",
                          model_cls="v3", teacher_dir=None)

    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        dev = torch.device(device)
        cam = get_cam()
        model = LF3DGSNetV3(out_shape=tuple(cam.grid_shape),
                             base_channels=32, topk=3500, n_views_cost=32).to(dev)
        state = torch.load(ckpt, map_location=dev)
        state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state)
        test_ds = LF3DGSDataset(ds_root, split="test")
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0)
            test_m = validate_model(model, test_loader, cam, dev,
                                    alpha_threshold=0.05, use_nms=False)
            print(f"\n[D4_conc005_ds8] Test F1={test_m['f1']:.4f} "
                  f"P={test_m['precision']:.4f} R={test_m['recall']:.4f} "
                  f"dist={test_m['mean_dist']:.4f}")
            result["test_metrics"] = test_m
            with open(os.path.join(out_dir, "test_eval.json"), "w") as f:
                json.dump(test_m, f, indent=2)
    return result

# ============================================================
# D5: PIV domain-gap finetune
# ============================================================

def run_D5_piv_finetune(device):
    """D5: Finetune D4_tw3 on PIV-style data to close the domain gap.

    Pipeline:
      1. Auto-generate PIV finetune dataset if not present
         (200 pairs, splatting render, mixed flow fields).
      2. Load D4_tw3 best.pt as pretrained weights.
      3. Finetune for 100 epochs at lr=5e-5 (cosine decay to 5e-7).
      4. Save to output/lf3dgs_s2/D5_piv_finetune/.

    Target: improve A3 benchmark model-mode RelErrXY from ~1.3 → <0.20.
    """
    import subprocess
    base_dir = get_base_dir()
    piv_ds_root = os.path.join(base_dir, "output", "piv_finetune_ds")
    pretrained = os.path.join(base_dir, OUTPUT_ROOT, "D4_tw3", "best.pt")
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D5_piv_finetune")

    # --- Step 1: generate PIV finetune dataset ---
    if not os.path.isdir(os.path.join(piv_ds_root, "train")):
        print("[D5] Generating PIV finetune dataset (200 pairs, splatting)...")
        cmd = [
            sys.executable, "-m", "lf_3dgs.piv.prepare_piv_finetune_data",
            "--output", piv_ds_root,
            "--n-pairs", "200",
            "--ppm", "0.025",
            "--render-mode", "splatting",
        ]
        ret = subprocess.run(cmd, cwd=base_dir, timeout=3600)
        if ret.returncode != 0:
            print("[D5] Dataset generation FAILED.")
            return {}
        print("[D5] Dataset generated.")
    else:
        print(f"[D5] PIV finetune dataset found at {piv_ds_root}")

    train_ds = LF3DGSDataset(piv_ds_root, split="train")
    val_ds   = LF3DGSDataset(piv_ds_root, split="val")
    print(f"[D5] Finetune splits: train={len(train_ds)}, val={len(val_ds)}")
    if len(train_ds) == 0:
        print("[D5] Empty dataset. Abort.")
        return {}
    tl, vl = make_loaders(train_ds, val_ds)

    # --- Step 2: finetune config ---
    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 1500,
        "epochs": 100,
        "lr": 5e-5,
        "lr_scheduler": "cosine",
        "render_loss_type": "charbonnier",
        "center_loss_weight": 0.5,
        "center_loss_type": "chamfer",
        "volume_alpha_weight": 1.0,
        "teacher_loss_weight": 0.0,
        "use_amp": True,
        "log_every": 10,
    })

    print(f"[D5] Finetuning from {pretrained}, 100 epochs, lr=5e-5")
    result = run_training(
        cfg, tl, vl, out_dir, device,
        tag="D5: PIV domain-gap finetune (from D4_tw3)",
        model_cls="v3",
        teacher_dir=None,
        pretrained_ckpt=pretrained if os.path.exists(pretrained) else None,
    )
    return result


def run_D5_piv_finetune_raytrace(device):
    """D5 (raytrace): Finetune D4_full_train on RAYTRACE-rendered PIV data.

    Identical to run_D5_piv_finetune but the finetune dataset is rendered with
    the physically-accurate ray tracer (matching the D4 training distribution
    and the benchmark_xy evaluation), eliminating the splatting-vs-raytrace
    rendering domain gap that confounded the original D5 run.

    Dataset is expected at output/piv_finetune_ds_raytrace (generate it first
    with: python -m lf_3dgs.piv.prepare_piv_finetune_data
          --output output/piv_finetune_ds_raytrace --render-mode raytrace).
    """
    base_dir = get_base_dir()
    piv_ds_root = os.path.join(base_dir, "output", "piv_finetune_ds_raytrace")
    pretrained = os.path.join(base_dir, OUTPUT_ROOT, "D4_full_train", "best.pt")
    out_dir = os.path.join(base_dir, OUTPUT_ROOT, "D5_piv_finetune_raytrace")

    if not os.path.isdir(os.path.join(piv_ds_root, "train")):
        print(f"[D5-raytrace] Dataset not found at {piv_ds_root}. Generate it first.")
        return {}

    train_ds = LF3DGSDataset(piv_ds_root, split="train")
    val_ds   = LF3DGSDataset(piv_ds_root, split="val")
    print(f"[D5-raytrace] Finetune splits: train={len(train_ds)}, val={len(val_ds)}")
    if len(train_ds) == 0:
        print("[D5-raytrace] Empty dataset. Abort.")
        return {}
    tl, vl = make_loaders(train_ds, val_ds)

    cfg = _get_v3_base_cfg()
    cfg.update({
        "topk": 1500,
        "epochs": 100,
        "lr": 5e-5,
        "lr_scheduler": "cosine",
        "render_loss_type": "charbonnier",
        "center_loss_weight": 0.5,
        "center_loss_type": "chamfer",
        "volume_alpha_weight": 1.0,
        "teacher_loss_weight": 0.0,
        "use_amp": True,
        "log_every": 10,
    })

    print(f"[D5-raytrace] Finetuning from {pretrained}, 100 epochs, lr=5e-5")
    result = run_training(
        cfg, tl, vl, out_dir, device,
        tag="D5-raytrace: PIV domain-gap finetune (raytrace data, from D4_full_train)",
        model_cls="v3",
        teacher_dir=None,
        pretrained_ckpt=pretrained if os.path.exists(pretrained) else None,
    )
    return result


PHASE_MAP = {
    "SC0": run_SC0, "SC1": run_SC1, "SC2": run_SC2,
    "B1": run_B1, "B2": run_B2, "B3": run_B3,
    "CV0": run_CV0, "CV2": run_CV2,
    "CV3_B1": run_CV3_B1, "CV3_B2": run_CV3_B2,
    "H1": run_H1, "H2": run_H2, "H3": run_H3, "H4": run_H4, "H5": run_H5,
    "V3_A2": run_V3_A2, "V3_A3": run_V3_A3,
    "V3_B4": run_V3_B4,
    "V3_sweep": run_V3_sweep, "V3_final": run_V3_final,
    "V3_eval_test": run_V3_eval_test,
    # Phase D: Precision improvement
    "D1_threshold_sweep": run_D1_threshold_sweep,
    "D2_gamma_sweep": run_D2_gamma_sweep,
    "D3_teacher_sweep": run_D3_teacher_sweep,
    "D4_full_train": run_D4_full_train,
    "D4_tw3": run_D4_tw3,
    "D4_no_teacher": run_D4_no_teacher,
    # Phase E: Multi-concentration generalization
    "E3_multiconc": run_E3_multiconc,
    "E4_eval_conc": run_E4_eval_conc,
    # Phase F: Multi-particle-size generalization
    "F3_multisize": run_F3_multisize,
    "F4_eval_size": run_F4_eval_size,
    # PIV plan B: 0.05ppm dedicated model
    "D4_conc005": run_D4_conc005,
    "D4_conc005_ds8": run_D4_conc005_ds8,
    # PIV domain-gap finetune
    "D5_piv_finetune": run_D5_piv_finetune,
    "D5_raytrace": run_D5_piv_finetune_raytrace,
}
GROUPS = {
    "all_sanity": ["SC0", "SC1", "SC2"],
    "all_baseline": ["B1", "B2", "B3"],
    "all_cv": ["CV0", "CV2", "CV3_B1", "CV3_B2"],
    "all_sweep": ["H1", "H2", "H3", "H4", "H5"],
    "all_v3": ["V3_A2", "V3_A3"],
    "phase_D": ["D1_threshold_sweep", "D2_gamma_sweep",
                "D3_teacher_sweep", "D4_full_train",
                "D4_tw3", "D4_no_teacher"],
    "phase_E": ["E3_multiconc", "E4_eval_conc"],
    "phase_F": ["F3_multisize", "F4_eval_size"],
}


def main():
    parser = argparse.ArgumentParser(description="LF-3DGS Stage 2 experiments")
    parser.add_argument("--phase", required=True,
                        help=", ".join(list(PHASE_MAP) + list(GROUPS)))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    phases = GROUPS.get(args.phase, [args.phase])
    for p in phases:
        if p not in PHASE_MAP:
            print(f"Unknown: {p}. Available: {list(PHASE_MAP)}+{list(GROUPS)}")
            sys.exit(1)

    all_results = {}
    for p in phases:
        print(f"\n{'#'*60}\n# Phase: {p}\n{'#'*60}")
        all_results[p] = PHASE_MAP[p](args.device)

    base_dir = get_base_dir()
    sp = os.path.join(base_dir, OUTPUT_ROOT, f"run_{args.phase}_summary.json")
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    ser = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            ser[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
        else:
            ser[k] = str(v)
    with open(sp, "w") as f:
        json.dump(ser, f, indent=2, default=str)
    print(f"\nAll done. Summary: {sp}")


if __name__ == "__main__":
    main()

