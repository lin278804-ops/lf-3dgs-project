"""
Stage 1: Per-sample optimisation of 3D Gaussians from SAI supervision.

Given a single light-field sample (sai_stack), directly fit N 3D Gaussians
via gradient descent so that their differentiable splatting matches the
observed SAI views.

Supports:
  - Random init or GT-guided init (for upper-bound validation)
  - Signal-weighted reconstruction loss
  - Sigma annealing (coarse-to-fine)
  - Three-phase alpha training with entropy regularisation
  - Gaussian pruning with safety floor
  - Intermediate snapshot saving for visualisation

Usage:
    python -m lf_3dgs.optimizer \
        --sample path/to/sample.npz \
        --output output/lf3dgs_opt/ \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from .camera import LFCameraParams
from .renderer import GaussianSplatRenderer
from .losses import total_loss, chamfer_distance
from .metrics import evaluate_gaussians
from .dataset import load_single_sample
from .backproject import backproj_init_centers


# ===================================================================== #
# Gaussian parameter container
# ===================================================================== #

class GaussianParticles(nn.Module):
    """Learnable 3D Gaussian particle set."""

    def __init__(
        self,
        n_gaussians: int,
        cam_params: LFCameraParams,
        init_sigma_2d: float = 2.0,
        init_alpha_logit: float = -2.0,
        init_centers: Optional[torch.Tensor] = None,
        init_noise_std: float = 0.2,
    ) -> None:
        """
        Args:
            n_gaussians: number of Gaussians to create.
            cam_params: camera parameters (used for volume bounds).
            init_sigma_2d: initial 2D rendering sigma in SAI pixels.
            init_alpha_logit: initial logit(alpha).
            init_centers: (M, 3) optional GT centres for guided init.
                If provided, n_gaussians is ignored and M is used.
            init_noise_std: Gaussian noise std (mm) added to init_centers.
        """
        super().__init__()
        xr = cam_params.x_range
        yr = cam_params.y_range
        zr = cam_params.z_range

        if init_centers is not None:
            # GT-guided initialisation
            n_gaussians = init_centers.shape[0]
            mu_init = init_centers.clone().float()
            if init_noise_std > 0:
                mu_init = mu_init + torch.randn_like(mu_init) * init_noise_std
            # Clamp to volume
            mu_init[:, 0].clamp_(xr[0], xr[1])
            mu_init[:, 1].clamp_(yr[0], yr[1])
            mu_init[:, 2].clamp_(zr[0], zr[1])
        else:
            # Random uniform init in the physical volume
            mu_init = torch.zeros(n_gaussians, 3)
            mu_init[:, 0].uniform_(xr[0], xr[1])
            mu_init[:, 1].uniform_(yr[0], yr[1])
            mu_init[:, 2].uniform_(zr[0], zr[1])

        self.mu = nn.Parameter(mu_init)
        self.log_sigma = nn.Parameter(
            torch.full((n_gaussians,), float(np.log(max(init_sigma_2d, 0.1))))
        )
        self.logit_alpha = nn.Parameter(
            torch.full((n_gaussians,), float(init_alpha_logit))
        )

    @property
    def n(self) -> int:
        return self.mu.shape[0]

    def prune(self, alpha_threshold: float = 0.01, min_keep: int = 10) -> int:
        """Remove Gaussians with sigmoid(alpha) < threshold. Returns count removed."""
        with torch.no_grad():
            alpha = torch.sigmoid(self.logit_alpha)
            keep = alpha >= alpha_threshold
            n_keep = keep.sum().item()
            # Safety: never prune below min_keep
            if n_keep < min_keep:
                _, topk_idx = torch.topk(alpha, min(min_keep, self.n))
                keep = torch.zeros_like(alpha, dtype=torch.bool)
                keep[topk_idx] = True
                n_keep = keep.sum().item()
            n_before = self.n
            if keep.all():
                return 0
            self.mu = nn.Parameter(self.mu.data[keep])
            self.log_sigma = nn.Parameter(self.log_sigma.data[keep])
            self.logit_alpha = nn.Parameter(self.logit_alpha.data[keep])
            return n_before - self.n


def _robust_gmax(gt_views: torch.Tensor, quantile: float = 0.999,
                 mode: str = "robust") -> torch.Tensor:
    """鲁棒归一化分母。

    mode="max": 直接用最大值（噪声+learn_bg 推荐）
    mode="robust": 抑制尖峰，同时避免稀疏场景分位数塌缩
    """
    flat = gt_views.reshape(-1)
    if flat.numel() < 10:
        return flat.max().detach().clamp(min=1e-6)
    mx = flat.max().detach().clamp(min=1e-6)
    if mode == "max":
        return mx
    q = torch.quantile(flat, quantile).detach().clamp(min=1e-6)
    g = q if bool(mx > 3.0 * q) else mx
    return torch.maximum(g, mx * 0.5)


def _apply_learnable_bg(rendered: torch.Tensor, bg_logit: Optional[nn.Parameter]):
    """渲染结果叠加可学习背景项 rendered += softplus(bg)。"""
    if bg_logit is None:
        return rendered
    return rendered + torch.nn.functional.softplus(bg_logit)


def _gmax_quantile(cfg: Dict) -> float:
    return float(cfg.get("gmax_quantile", 0.999))


def _gmax_mode(cfg: Dict) -> str:
    return str(cfg.get("gmax_mode", "robust"))

def _sigma_anneal_max(it: int, cfg: Dict) -> Optional[float]:
    """Return the maximum allowed sigma at iteration *it*, or None if disabled."""
    sigma_start = cfg.get("sigma_anneal_start", None)
    sigma_end = cfg.get("sigma_anneal_end", None)
    anneal_begin = int(cfg.get("sigma_anneal_iter_begin", 0))
    anneal_finish = int(cfg.get("sigma_anneal_iter_end", 3000))

    if sigma_start is None or sigma_end is None:
        return None

    sigma_start = float(sigma_start)
    sigma_end = float(sigma_end)

    if it <= anneal_begin:
        return sigma_start
    if it >= anneal_finish:
        return sigma_end
    # Linear interpolation
    t = (it - anneal_begin) / max(anneal_finish - anneal_begin, 1)
    return sigma_start + (sigma_end - sigma_start) * t


# ===================================================================== #
# Optimisation loop
# ===================================================================== #

def optimise_sample(
    sample_path: str,
    output_dir: str,
    cfg: Dict,
    device: str = "cuda",
) -> Dict:
    """
    Run per-sample 3D Gaussian optimisation.

    Returns dict of final metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    sai_stack, gt_centers, meta = load_single_sample(sample_path)
    sai_stack = sai_stack.to(dev)          # (C, H, W)
    gt_centers = gt_centers.to(dev)        # (M, 3)

    n_gt = gt_centers.shape[0]
    print(f"[LF-3DGS] Sample: {os.path.basename(sample_path)}")
    print(f"[LF-3DGS] GT particles: {n_gt}, SAI shape: {tuple(sai_stack.shape)}")

    # --- Camera params ---
    cam = LFCameraParams()
    if cfg.get("real_camera", False):
        # Prefer explicit cfg / sample meta (partner SAI is 230×184).
        rc = cfg.get("real_camera_params", {}) or {}
        H, W = int(sai_stack.shape[-2]), int(sai_stack.shape[-1])
        cam = LFCameraParams(
            pitch=float(rc.get("pitch", 0.100)),
            pixel_pitch=float(rc.get("pixel_pitch", 0.0045)),
            num_mla_x=int(rc.get("num_mla_x", W)),
            num_mla_y=int(rc.get("num_mla_y", H)),
            px_per_mla=int(rc.get("px_per_mla", 22)),
        )
    if "x_range" in meta:
        cam.x_range = meta["x_range"]
    if "y_range" in meta:
        cam.y_range = meta["y_range"]
    if "z_range" in meta:
        cam.z_range = meta["z_range"]
    if "grid_shape" in meta:
        cam.grid_shape = meta["grid_shape"]

    # --- Init Gaussians ---
    s1 = cfg.get("stage1", cfg)
    factor = float(s1.get("n_gaussians_factor", 2.0))
    n_fixed = s1.get("n_gaussians_fixed", None)
    n_init = int(n_fixed) if n_fixed else max(int(n_gt * factor), 50)
    init_sigma = float(s1.get("init_sigma_2d", 2.0))
    init_alpha = float(s1.get("init_alpha_logit", -2.0))

    # GT-guided initialisation
    init_mode = str(s1.get("init_mode", "random"))
    init_noise_std = float(s1.get("init_noise_std", 0.2))

    if init_mode == "gt" and gt_centers.shape[0] > 0:
        particles = GaussianParticles(
            n_init, cam, init_sigma, init_alpha,
            init_centers=gt_centers.cpu(),
            init_noise_std=init_noise_std,
        ).to(dev)
        print(f"[LF-3DGS] Init {particles.n} Gaussians from GT (noise={init_noise_std:.2f}mm)")
    elif init_mode in ("backproj", "hybrid"):
        bp_stride = int(s1.get("backproj_stride", 2))
        bp_thr = float(s1.get("backproj_threshold", 0.20))
        bp_mode = str(s1.get("backproj_mode", "mean"))
        bp_min_dist = int(s1.get("backproj_min_distance", 2))
        bp_centers, bp_volume = backproj_init_centers(
            sai_stack, cam, device=str(dev),
            mode=bp_mode, view_stride=bp_stride,
            peak_threshold=bp_thr,
            peak_min_distance=bp_min_dist,
        )
        if bp_centers.shape[0] > 0:
            if init_mode == "hybrid":
                hybrid_factor = float(s1.get("hybrid_extra_factor", 0.5))
                n_extra = max(1, int(bp_centers.shape[0] * hybrid_factor))
                xr, yr, zr = cam.x_range, cam.y_range, cam.z_range
                extra_mu = torch.zeros(n_extra, 3)
                extra_mu[:, 0].uniform_(xr[0], xr[1])
                extra_mu[:, 1].uniform_(yr[0], yr[1])
                extra_mu[:, 2].uniform_(zr[0], zr[1])
                combined = torch.cat([bp_centers, extra_mu], dim=0)
                particles = GaussianParticles(
                    combined.shape[0], cam, init_sigma, init_alpha,
                    init_centers=combined,
                    init_noise_std=init_noise_std,
                ).to(dev)
                print(f"[LF-3DGS] Hybrid init: {bp_centers.shape[0]} backproj + "
                      f"{n_extra} random = {particles.n} total "
                      f"(factor={hybrid_factor:.2f})")
            else:
                particles = GaussianParticles(
                    bp_centers.shape[0], cam, init_sigma, init_alpha,
                    init_centers=bp_centers,
                    init_noise_std=init_noise_std,
                ).to(dev)
                print(f"[LF-3DGS] Init {particles.n} Gaussians from back-projection "
                      f"(stride={bp_stride}, thr={bp_thr})")
        else:
            particles = GaussianParticles(n_init, cam, init_sigma, init_alpha).to(dev)
            print(f"[LF-3DGS] BackProj found 0 peaks, fallback to random ({n_init})")
        bp_vol_path = os.path.join(output_dir, "backproj_volume.npz")
        np.savez_compressed(bp_vol_path, volume=bp_volume.cpu().numpy())
    else:
        particles = GaussianParticles(n_init, cam, init_sigma, init_alpha).to(dev)
        print(f"[LF-3DGS] Init {n_init} Gaussians randomly (factor={factor:.1f})")

    # 已知平面深度先验（标定板台架 z；不用孔位 XY GT）
    plane_z = s1.get("plane_z", None)
    plane_z_weight = float(s1.get("plane_z_weight", 0.0))
    if plane_z is not None:
        z0 = float(plane_z)
        z_noise = float(s1.get("plane_z_init_noise", 0.05))
        with torch.no_grad():
            particles.mu[:, 2] = z0 + torch.randn(
                particles.n, device=dev) * z_noise
        print(f"[LF-3DGS] Plane-z prior: snap init z→{z0:.3f}±{z_noise} "
              f"(weight={plane_z_weight})")

    renderer = GaussianSplatRenderer(cam).to(dev)

    # --- Optimiser ---
    lr_mu = float(s1.get("lr_mu", 0.01))
    lr_alpha = float(s1.get("lr_alpha", 0.001))
    lr_sigma = float(s1.get("lr_sigma", 0.0005))
    num_iters = int(s1.get("num_iters", 2000))
    K = int(s1.get("views_per_step", 16))
    ds = int(s1.get("render_downsample", 1))
    P = cam.px_per_mla

    # Phase boundaries
    phase_mu_only = s1.get("phase_mu_only", [0, 500])
    phase_mu_alpha = s1.get("phase_mu_alpha", [500, 1000])
    phase_all = s1.get("phase_all", [1000, num_iters])

    # Loss cfg
    loss_cfg = {
        "recon_type": s1.get("recon_type", "charbonnier"),
        "recon_weight": float(s1.get("recon_weight", 1.0)),
        "sparse_weight": float(s1.get("sparse_weight", 0.01)),
        "scale_weight": float(s1.get("scale_weight", 0.001)),
        "entropy_weight": float(s1.get("entropy_weight", 0.0)),
        "signal_weight": float(s1.get("signal_weight", 10.0)),
        "signal_thr": float(s1.get("signal_thr", 0.01)),
        "hungarian_pos_weight": float(s1.get("hungarian_pos_weight", 0.0)),
        "hungarian_alpha_weight": float(s1.get("hungarian_alpha_weight", 0.0)),
        "hungarian_match_radius": float(s1.get("hungarian_match_radius", 5.0)),
    }

    # Pruning
    prune_every = int(s1.get("prune_every", 500))
    prune_after = int(s1.get("prune_after", 500))
    prune_thr = float(s1.get("prune_alpha_threshold", 0.01))

    # NMS at evaluation time
    use_nms = bool(s1.get("use_nms", False))
    nms_radius = float(s1.get("nms_radius_voxel", 1.5))

    log_every = int(s1.get("log_every", 50))
    vis_every = int(s1.get("vis_every", 200))
    snapshot_every = int(s1.get("snapshot_every", 0))  # 0 = disabled

    optimizer = torch.optim.Adam([
        {"params": [particles.mu], "lr": lr_mu},
        {"params": [particles.logit_alpha], "lr": lr_alpha},
        {"params": [particles.log_sigma], "lr": lr_sigma},
    ])

    # 可学习背景项（真实数据 ~12% 直流；cfg.learn_bg=true 启用；
    # real_camera 默认开启，可用 learn_bg=false 显式关闭）
    if "learn_bg" in s1:
        learn_bg = bool(s1.get("learn_bg"))
    else:
        learn_bg = bool(cfg.get("real_camera", False))
    bg_logit = nn.Parameter(torch.tensor(-2.0, device=dev)) if learn_bg else None
    if bg_logit is not None:
        # 用全图中位数初始化背景，贴近真实直流 / 加性偏置
        with torch.no_grad():
            med = float(sai_stack.median().item())
            # softplus(bg) ≈ med → bg ≈ log(exp(med)-1)
            med = max(med, 1e-4)
            bg0 = math.log(math.expm1(med)) if med < 20 else med
            bg_logit.data.fill_(bg0)
        optimizer.add_param_group({"params": [bg_logit], "lr": float(s1.get("lr_bg", 0.01))})
        print(f"[LF-3DGS] Learnable background term enabled "
              f"(init softplus(bg)≈{med:.4f})")

    # --- Training loop ---
    history = []
    snapshots_dir = os.path.join(output_dir, "snapshots")
    if snapshot_every > 0:
        os.makedirs(snapshots_dir, exist_ok=True)

    t0 = time.time()

    for it in range(num_iters):
        # Freeze schedule
        particles.logit_alpha.requires_grad_(it >= phase_mu_alpha[0])
        particles.log_sigma.requires_grad_(it >= phase_all[0])

        # --- Sigma annealing ---
        sigma_max = _sigma_anneal_max(it, s1)
        if sigma_max is not None:
            with torch.no_grad():
                max_log_sigma = math.log(max(sigma_max, 0.1))
                particles.log_sigma.data.clamp_(max=max_log_sigma)

        # Sample views
        s_idx, t_idx = renderer.sample_views(K, P, dev)
        flat_idx = renderer.sai_index(s_idx, t_idx, P)

        # Render
        rendered = renderer(
            particles.mu, particles.log_sigma, particles.logit_alpha,
            s_idx, t_idx, downsample=ds,
        )  # (K, H', W')
        rendered = _apply_learnable_bg(rendered, bg_logit)

        # GT views
        gt_views = sai_stack[flat_idx]  # (K, H, W)
        if ds > 1:
            gt_views = gt_views[:, ::ds, ::ds]

        # 鲁棒归一化（分位数分母，抗噪声尖峰）
        g_max = _robust_gmax(gt_views, _gmax_quantile(s1), _gmax_mode(s1))
        rendered_n = rendered / g_max
        gt_n = gt_views / g_max

        # Loss
        losses = total_loss(rendered_n, gt_n,
                            particles.logit_alpha, particles.log_sigma, loss_cfg,
                            mu_pred=particles.mu, mu_gt=gt_centers)
        if plane_z is not None and plane_z_weight > 0:
            pz = ((particles.mu[:, 2] - float(plane_z)) ** 2).mean()
            losses["plane_z"] = pz
            losses["total"] = losses["total"] + plane_z_weight * pz

        optimizer.zero_grad()
        losses["total"].backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(particles.parameters(), max_norm=1.0)
        optimizer.step()

        # Clamp mu to physical volume
        with torch.no_grad():
            particles.mu[:, 0].clamp_(*cam.x_range)
            particles.mu[:, 1].clamp_(*cam.y_range)
            particles.mu[:, 2].clamp_(*cam.z_range)

        # Pruning (keep at least n_gt Gaussians)
        if it > prune_after and it % prune_every == 0 and particles.n > max(n_gt, 20):
            n_pruned = particles.prune(prune_thr, min_keep=max(n_gt, 20))
            if n_pruned > 0:
                # Rebuild optimiser after pruning
                optimizer = torch.optim.Adam([
                    {"params": [particles.mu], "lr": lr_mu},
                    {"params": [particles.logit_alpha], "lr": lr_alpha},
                    {"params": [particles.log_sigma], "lr": lr_sigma},
                ])
                print(f"  [prune] iter {it}: removed {n_pruned}, remaining {particles.n}")

        # Snapshot saving
        if snapshot_every > 0 and (it % snapshot_every == 0 or it == num_iters - 1):
            snap = {
                "mu": particles.mu.detach().cpu().numpy(),
                "log_sigma": particles.log_sigma.detach().cpu().numpy(),
                "logit_alpha": particles.logit_alpha.detach().cpu().numpy(),
            }
            np.savez_compressed(os.path.join(snapshots_dir, f"iter_{it:05d}.npz"), **snap)

        # Logging
        if it % log_every == 0 or it == num_iters - 1:
            with torch.no_grad():
                m = evaluate_gaussians(
                    particles.mu, particles.logit_alpha, gt_centers,
                    cam, alpha_threshold=prune_thr,
                    match_radius_voxel=float(cfg.get("eval", {}).get("match_radius", 2.0)),
                    use_nms=use_nms, nms_radius_voxel=nms_radius,
                )
            elapsed = time.time() - t0
            entry = {
                "iter": it,
                "loss": float(losses["total"].item()),
                "recon": float(losses["recon"].item()),
                "sparse": float(losses["sparse"].item()),
                "scale": float(losses["scale"].item()),
                "entropy": float(losses.get("entropy", torch.tensor(0.0)).item()),
                "hungarian_pos": float(losses["hungarian_pos"].item()) if "hungarian_pos" in losses else 0.0,
                "hungarian_alpha": float(losses["hungarian_alpha"].item()) if "hungarian_alpha" in losses else 0.0,
                "hungarian_n_matched": int(losses.get("hungarian_n_matched", 0)),
                "n_active": int(m["n_active"]),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "mean_dist": m["mean_dist"],
                "elapsed": elapsed,
            }
            history.append(entry)
            if it % log_every == 0:
                print(
                    f"  iter {it:5d} | loss {entry['loss']:.4f} "
                    f"(recon {entry['recon']:.4f}) | "
                    f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f} | "
                    f"active {int(m['n_active'])}/{particles.n} | "
                    f"dist {m['mean_dist']:.2f} | {elapsed:.1f}s"
                )

    # --- Save results ---
    final_metrics = history[-1] if history else {}

    # Save Gaussian parameters
    save_dict = {
        "mu": particles.mu.detach().cpu().numpy(),
        "log_sigma": particles.log_sigma.detach().cpu().numpy(),
        "logit_alpha": particles.logit_alpha.detach().cpu().numpy(),
        "gt_centers": gt_centers.cpu().numpy(),
    }
    if bg_logit is not None:
        save_dict["bg_logit"] = np.float32(bg_logit.detach().cpu().item())
    np.savez_compressed(os.path.join(output_dir, "gaussians.npz"), **save_dict)

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\n[LF-3DGS] Optimisation complete. Results saved to {output_dir}")
    print(f"[LF-3DGS] Final: F1={final_metrics.get('f1', 0):.3f}, "
          f"Precision={final_metrics.get('precision', 0):.3f}, "
          f"Recall={final_metrics.get('recall', 0):.3f}")

    return final_metrics


# ===================================================================== #
# Iterative refinement (two-round optimisation)
# ===================================================================== #

def optimise_sample_iterative(
    sample_path: str,
    output_dir: str,
    cfg: Dict,
    device: str = "cuda",
) -> Dict:
    """Two-round optimisation with residual-guided re-seeding.

    Round 1: Standard optimisation -> prune -> render residual
    Round 2: Back-project residual -> add new Gaussians -> re-optimise
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    s1 = cfg.get("stage1", cfg)
    round2_iters = int(s1.get("refine_round2_iters", 5000))

    # --- Round 1 ---
    r1_dir = os.path.join(output_dir, "round1")
    print(f"\n[Iterative] === Round 1 ===")
    r1_metrics = optimise_sample(sample_path, r1_dir, cfg, device)

    # Load round 1 results
    r1_data = np.load(os.path.join(r1_dir, "gaussians.npz"))
    mu_r1 = torch.from_numpy(r1_data["mu"]).float().to(dev)
    log_sigma_r1 = torch.from_numpy(r1_data["log_sigma"]).float().to(dev)
    logit_alpha_r1 = torch.from_numpy(r1_data["logit_alpha"]).float().to(dev)
    gt_centers = torch.from_numpy(r1_data["gt_centers"]).float().to(dev)

    # Keep only high-confidence Gaussians from round 1
    alpha_r1 = torch.sigmoid(logit_alpha_r1)
    keep_mask = alpha_r1 > 0.3
    mu_kept = mu_r1[keep_mask]
    log_sigma_kept = log_sigma_r1[keep_mask]
    logit_alpha_kept = logit_alpha_r1[keep_mask]
    print(f"[Iterative] Round 1: {mu_r1.shape[0]} -> {mu_kept.shape[0]} kept (alpha>0.3)")

    # --- Compute residual SAI ---
    sai_stack, _, meta = load_single_sample(sample_path)
    sai_stack = sai_stack.to(dev)
    cam = LFCameraParams()
    if "x_range" in meta:
        cam.x_range = meta["x_range"]
    if "y_range" in meta:
        cam.y_range = meta["y_range"]
    if "z_range" in meta:
        cam.z_range = meta["z_range"]
    if "grid_shape" in meta:
        cam.grid_shape = meta["grid_shape"]

    renderer = GaussianSplatRenderer(cam).to(dev)
    P = cam.px_per_mla

    with torch.no_grad():
        all_rendered = renderer.render_all_views(
            mu_kept, log_sigma_kept, logit_alpha_kept, P, batch_size=64,
        )
        g_max = _robust_gmax(sai_stack.unsqueeze(0), _gmax_quantile(s1), _gmax_mode(s1))
        residual = (sai_stack / g_max - all_rendered / g_max).clamp(min=0)

    # --- Back-project residual to find missed particles ---
    bp_stride = int(s1.get("backproj_stride", 2))
    bp_mode = str(s1.get("backproj_mode", "mean"))
    residual_stack = residual * g_max

    from .backproject import backproject_volume, find_peaks_3d, voxel_to_physical
    res_volume = backproject_volume(
        residual_stack, cam, device=str(dev),
        mode=bp_mode, view_stride=bp_stride, chunk_views=64,
    )
    res_vol_np = res_volume.cpu().numpy()
    new_peaks = find_peaks_3d(res_vol_np, threshold=0.10, min_distance=2)

    if new_peaks.shape[0] > 0:
        new_centers_phys = voxel_to_physical(new_peaks, cam)
        new_centers = torch.from_numpy(new_centers_phys).float().to(dev)
        print(f"[Iterative] Residual back-projection found {new_centers.shape[0]} new candidates")
    else:
        new_centers = torch.empty(0, 3, device=dev)
        print("[Iterative] No new peaks in residual")

    # --- Round 2: Merge and re-optimise ---
    print(f"\n[Iterative] === Round 2 ({round2_iters} iters) ===")
    n_old = mu_kept.shape[0]
    n_new = new_centers.shape[0]

    if n_new > 0:
        combined_mu = torch.cat([mu_kept, new_centers], dim=0)
        combined_log_sigma = torch.cat([
            log_sigma_kept,
            torch.full((n_new,), float(np.log(0.8)), device=dev),
        ], dim=0)
        combined_logit_alpha = torch.cat([
            logit_alpha_kept,
            torch.full((n_new,), 0.0, device=dev),
        ], dim=0)
    else:
        combined_mu = mu_kept
        combined_log_sigma = log_sigma_kept
        combined_logit_alpha = logit_alpha_kept

    particles = GaussianParticles.__new__(GaussianParticles)
    nn.Module.__init__(particles)
    particles.mu = nn.Parameter(combined_mu.clone())
    particles.log_sigma = nn.Parameter(combined_log_sigma.clone())
    particles.logit_alpha = nn.Parameter(combined_logit_alpha.clone())
    print(f"[Iterative] Round 2 init: {n_old} kept + {n_new} new = {particles.n} total")

    # Round 2 config: shorter, all params trainable from start
    lr_mu = float(s1.get("lr_mu", 0.02))
    lr_alpha = float(s1.get("lr_alpha", 0.005))
    lr_sigma = float(s1.get("lr_sigma", 0.001))
    K = int(s1.get("views_per_step", 64))
    ds = int(s1.get("render_downsample", 1))

    loss_cfg = {
        "recon_type": s1.get("recon_type", "weighted_l1"),
        "recon_weight": float(s1.get("recon_weight", 1.0)),
        "sparse_weight": float(s1.get("sparse_weight", 0.0)),
        "scale_weight": float(s1.get("scale_weight", 0.0)),
        "entropy_weight": float(s1.get("entropy_weight", 0.0)),
        "signal_weight": float(s1.get("signal_weight", 10.0)),
        "signal_thr": float(s1.get("signal_thr", 0.01)),
        "hungarian_pos_weight": float(s1.get("hungarian_pos_weight", 0.0)),
        "hungarian_alpha_weight": float(s1.get("hungarian_alpha_weight", 0.0)),
        "hungarian_match_radius": float(s1.get("hungarian_match_radius", 5.0)),
    }

    prune_thr = float(s1.get("prune_alpha_threshold", 0.3))
    use_nms = bool(s1.get("use_nms", False))
    nms_radius = float(s1.get("nms_radius_voxel", 1.5))

    optimizer = torch.optim.Adam([
        {"params": [particles.mu], "lr": lr_mu},
        {"params": [particles.logit_alpha], "lr": lr_alpha},
        {"params": [particles.log_sigma], "lr": lr_sigma},
    ])

    history = []
    t0 = time.time()

    for it in range(round2_iters):
        s_idx, t_idx = renderer.sample_views(K, P, dev)
        flat_idx = renderer.sai_index(s_idx, t_idx, P)

        rendered = renderer(
            particles.mu, particles.log_sigma, particles.logit_alpha,
            s_idx, t_idx, downsample=ds,
        )
        gt_views = sai_stack[flat_idx]
        if ds > 1:
            gt_views = gt_views[:, ::ds, ::ds]

        g_max_val = _robust_gmax(gt_views, _gmax_quantile(s1), _gmax_mode(s1))
        rendered_n = rendered / g_max_val
        gt_n = gt_views / g_max_val

        losses = total_loss(rendered_n, gt_n,
                            particles.logit_alpha, particles.log_sigma, loss_cfg,
                            mu_pred=particles.mu, mu_gt=gt_centers)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(particles.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            particles.mu[:, 0].clamp_(*cam.x_range)
            particles.mu[:, 1].clamp_(*cam.y_range)
            particles.mu[:, 2].clamp_(*cam.z_range)

        # Pruning in round 2: prune aggressively after half-way
        if it > round2_iters // 2 and it % 500 == 0:
            n_gt = gt_centers.shape[0]
            if particles.n > max(n_gt, 20):
                n_pruned = particles.prune(prune_thr, min_keep=max(n_gt, 20))
                if n_pruned > 0:
                    optimizer = torch.optim.Adam([
                        {"params": [particles.mu], "lr": lr_mu},
                        {"params": [particles.logit_alpha], "lr": lr_alpha},
                        {"params": [particles.log_sigma], "lr": lr_sigma},
                    ])

        if it % 500 == 0 or it == round2_iters - 1:
            with torch.no_grad():
                m = evaluate_gaussians(
                    particles.mu, particles.logit_alpha, gt_centers,
                    cam, alpha_threshold=prune_thr,
                    match_radius_voxel=float(cfg.get("eval", {}).get("match_radius", 2.5)),
                    use_nms=use_nms, nms_radius_voxel=nms_radius,
                )
            elapsed = time.time() - t0
            entry = {
                "iter": it, "round": 2,
                "loss": float(losses["total"].item()),
                "n_active": int(m["n_active"]),
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "mean_dist": m["mean_dist"],
                "elapsed": elapsed,
            }
            history.append(entry)
            print(f"  R2 iter {it:5d} | loss {entry['loss']:.4f} | "
                  f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f} | "
                  f"active {int(m['n_active'])}/{particles.n}")

    # --- Save final ---
    final_metrics = history[-1] if history else r1_metrics
    final_metrics["r1_f1"] = r1_metrics.get("f1", 0)
    final_metrics["n_new_from_residual"] = n_new

    save_dict = {
        "mu": particles.mu.detach().cpu().numpy(),
        "log_sigma": particles.log_sigma.detach().cpu().numpy(),
        "logit_alpha": particles.logit_alpha.detach().cpu().numpy(),
        "gt_centers": gt_centers.cpu().numpy(),
    }
    np.savez_compressed(os.path.join(output_dir, "gaussians.npz"), **save_dict)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\n[Iterative] Complete. R1 F1={r1_metrics.get('f1', 0):.3f} -> "
          f"R2 F1={final_metrics.get('f1', 0):.3f}")
    return final_metrics


# ===================================================================== #
# CLI entry point
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="LF-3DGS per-sample optimisation")
    parser.add_argument("--sample", type=str, required=True, help="Path to .npz sample")
    parser.add_argument("--output", type=str, default="output/lf3dgs_opt", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=None, help="Override num_iters")
    parser.add_argument("--views", type=int, default=None, help="Override views_per_step")
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

    # CLI overrides
    if args.iters:
        cfg.setdefault("stage1", {})["num_iters"] = args.iters
    if args.views:
        cfg.setdefault("stage1", {})["views_per_step"] = args.views

    optimise_sample(args.sample, args.output, cfg, args.device)


if __name__ == "__main__":
    main()
