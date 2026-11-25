"""
Stage 2 V2: Cost-Volume-based amortised inference network for LF-3DGS.

Key improvement over V1 (model.py):
  V1: AngularMixer does attention-weighted mean of 484 views -> loses depth info.
  V2: Plane-sweep cost volume preserves the depth<->disparity relationship.

Architecture:
  Input: lf4d (B, V, U, H, W)
    -> ViewEncoderShallow (shared 2D CNN, fewer downsampling stages)
    -> Sample K views for cost volume construction
    -> Plane Sweep: project voxel grid to each view, sample features
    -> Variance-based cost aggregation across views
    -> 3D Decoder (regularizer + progressive upsampling to output resolution)
    -> Per-voxel heads: alpha, offset, log_sigma
    -> Top-K extraction
  Output: {mu, sigma, alpha} for K Gaussians per sample
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import DifferentiableLFProjection, LFCameraParams


# ------------------------------------------------------------------ #
# Building blocks
# ------------------------------------------------------------------ #

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=ks // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBnRelu3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------------------ #
# Encoder
# ------------------------------------------------------------------ #

class ViewEncoderShallow(nn.Module):
    """2D encoder with configurable downsampling depth (default 2x = 4x spatial)."""

    def __init__(self, base_channels: int, n_downsample: int = 2) -> None:
        super().__init__()
        self.stem = ConvBnRelu2d(1, base_channels)
        self.downs = nn.ModuleList()
        for _ in range(n_downsample):
            self.downs.append(ConvBnRelu2d(base_channels, base_channels))
        self.n_downsample = n_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.downs:
            x = F.avg_pool2d(x, 2, 2)
            x = blk(x)
        return x


# ------------------------------------------------------------------ #
# 3D Decoder with progressive spatial upsampling
# ------------------------------------------------------------------ #

class Decoder3D(nn.Module):
    """Regularizer + learnable spatial upsampling (depth stays fixed)."""

    def __init__(self, in_channels: int, mid_channels: int,
                 n_upsample: int = 2) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            ConvBnRelu3d(in_channels, mid_channels),
            ConvBnRelu3d(mid_channels, mid_channels),
        )
        self.up_blocks = nn.ModuleList()
        for _ in range(n_upsample):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose3d(
                    mid_channels, mid_channels,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2),
                    padding=(0, 1, 1), bias=False,
                ),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                ConvBnRelu3d(mid_channels, mid_channels),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        for blk in self.up_blocks:
            skip = F.interpolate(
                x, scale_factor=(1, 2, 2), mode="trilinear",
                align_corners=False,
            )
            x = blk(x) + skip
        return x


# ------------------------------------------------------------------ #
# Main model
# ------------------------------------------------------------------ #

class LF3DGSNetV2(nn.Module):
    """
    Cost-volume-based amortised inference network.

    Uses plane sweep to build a 3D cost volume from sampled views,
    then decodes per-voxel Gaussian parameters via a 3D decoder
    with progressive spatial upsampling.
    """

    def __init__(
        self,
        out_shape: Tuple[int, int, int] = (64, 128, 128),
        base_channels: int = 32,
        topk: int = 3500,
        n_views_cost: int = 16,
        n_downsample: int = 2,
    ) -> None:
        super().__init__()
        D, H, W = out_shape
        self.out_shape = out_shape
        self.topk = topk
        self.n_views_cost = n_views_cost
        self.n_downsample = n_downsample

        self.view_enc = ViewEncoderShallow(base_channels, n_downsample)

        cv_channels = 2 * base_channels  # mean + variance
        self.decoder = Decoder3D(cv_channels, base_channels, n_downsample)

        out_per_voxel = 5  # alpha, dx, dy, dz, log_sigma
        self.head_3d = nn.Conv3d(base_channels, out_per_voxel, 1, bias=True)

        with torch.no_grad():
            self.head_3d.bias.data.zero_()
            self.head_3d.bias.data[0] = -4.0  # sparse alpha init

    # ----------------------------------------------------------
    # Plane sweep cost volume construction
    # ----------------------------------------------------------
    def _build_cost_volume(
        self,
        feat_views: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        cam_params: LFCameraParams,
        cv_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Build variance-based cost volume via plane sweep.

        Returns: (B, 2C, D_cv, H_cv, W_cv)
        """
        B, K, C, Hf, Wf = feat_views.shape
        D_cv, H_cv, W_cv = cv_shape
        device = feat_views.device

        xr = cam_params.x_range
        yr = cam_params.y_range
        zr = cam_params.z_range

        z_vals = torch.linspace(zr[0], zr[1], D_cv, device=device)
        y_vals = torch.linspace(yr[0], yr[1], H_cv, device=device)
        x_vals = torch.linspace(xr[0], xr[1], W_cv, device=device)
        zz, yy, xx = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
        voxel_centers = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        N_vox = voxel_centers.shape[0]

        proj = DifferentiableLFProjection(cam_params).to(device)
        with torch.no_grad():
            u_proj, v_proj = proj(voxel_centers, s_idx, t_idx)  # (K, N_vox)

        # Normalize SAI coordinates to [-1, 1] for grid_sample
        grid_y = 2.0 * u_proj / max(cam_params.num_mla_y - 1, 1) - 1.0
        grid_x = 2.0 * v_proj / max(cam_params.num_mla_x - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (K, N_vox, 2)

        feat_flat = feat_views.reshape(B * K, C, Hf, Wf)
        grid_bk = grid.unsqueeze(0).expand(B, K, N_vox, 2)
        grid_bk = grid_bk.reshape(B * K, 1, N_vox, 2)

        sampled = F.grid_sample(
            feat_flat, grid_bk, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )  # (B*K, C, 1, N_vox)

        sampled = sampled.squeeze(2).reshape(B, K, C, D_cv, H_cv, W_cv)

        mean_feat = sampled.mean(dim=1)
        var_feat = sampled.var(dim=1, unbiased=False).clamp(min=1e-6)
        return torch.cat([mean_feat, var_feat], dim=1)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(
        self,
        lf4d: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
        cam_params: Optional[LFCameraParams] = None,
    ) -> Dict[str, torch.Tensor]:
        b, v, u, h, w = lf4d.shape
        D, H_out, W_out = self.out_shape
        device = lf4d.device

        if out_hw is None:
            out_hw = (H_out, W_out)
        oh, ow = int(out_hw[0]), int(out_hw[1])

        total_views = v * u
        K = min(self.n_views_cost, total_views)

        if self.training:
            perm = torch.randperm(total_views, device=device)[:K]
        else:
            perm = torch.linspace(0, total_views - 1, K, device=device).long()

        s_sel = perm // u
        t_sel = perm % u

        # Gather and resize selected views
        lf_flat = lf4d.reshape(b, total_views, h, w)
        selected = lf_flat[:, perm]  # (B, K, H, W)

        if oh != h or ow != w:
            selected = F.interpolate(
                selected.reshape(b * K, 1, h, w),
                size=(oh, ow), mode="bilinear", align_corners=False,
            ).reshape(b, K, oh, ow)

        # Encode views
        sel_flat = selected.reshape(b * K, 1, selected.shape[2], selected.shape[3])
        feat = self.view_enc(sel_flat)
        _, C, Hf, Wf = feat.shape
        feat_views = feat.reshape(b, K, C, Hf, Wf)

        # Cost volume at feature resolution
        cv_shape = (D, Hf, Wf)
        cost_vol = self._build_cost_volume(
            feat_views, s_sel, t_sel, cam_params, cv_shape,
        )

        # 3D decode + upsample
        decoded = self.decoder(cost_vol)
        _, Cd, Dd, Hd, Wd = decoded.shape

        # Ensure final spatial dims match output
        if Hd != H_out or Wd != W_out:
            dec_2d = decoded.reshape(b * Dd, Cd, Hd, Wd)
            dec_2d = F.interpolate(dec_2d, size=(H_out, W_out),
                                   mode="bilinear", align_corners=False)
            decoded = dec_2d.reshape(b, Cd, Dd, H_out, W_out)

        # Per-voxel predictions
        raw = self.head_3d(decoded)  # (B, 5, D, H, W)
        alpha_logits = raw[:, 0]
        offsets = raw[:, 1:4].permute(0, 2, 3, 4, 1)
        log_sigma = raw[:, 4]

        gauss = {
            "alpha_logits": alpha_logits,
            "offsets": offsets,
            "log_sigma": log_sigma,
        }

        result = self._extract_topk(gauss, cam_params)
        result["volume"] = gauss
        return result

    def _extract_topk(
        self,
        gauss: Dict[str, torch.Tensor],
        cam_params: Optional[LFCameraParams] = None,
    ) -> Dict[str, torch.Tensor]:
        alpha_logits = gauss["alpha_logits"]
        offsets = gauss["offsets"]
        log_sigma = gauss["log_sigma"]

        b, D, H, W = alpha_logits.shape
        alpha_flat = alpha_logits.reshape(b, -1)
        K = min(self.topk, alpha_flat.shape[1])

        _, topk_idx = torch.topk(alpha_flat, K, dim=1)

        offsets_flat = offsets.reshape(b, -1, 3)
        log_sigma_flat = log_sigma.reshape(b, -1)

        bi = torch.arange(b, device=alpha_flat.device).unsqueeze(1).expand(-1, K)
        topk_alpha = alpha_flat[bi, topk_idx]
        topk_offset = offsets_flat[bi, topk_idx]
        topk_log_sigma = log_sigma_flat[bi, topk_idx]

        d_idx = topk_idx // (H * W)
        hw_rem = topk_idx % (H * W)
        h_idx = hw_rem // W
        w_idx = hw_rem % W

        if cam_params is not None:
            D_vol, H_vol, W_vol = self.out_shape
            xr = cam_params.x_range
            yr = cam_params.y_range
            zr = cam_params.z_range

            x_phys = xr[0] + (w_idx.float() + 0.5) / W_vol * (xr[1] - xr[0])
            y_phys = yr[0] + (h_idx.float() + 0.5) / H_vol * (yr[1] - yr[0])
            z_phys = zr[0] + (d_idx.float() + 0.5) / D_vol * (zr[1] - zr[0])

            voxel_size_x = (xr[1] - xr[0]) / W_vol
            voxel_size_y = (yr[1] - yr[0]) / H_vol
            voxel_size_z = (zr[1] - zr[0]) / D_vol
            x_phys = x_phys + topk_offset[:, :, 0] * voxel_size_x
            y_phys = y_phys + topk_offset[:, :, 1] * voxel_size_y
            z_phys = z_phys + topk_offset[:, :, 2] * voxel_size_z

            mu = torch.stack([x_phys, y_phys, z_phys], dim=-1)
        else:
            mu = torch.stack([
                w_idx.float(), h_idx.float(), d_idx.float(),
            ], dim=-1)

        return {
            "mu": mu,
            "logit_alpha": topk_alpha,
            "log_sigma": topk_log_sigma,
        }
