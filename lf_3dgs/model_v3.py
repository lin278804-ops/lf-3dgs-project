"""
Stage 2 V3: Cost-volume amortised inference (2026-08 repair).

2026-08 P0/P1 (diagnose_stage2):
  - Encode at native SAI resolution (186x232). Do NOT stretch to 128x128
    before the cost volume (that caused Δy ≈ −2.66 vox).
  - Occupancy head still emits (D, 128, 128); decoder is resampled at the end.
  - Local-max gating before top-K (extract_mode="peak").
  - Offset head is tanh-bounded to ±0.5 voxel.
  - Train/eval both use uniform angular sampling.
  - All interpolations use align_corners=True (same as grid_sample).
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
# Encoder: outputs at two resolutions
# ------------------------------------------------------------------ #

class ViewEncoderV3(nn.Module):
    """
    2D encoder producing features at stem (128x128) and downsampled (64x64)
    resolution simultaneously.

    stem_feat → used as high-frequency skip connection
    down_feat → used for cost volume construction
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.stem = ConvBnRelu2d(1, base_channels)      # 128x128
        self.down1 = ConvBnRelu2d(base_channels, base_channels)  # 64x64

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, 1, H, W)
        Returns:
            stem_feat: (N, C, H, W)         — full resolution (e.g. 128x128)
            down_feat: (N, C, H//2, W//2)   — half resolution (e.g. 64x64)
        """
        stem = self.stem(x)                         # (N, C, H, W)
        down = F.avg_pool2d(stem, 2, 2)
        down = self.down1(down)                     # (N, C, H//2, W//2)
        return stem, down


# ------------------------------------------------------------------ #
# 3D Decoder with skip connection fusion
# ------------------------------------------------------------------ #

class Decoder3DV3(nn.Module):
    """
    3D regularizer + spatial upsample (64→128) + fuse with full-res skip.

    The skip connection carries aggregated 128x128 stem features
    (mean + variance across views) replicated along the depth axis,
    providing high-frequency spatial cues absent from the coarser cost volume.
    """

    def __init__(self, cv_channels: int, mid_channels: int,
                 skip_channels: int) -> None:
        """
        Args:
            cv_channels:   input channels from cost volume (typically 2*base_ch)
            mid_channels:  internal channels (typically base_ch)
            skip_channels: channels of skip features (typically 2*base_ch)
        """
        super().__init__()

        # Initial 3D regularization at 64x64 spatial
        self.initial = nn.Sequential(
            ConvBnRelu3d(cv_channels, mid_channels),
            ConvBnRelu3d(mid_channels, mid_channels),
        )

        # Learnable 64→128 spatial upsample (depth unchanged)
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                mid_channels, mid_channels,
                kernel_size=(1, 4, 4), stride=(1, 2, 2),
                padding=(0, 1, 1), bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            ConvBnRelu3d(mid_channels, mid_channels),
        )

        # Fuse with skip (concat then 3D conv)
        self.fusion = ConvBnRelu3d(mid_channels + skip_channels, mid_channels)

    def forward(self, cost_vol: torch.Tensor,
                skip_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_vol: (B, 2C, D, 64, 64)
            skip_3d:  (B, 2C, D, 128, 128)  — stem features replicated along D
        Returns:
            (B, C, D, 128, 128)
        """
        x = self.initial(cost_vol)                        # (B, C, D, 64, 64)
        residual = F.interpolate(
            x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True
        )
        x = self.up(x) + residual                         # (B, C, D, 128, 128)
        x = torch.cat([x, skip_3d], dim=1)                # (B, C+2C, D, 128, 128)
        x = self.fusion(x)                                # (B, C, D, 128, 128)
        return x


class DepthAwareSkip(nn.Module):
    """Lift a 2D skip to 3D without copying the same map onto every z.

    Identity at init: output == broadcast(skip_2d). New params can warm-start
    from a broadcast-trained checkpoint (strict=False).
    """

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, depth, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, depth, 1, 1))
        self.mix = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.mix.weight)

    def forward(self, skip_2d: torch.Tensor, depth: int) -> torch.Tensor:
        x = skip_2d.unsqueeze(2).expand(-1, -1, depth, -1, -1).contiguous()
        return x * (1.0 + self.gamma) + self.beta + self.mix(x)


# ------------------------------------------------------------------ #
# Main model
# ------------------------------------------------------------------ #

class LF3DGSNetV3(nn.Module):
    """
    High-resolution cost-volume-based amortised inference network.

    Improvements vs V2:
      - Cost volume at 64x64 spatial resolution (vs 32x32 in V2)
      - Full-resolution 128x128 skip connection via stem features
      - AMP-compatible
    """

    def __init__(
        self,
        out_shape: Tuple[int, int, int] = (64, 128, 128),
        base_channels: int = 32,
        topk: int = 3500,
        n_views_cost: int = 16,
        encode_native: bool = True,
        extract_mode: str = "peak",
        bound_offset: bool = True,
        view_sample: str = "uniform",
        skip_mode: str = "broadcast",
    ) -> None:
        super().__init__()
        D, H, W = out_shape
        self.out_shape = out_shape
        self.topk = topk
        self.n_views_cost = n_views_cost
        self.encode_native = bool(encode_native)
        self.extract_mode = str(extract_mode)
        self.bound_offset = bool(bound_offset)
        self.view_sample = str(view_sample)
        self.skip_mode = str(skip_mode)

        self.view_enc = ViewEncoderV3(base_channels)

        cv_channels = 2 * base_channels   # mean + variance of down features
        skip_channels = 2 * base_channels  # mean + variance of stem features

        self.decoder = Decoder3DV3(cv_channels, base_channels, skip_channels)
        self.skip_lift = (
            DepthAwareSkip(skip_channels, D) if self.skip_mode == "depth_aware"
            else None
        )

        out_per_voxel = 5  # alpha, dx, dy, dz, log_sigma
        self.head_3d = nn.Conv3d(base_channels, out_per_voxel, 1, bias=True)

        with torch.no_grad():
            self.head_3d.bias.data.zero_()
            self.head_3d.bias.data[0] = -4.0  # sparse alpha init

    # ----------------------------------------------------------
    # Plane sweep cost volume
    # ----------------------------------------------------------
    def _build_cost_volume(
        self,
        feat_down: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        cam_params: LFCameraParams,
        cv_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Variance-based cost volume via plane sweep at 64x64 resolution.
        Returns: (B, 2C, D_cv, H_cv, W_cv)
        """
        B, K, C, Hf, Wf = feat_down.shape
        D_cv, H_cv, W_cv = cv_shape
        device = feat_down.device

        xr = cam_params.x_range
        yr = cam_params.y_range
        zr = cam_params.z_range

        z_vals = torch.linspace(zr[0], zr[1], D_cv, device=device)
        y_vals = torch.linspace(yr[0], yr[1], H_cv, device=device)
        x_vals = torch.linspace(xr[0], xr[1], W_cv, device=device)
        zz, yy, xx = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
        voxel_centers = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        proj = DifferentiableLFProjection(cam_params).to(device)
        with torch.no_grad():
            u_proj, v_proj = proj(voxel_centers, s_idx, t_idx)  # (K, N_vox)

        # Normalize to [-1, 1] for grid_sample
        grid_y = 2.0 * u_proj / max(cam_params.num_mla_y - 1, 1) - 1.0
        grid_x = 2.0 * v_proj / max(cam_params.num_mla_x - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (K, N_vox, 2)

        feat_flat = feat_down.reshape(B * K, C, Hf, Wf)
        N_vox = voxel_centers.shape[0]
        grid_bk = grid.unsqueeze(0).expand(B, K, N_vox, 2)
        grid_bk = grid_bk.reshape(B * K, 1, N_vox, 2)

        sampled = F.grid_sample(
            feat_flat, grid_bk, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )  # (B*K, C, 1, N_vox)

        sampled = sampled.squeeze(2).reshape(B, K, C, D_cv, H_cv, W_cv)

        mean_feat = sampled.mean(dim=1)
        var_feat = sampled.var(dim=1, unbiased=False).clamp(min=1e-6)
        return torch.cat([mean_feat, var_feat], dim=1)  # (B, 2C, D, H_cv, W_cv)

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

        if self.view_sample == "random" and self.training:
            perm = torch.randperm(total_views, device=device)[:K]
        else:
            perm = torch.linspace(0, total_views - 1, K, device=device).long()

        s_sel = perm // u
        t_sel = perm % u

        lf_flat = lf4d.reshape(b, total_views, h, w)
        selected = lf_flat[:, perm]  # (B, K, H, W) native SAI

        # Legacy path only: anisotropic stretch to occupancy HW (DO NOT use).
        if (not self.encode_native) and (oh != h or ow != w):
            selected = F.interpolate(
                selected.reshape(b * K, 1, h, w),
                size=(oh, ow), mode="bilinear", align_corners=True,
            ).reshape(b, K, oh, ow)

        sel_flat = selected.reshape(b * K, 1, selected.shape[2], selected.shape[3])
        stem_flat, down_flat = self.view_enc(sel_flat)
        _, C_stem, Hs, Ws = stem_flat.shape
        _, C_down, Hf, Wf = down_flat.shape

        stem_views = stem_flat.reshape(b, K, C_stem, Hs, Ws)
        down_views = down_flat.reshape(b, K, C_down, Hf, Wf)

        # Cost volume at 64x64 spatial resolution
        cv_shape = (D, Hf, Wf)
        cost_vol = self._build_cost_volume(
            down_views, s_sel, t_sel, cam_params, cv_shape,
        )  # (B, 2C, D, 64, 64)

        # Stem skip: aggregate across views → replicate along depth
        stem_mean = stem_views.mean(dim=1)                          # (B, C, 128, 128)
        stem_var = stem_views.var(dim=1, unbiased=False).clamp(min=1e-6)
        skip_2d = torch.cat([stem_mean, stem_var], dim=1)           # (B, 2C, 128, 128)
        if self.skip_lift is not None:
            skip_3d = self.skip_lift(skip_2d, D)
        else:
            skip_3d = skip_2d.unsqueeze(2).expand(-1, -1, D, -1, -1).contiguous()
        # (B, 2C, D, 128, 128)

        # Decode: regularize + upsample + fuse skip
        decoded = self.decoder(cost_vol, skip_3d)   # (B, C, D, 128, 128)

        # Ensure final spatial dims match output
        _, Cd, Dd, Hd, Wd = decoded.shape
        if Hd != H_out or Wd != W_out:
            dec_2d = decoded.reshape(b * Dd, Cd, Hd, Wd)
            dec_2d = F.interpolate(dec_2d, size=(H_out, W_out),
                                   mode="bilinear", align_corners=True)
            decoded = dec_2d.reshape(b, Cd, Dd, H_out, W_out)

        raw = self.head_3d(decoded)               # (B, 5, D, H, W)
        alpha_logits = raw[:, 0]
        off_raw = raw[:, 1:4]
        if self.bound_offset:
            off_raw = torch.tanh(off_raw) * 0.5
        offsets = off_raw.permute(0, 2, 3, 4, 1)
        log_sigma = raw[:, 4]

        gauss = {
            "alpha_logits": alpha_logits,
            "offsets": offsets,
            "log_sigma": log_sigma,
        }

        result = self._extract_topk(gauss, cam_params)
        result["volume"] = gauss
        return result

    # ----------------------------------------------------------
    # Top-K extraction (same as V2)
    # ----------------------------------------------------------
    def _extract_topk(
        self,
        gauss: Dict[str, torch.Tensor],
        cam_params: Optional[LFCameraParams] = None,
    ) -> Dict[str, torch.Tensor]:
        alpha_logits = gauss["alpha_logits"]
        offsets = gauss["offsets"]
        log_sigma = gauss["log_sigma"]

        b, D, H, W = alpha_logits.shape
        if self.extract_mode == "peak":
            pooled = F.max_pool3d(
                alpha_logits.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            is_peak = alpha_logits >= (pooled - 1e-6)
            scored = alpha_logits.masked_fill(~is_peak, -1.0e4)
        else:
            scored = alpha_logits
        alpha_flat = scored.reshape(b, -1)
        raw_flat = alpha_logits.reshape(b, -1)
        K = min(self.topk, alpha_flat.shape[1])

        _, topk_idx = torch.topk(alpha_flat, K, dim=1)

        offsets_flat = offsets.reshape(b, -1, 3)
        log_sigma_flat = log_sigma.reshape(b, -1)

        bi = torch.arange(b, device=alpha_flat.device).unsqueeze(1).expand(-1, K)
        topk_alpha = raw_flat[bi, topk_idx]
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
