"""
Stage 2: Amortised inference network for LF-3DGS.

Architecture:
  Input: lf4d (B, V, U, H, W)
    -> ViewEncoder (shared 2D CNN per view)
    -> AngularMixer (attention-weighted aggregation across views)
    -> SpatialMixer (2D refinement)
    -> GaussianHead (predict per-voxel Gaussian parameters)
    -> Top-K extraction (select active Gaussians)
  Output: {mu, sigma, alpha} for K Gaussians per sample

The ViewEncoder and AngularMixer are architecturally compatible with
lf4d_center/model.py for potential weight transfer.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Building blocks (compatible with lf4d_center/model.py)
# ------------------------------------------------------------------ #

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViewEncoder(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.stem = ConvBlock2d(1, base_channels)
        self.down1 = ConvBlock2d(base_channels, base_channels)
        self.down2 = ConvBlock2d(base_channels, base_channels)
        self.down3 = ConvBlock2d(base_channels, base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.down1(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.down2(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.down3(x)
        return x


class AngularMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = feats.shape
        x = feats.reshape(b * n, c, h, w)
        x = self.mix(x)
        wmap = torch.sigmoid(self.attn(x))
        x = x * wmap
        x = x.reshape(b, n, c, h, w).mean(dim=1)
        return x


class SpatialMixer(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock2d(channels, channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ------------------------------------------------------------------ #
# Gaussian prediction head
# ------------------------------------------------------------------ #

class GaussianHead(nn.Module):
    """
    Predicts per-voxel Gaussian parameters from 2D feature maps.

    Outputs a 3D volume (B, C_out, D, H, W) where C_out = 5:
      - channel 0: alpha logit (existence probability)
      - channels 1-3: position offset (dx, dy, dz) relative to voxel centre
      - channel 4: log sigma (2D rendering spread)
    """

    def __init__(self, in_channels: int, depth_bins: int) -> None:
        super().__init__()
        self.depth_bins = depth_bins
        out_per_depth = 5  # alpha, dx, dy, dz, log_sigma
        self.head = nn.Conv2d(in_channels, depth_bins * out_per_depth, 1, bias=True)

        # Init alpha bias to be sparse
        with torch.no_grad():
            bias = self.head.bias.data
            bias.zero_()
            # Set alpha channel biases to a negative value for sparse init
            for d in range(depth_bins):
                bias[d * out_per_depth] = -4.0

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) spatial feature map
        Returns:
            dict with:
                alpha_logits: (B, D, H, W)
                offsets: (B, D, H, W, 3)
                log_sigma: (B, D, H, W)
        """
        b, c, h, w = feat.shape
        D = self.depth_bins
        raw = self.head(feat)  # (B, D*5, H, W)
        raw = raw.reshape(b, D, 5, h, w)

        alpha_logits = raw[:, :, 0, :, :]           # (B, D, H, W)
        offsets = raw[:, :, 1:4, :, :].permute(0, 1, 3, 4, 2)  # (B, D, H, W, 3)
        log_sigma = raw[:, :, 4, :, :]              # (B, D, H, W)

        return {
            "alpha_logits": alpha_logits,
            "offsets": offsets,
            "log_sigma": log_sigma,
        }


# ------------------------------------------------------------------ #
# Full network
# ------------------------------------------------------------------ #

class LF3DGSNet(nn.Module):
    """
    Amortised inference network: lf4d -> Gaussian parameters.

    Input:  (B, V, U, H_in, W_in)  light field
    Output: dict with per-voxel predictions and extracted top-K Gaussians
    """

    def __init__(
        self,
        out_shape: Tuple[int, int, int] = (64, 128, 128),
        base_channels: int = 32,
        spatial_align: bool = True,
        topk: int = 200,
    ) -> None:
        super().__init__()
        D, H, W = out_shape
        self.out_shape = out_shape
        self.spatial_align = spatial_align
        self.topk = topk

        self.view_enc = ViewEncoder(base_channels)
        self.ang_mix = AngularMixer(base_channels)
        self.spa_mix = SpatialMixer(base_channels, num_blocks=2)
        self.gauss_head = GaussianHead(base_channels, depth_bins=D)

    def forward(
        self,
        lf4d: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
        cam_params=None,
    ) -> Dict[str, torch.Tensor]:
        b, v, u, h, w = lf4d.shape
        D, H_out, W_out = self.out_shape

        if out_hw is None:
            out_hw = (H_out, W_out)
        oh, ow = int(out_hw[0]), int(out_hw[1])

        # Optionally downsample input for efficiency
        if self.spatial_align and oh <= h and ow <= w:
            x = lf4d.reshape(b * v * u, 1, h, w)
            x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)
            lf4d = x.reshape(b, v, u, oh, ow)
            h, w = oh, ow

        # Per-view encoding
        x = lf4d.reshape(b * v * u, 1, h, w)
        feat = self.view_enc(x)
        _, c, hp, wp = feat.shape
        feat = feat.reshape(b, v * u, c, hp, wp)

        # Angular mixing
        feat = self.ang_mix(feat)   # (B, C, hp, wp)
        feat = self.spa_mix(feat)

        # Upsample to output resolution
        feat = F.interpolate(feat, size=(oh, ow), mode="bilinear", align_corners=False)

        # Predict Gaussian parameters
        gauss = self.gauss_head(feat)

        # Extract top-K Gaussians per sample
        result = self._extract_topk(gauss, cam_params)
        result["volume"] = gauss  # keep volume predictions for loss computation
        return result

    def _extract_topk(
        self,
        gauss: Dict[str, torch.Tensor],
        cam_params=None,
    ) -> Dict[str, torch.Tensor]:
        """Extract top-K Gaussians by alpha probability."""
        alpha_logits = gauss["alpha_logits"]    # (B, D, H, W)
        offsets = gauss["offsets"]              # (B, D, H, W, 3)
        log_sigma = gauss["log_sigma"]         # (B, D, H, W)

        b, D, H, W = alpha_logits.shape
        alpha_flat = alpha_logits.reshape(b, -1)  # (B, D*H*W)
        K = min(self.topk, alpha_flat.shape[1])

        _, topk_idx = torch.topk(alpha_flat, K, dim=1)  # (B, K)

        # Gather parameters
        offsets_flat = offsets.reshape(b, -1, 3)
        log_sigma_flat = log_sigma.reshape(b, -1)

        batch_idx = torch.arange(b, device=alpha_flat.device).unsqueeze(1).expand(-1, K)
        topk_alpha = alpha_logits.reshape(b, -1)[batch_idx, topk_idx]
        topk_offset = offsets_flat[batch_idx, topk_idx]
        topk_log_sigma = log_sigma_flat[batch_idx, topk_idx]

        # Convert flat index to (d, h, w)
        d_idx = topk_idx // (H * W)
        hw_rem = topk_idx % (H * W)
        h_idx = hw_rem // W
        w_idx = hw_rem % W

        # Convert voxel indices + offsets to physical coordinates
        if cam_params is not None:
            D_vol, H_vol, W_vol = self.out_shape
            xr = cam_params.x_range
            yr = cam_params.y_range
            zr = cam_params.z_range

            x_phys = xr[0] + (w_idx.float() + 0.5) / W_vol * (xr[1] - xr[0])
            y_phys = yr[0] + (h_idx.float() + 0.5) / H_vol * (yr[1] - yr[0])
            z_phys = zr[0] + (d_idx.float() + 0.5) / D_vol * (zr[1] - zr[0])

            # Add learned offsets (scaled to voxel size)
            voxel_size_x = (xr[1] - xr[0]) / W_vol
            voxel_size_y = (yr[1] - yr[0]) / H_vol
            voxel_size_z = (zr[1] - zr[0]) / D_vol
            x_phys = x_phys + topk_offset[:, :, 0] * voxel_size_x
            y_phys = y_phys + topk_offset[:, :, 1] * voxel_size_y
            z_phys = z_phys + topk_offset[:, :, 2] * voxel_size_z

            mu = torch.stack([x_phys, y_phys, z_phys], dim=-1)  # (B, K, 3)
        else:
            mu = torch.stack([w_idx.float(), h_idx.float(), d_idx.float()], dim=-1)

        return {
            "mu": mu,                      # (B, K, 3) physical coords
            "logit_alpha": topk_alpha,     # (B, K)
            "log_sigma": topk_log_sigma,   # (B, K)
        }
