"""
Differentiable Light-Field Gaussian Splatting Renderer.

Core idea:
  For each sampled view (s, t), project every 3D Gaussian to a 2D Gaussian on
  the SAI spatial grid (num_mla_y x num_mla_x), then splat with additive blending.

The splatting exploits the **separability** of isotropic 2D Gaussians:
  I(u,v) = sum_i  alpha_i * exp(-(u-u_i)^2/(2*sig^2)) * exp(-(v-v_i)^2/(2*sig^2))
         = (alpha * g_u)^T  @  g_v          <-- matrix multiply  (H,N) @ (N,W) = (H,W)

This avoids O(N*H*W) memory and makes the backward pass efficient.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import DifferentiableLFProjection, LFCameraParams


class GaussianSplatRenderer(nn.Module):
    """
    Differentiable renderer:  3D Gaussians + view indices  -->  rendered SAI images.

    Workflow per optimisation step:
      1. Sample K views from (s, t) ∈ [0, px_per_mla)^2
      2. Project all N Gaussians to each view -> (K, N) continuous (u, v)
      3. Evaluate separable 2D Gaussian -> splat to (K, H, W) images
      4. Return rendered images + sampled view indices
    """

    def __init__(self, cam_params: LFCameraParams) -> None:
        super().__init__()
        self.cam = cam_params
        self.proj = DifferentiableLFProjection(cam_params)
        self.H = cam_params.num_mla_y   # 186
        self.W = cam_params.num_mla_x   # 232

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        mu: torch.Tensor,              # (N, 3)  positions [x, y, z] mm
        log_sigma: torch.Tensor,        # (N,)    log 2D rendering sigma (pixels)
        logit_alpha: torch.Tensor,      # (N,)    logit intensity
        s_idx: torch.Tensor,            # (K,)    sampled angular row indices
        t_idx: torch.Tensor,            # (K,)    sampled angular col indices
        downsample: int = 1,
    ) -> torch.Tensor:
        """
        Render K SAI views.

        Returns:
            images: (K, H', W')  where H'=H//ds, W'=W//ds
        """
        N = mu.shape[0]
        K = s_idx.shape[0]
        ds = max(1, int(downsample))
        H_r = self.H // ds
        W_r = self.W // ds

        if N == 0:
            return torch.zeros(K, H_r, W_r, device=mu.device)

        # 1. Project all Gaussians to all K views
        u_proj, v_proj = self.proj(mu, s_idx, t_idx)    # (K, N)

        if ds > 1:
            u_proj = u_proj / ds
            v_proj = v_proj / ds

        # 2. Gaussian parameters
        sigma = torch.exp(log_sigma).clamp(min=0.2)     # (N,)
        alpha = torch.sigmoid(logit_alpha)               # (N,)

        # 3. Splat each view
        images = self._splat_separable(u_proj, v_proj, sigma, alpha, H_r, W_r)
        return images   # (K, H_r, W_r)

    def render_all_views(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        logit_alpha: torch.Tensor,
        batch_size: int = 32,
        downsample: int = 1,
    ) -> torch.Tensor:
        """Render all px_per_mla^2 views. Returns (V*U, H', W')."""
        P = self.cam.px_per_mla
        device = mu.device
        all_images = []
        for start in range(0, P * P, batch_size):
            end = min(start + batch_size, P * P)
            flat = torch.arange(start, end, device=device)
            s = flat // P
            t = flat % P
            imgs = self.forward(mu, log_sigma, logit_alpha, s, t, downsample)
            all_images.append(imgs)
        return torch.cat(all_images, dim=0)

    # ------------------------------------------------------------------ #
    # Separable splatting
    # ------------------------------------------------------------------ #
    def _splat_separable(
        self,
        u_proj: torch.Tensor,   # (K, N)
        v_proj: torch.Tensor,   # (K, N)
        sigma: torch.Tensor,    # (N,)
        alpha: torch.Tensor,    # (N,)
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Separable Gaussian splatting via matrix multiply.

        I_k(u,v) = sum_i alpha_i * g_u(u, u_i, sigma_i) * g_v(v, v_i, sigma_i)

        Equivalent to:  I_k = (alpha * G_u_k)^T @ G_v_k
        where G_u_k has shape (N, H), G_v_k has shape (N, W).

        Total compute per view:  H*N + N*W  (Gaussian eval)  +  H*N*W  (matmul)
        """
        K, N = u_proj.shape
        device = u_proj.device

        # Pixel grids
        u_grid = torch.arange(H, device=device, dtype=torch.float32)  # (H,)
        v_grid = torch.arange(W, device=device, dtype=torch.float32)  # (W,)

        # sigma: (N,) -> (1, N, 1) for broadcasting
        inv_2sig2 = 1.0 / (2.0 * sigma * sigma + 1e-8)   # (N,)

        images = []
        for k in range(K):
            u_k = u_proj[k]   # (N,)
            v_k = v_proj[k]   # (N,)

            # Gaussian values along rows: (N, H)
            du = u_grid.unsqueeze(0) - u_k.unsqueeze(1)       # (N, H)
            g_u = torch.exp(-du * du * inv_2sig2.unsqueeze(1)) # (N, H)

            # Gaussian values along cols: (N, W)
            dv = v_grid.unsqueeze(0) - v_k.unsqueeze(1)       # (N, W)
            g_v = torch.exp(-dv * dv * inv_2sig2.unsqueeze(1)) # (N, W)

            # Weighted: (N, H)  with alpha
            ag_u = g_u * alpha.unsqueeze(1)                    # (N, H)

            # Image = (H, N) @ (N, W) = (H, W)
            img = ag_u.T @ g_v                                 # (H, W)
            images.append(img)

        return torch.stack(images, dim=0)   # (K, H, W)

    # ------------------------------------------------------------------ #
    # Utility: sample random views
    # ------------------------------------------------------------------ #
    @staticmethod
    def sample_views(K: int, px_per_mla: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample K random sub-aperture views."""
        s = torch.randint(0, px_per_mla, (K,), device=device)
        t = torch.randint(0, px_per_mla, (K,), device=device)
        return s, t

    @staticmethod
    def sai_index(s: torch.Tensor, t: torch.Tensor, px_per_mla: int) -> torch.Tensor:
        """Convert (s, t) pairs to flat sai_stack indices: idx = s * U + t."""
        return s * px_per_mla + t
