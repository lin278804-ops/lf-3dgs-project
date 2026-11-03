"""
Analytic back-projection for Light-Field 3D reconstruction.

Implements tomographic back-projection using the exact differentiable camera
model.  For each voxel in the 3D grid, projects into all SAI views and
accumulates intensity — producing a 3D reconstruction volume.

Two aggregation modes:
  - "mean":  V(x) = mean_k  SAI_k(u_k(x), v_k(x))      (additive)
  - "mart":  V(x) = prod_k  SAI_k(u_k(x), v_k(x))^(1/K) (multiplicative, MART-like)

Peak detection on the resulting volume yields initial particle centre
estimates that can seed 3DGS optimisation.

Usage:
    from lf_3dgs.backproject import backproj_init_centers

    centers = backproj_init_centers(sai_stack, cam_params, device="cuda")
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .camera import DifferentiableLFProjection, LFCameraParams


# ===================================================================== #
# Core back-projection
# ===================================================================== #

def _make_voxel_grid(cam: LFCameraParams, device: torch.device) -> torch.Tensor:
    """Create physical-coordinate grid for all voxel centres.

    Returns:
        points: (D*H*W, 3)  [x, y, z] in mm
    """
    D, H, W = cam.grid_shape
    z_vals = torch.linspace(cam.z_range[0], cam.z_range[1], D, device=device)
    y_vals = torch.linspace(cam.y_range[0], cam.y_range[1], H, device=device)
    x_vals = torch.linspace(cam.x_range[0], cam.x_range[1], W, device=device)

    # meshgrid: z, y, x  →  (D, H, W) each
    gz, gy, gx = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    return points  # (D*H*W, 3)


@torch.no_grad()
def backproject_volume(
    sai_stack: torch.Tensor,
    cam: LFCameraParams,
    device: str | torch.device = "cuda",
    mode: str = "mean",
    view_stride: int = 1,
    chunk_views: int = 32,
) -> torch.Tensor:
    """Analytic back-projection from SAI stack to 3D volume.

    Args:
        sai_stack: (V*U, H_sai, W_sai) normalised SAI images.
        cam: camera parameters.
        device: compute device.
        mode: "mean" or "mart" aggregation.
        view_stride: subsample angular views (1 = all 484).
        chunk_views: process this many views at a time (GPU memory control).

    Returns:
        volume: (D, H, W) back-projected intensity volume.
    """
    dev = torch.device(device)
    D, Hg, Wg = cam.grid_shape
    P = cam.px_per_mla
    H_sai, W_sai = sai_stack.shape[1], sai_stack.shape[2]

    # Build projection module on device
    proj = DifferentiableLFProjection(cam).to(dev)

    # Voxel grid  (D*H*W, 3)
    voxel_pts = _make_voxel_grid(cam, dev)
    N_voxels = voxel_pts.shape[0]

    # Build view indices
    s_all = torch.arange(0, P, view_stride, device=dev)
    t_all = torch.arange(0, P, view_stride, device=dev)
    ss, tt = torch.meshgrid(s_all, t_all, indexing="ij")
    s_flat = ss.reshape(-1)  # (K_total,)
    t_flat = tt.reshape(-1)
    K_total = s_flat.shape[0]

    # SAI on device  — shape (V*U, H_sai, W_sai)
    sai_dev = sai_stack.to(dev).float()
    # Normalise SAI to [0, 1] for MART stability
    sai_max = sai_dev.max().clamp(min=1e-6)
    sai_norm = sai_dev / sai_max

    # Accumulator
    if mode == "mart":
        # log-sum for numerical stability  →  exp at end
        log_accum = torch.zeros(N_voxels, device=dev)
    else:
        accum = torch.zeros(N_voxels, device=dev)

    n_views_used = 0

    for v_start in range(0, K_total, chunk_views):
        v_end = min(v_start + chunk_views, K_total)
        s_chunk = s_flat[v_start:v_end]
        t_chunk = t_flat[v_start:v_end]
        K_chunk = s_chunk.shape[0]

        # Project all voxels into these views  →  (K_chunk, N_voxels)
        u_proj, v_proj = proj(voxel_pts, s_chunk, t_chunk)

        # Flat SAI indices for these views
        flat_idx = s_chunk * P + t_chunk  # (K_chunk,)

        # Bilinear sample from SAI
        for k in range(K_chunk):
            sai_view = sai_norm[flat_idx[k]]  # (H_sai, W_sai)

            # Normalise coords to [-1, 1] for grid_sample
            u_norm = 2.0 * u_proj[k] / (H_sai - 1) - 1.0  # row → y
            v_norm = 2.0 * v_proj[k] / (W_sai - 1) - 1.0  # col → x

            # grid_sample expects (N, C, H, W) and grid (N, H_out, W_out, 2) [x, y]
            grid = torch.stack([v_norm, u_norm], dim=-1)  # (N_voxels, 2)
            grid = grid.reshape(1, 1, N_voxels, 2)
            sai_4d = sai_view.reshape(1, 1, H_sai, W_sai)

            sampled = F.grid_sample(
                sai_4d, grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )  # (1, 1, 1, N_voxels)
            vals = sampled.reshape(N_voxels).clamp(min=0)

            if mode == "mart":
                log_accum += torch.log(vals + 1e-10)
            else:
                accum += vals

            n_views_used += 1

    # Final volume
    if mode == "mart":
        volume = torch.exp(log_accum / max(n_views_used, 1))
    else:
        volume = accum / max(n_views_used, 1)

    volume = volume.reshape(D, Hg, Wg)
    return volume


# ===================================================================== #
# Peak detection
# ===================================================================== #

def find_peaks_3d(
    volume: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 2,
) -> np.ndarray:
    """Detect local maxima in a 3D volume.

    Args:
        volume: (D, H, W) intensity volume.
        threshold: minimum intensity to consider as peak (relative to max).
        min_distance: minimum separation between peaks (voxels).

    Returns:
        peaks: (N, 3) voxel indices  [d, h, w]  (float, sub-voxel via centroid).
    """
    from scipy.ndimage import maximum_filter, label

    # Relative threshold
    v_max = volume.max()
    if v_max < 1e-12:
        return np.empty((0, 3), dtype=np.float32)

    abs_thr = threshold * v_max

    # Local maxima detection
    size = 2 * min_distance + 1
    local_max = maximum_filter(volume, size=size)
    is_peak = (volume == local_max) & (volume >= abs_thr)

    # Label connected components and find centroids
    labelled, n_features = label(is_peak)
    if n_features == 0:
        return np.empty((0, 3), dtype=np.float32)

    from scipy.ndimage import center_of_mass
    centroids = center_of_mass(volume, labelled, range(1, n_features + 1))
    peaks = np.array(centroids, dtype=np.float32)  # (N, 3) [d, h, w]
    return peaks


def voxel_to_physical(
    voxel_coords: np.ndarray,
    cam: LFCameraParams,
) -> np.ndarray:
    """Convert voxel indices [d, h, w] to physical coordinates [x, y, z] (mm).

    Args:
        voxel_coords: (N, 3)  [d_idx, h_idx, w_idx]
        cam: camera parameters.
    Returns:
        physical: (N, 3)  [x, y, z] in mm.
    """
    D, H, W = cam.grid_shape
    d = voxel_coords[:, 0]
    h = voxel_coords[:, 1]
    w = voxel_coords[:, 2]

    x = cam.x_range[0] + w / max(W - 1, 1) * (cam.x_range[1] - cam.x_range[0])
    y = cam.y_range[0] + h / max(H - 1, 1) * (cam.y_range[1] - cam.y_range[0])
    z = cam.z_range[0] + d / max(D - 1, 1) * (cam.z_range[1] - cam.z_range[0])

    return np.stack([x, y, z], axis=1).astype(np.float32)


# ===================================================================== #
# High-level interface
# ===================================================================== #

@torch.no_grad()
def backproj_init_centers(
    sai_stack: torch.Tensor,
    cam: LFCameraParams,
    device: str | torch.device = "cuda",
    mode: str = "mean",
    view_stride: int = 2,
    peak_threshold: float = 0.15,
    peak_min_distance: int = 2,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Back-project SAI to 3D volume, detect peaks, return centres.

    Args:
        sai_stack: (V*U, H, W) SAI images.
        cam: camera parameters.
        device: compute device.
        mode: "mean" or "mart" aggregation.
        view_stride: angular subsampling (2 = use every other view).
        peak_threshold: relative threshold for peak detection.
        peak_min_distance: minimum distance between peaks (voxels).
        verbose: print progress info.

    Returns:
        centers: (N, 3) physical coordinates [x, y, z] as torch.Tensor.
        volume: (D, H, W) the back-projected volume as torch.Tensor.
    """
    if verbose:
        print(f"[BackProj] Starting back-projection (mode={mode}, stride={view_stride})...")

    volume = backproject_volume(
        sai_stack, cam, device=device, mode=mode,
        view_stride=view_stride, chunk_views=64,
    )

    vol_np = volume.cpu().numpy()

    if verbose:
        print(f"[BackProj] Volume range: [{vol_np.min():.4f}, {vol_np.max():.4f}]")

    # Peak detection on CPU
    peaks_voxel = find_peaks_3d(
        vol_np,
        threshold=peak_threshold,
        min_distance=peak_min_distance,
    )

    if peaks_voxel.shape[0] == 0:
        if verbose:
            print("[BackProj] No peaks found! Falling back to random init.")
        return torch.empty(0, 3), volume

    # Convert to physical coordinates
    centers_phys = voxel_to_physical(peaks_voxel, cam)

    if verbose:
        print(f"[BackProj] Detected {centers_phys.shape[0]} peaks")

    centers_t = torch.from_numpy(centers_phys).float()
    return centers_t, volume


# ===================================================================== #
# CLI for standalone testing
# ===================================================================== #

if __name__ == "__main__":
    import argparse
    from .dataset import load_single_sample

    parser = argparse.ArgumentParser(description="Back-projection test")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", default="mean", choices=["mean", "mart"])
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    sai, gt_c, meta = load_single_sample(args.sample)
    cam = LFCameraParams()
    if "x_range" in meta:
        cam.x_range = meta["x_range"]
    if "y_range" in meta:
        cam.y_range = meta["y_range"]
    if "z_range" in meta:
        cam.z_range = meta["z_range"]
    if "grid_shape" in meta:
        cam.grid_shape = meta["grid_shape"]

    centers, vol = backproj_init_centers(
        sai, cam, device=args.device,
        mode=args.mode, view_stride=args.stride,
        peak_threshold=args.threshold,
    )
    print(f"Found {centers.shape[0]} centres (GT: {gt_c.shape[0]})")
    print(f"Volume shape: {vol.shape}")
