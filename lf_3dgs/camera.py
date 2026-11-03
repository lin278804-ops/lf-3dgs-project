"""
Differentiable Light-Field Camera Model for 3D Gaussian Splatting.

Encapsulates the analytic projection from 3D world coordinates to continuous
sub-aperture image (SAI) pixel coordinates. Derived from the ray-tracing optics
in simulate_lightfield.py but expressed as differentiable PyTorch operations.

Optical Pipeline (unfocused plenoptic camera):
  Object(x,y,z) --> Main Lens --> MLA --> CCD --> SAI(s,t,u,v)

Key coordinate convention:
  - Physical: (x, y, z) in mm, origin at object plane
  - SAI spatial: (u, v) = (row, col) = (num_mla_y, num_mla_x) indices
  - SAI angular: (s, t) = (view_row, view_col) = pixel-within-microlens indices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class LFCameraParams:
    """Light field camera parameters matching the simulation."""

    f_main: float = 100.0          # Main lens focal length (mm)
    u_obj: float = 200.0           # Object distance (mm)
    f_mla: float = 0.8             # MLA focal length (mm)
    pitch: float = 0.099           # Microlens pitch / aperture (mm)
    pixel_pitch: float = 0.0045    # CCD pixel pitch (mm)
    num_mla_x: int = 232           # Number of microlenses, horizontal (cols)
    num_mla_y: int = 186           # Number of microlenses, vertical (rows)
    px_per_mla: int = 22           # Pixels per microlens (angular resolution per axis)

    # Physical volume ranges (mm) — for coordinate transforms
    x_range: Tuple[float, float] = (-10.0, 10.0)
    y_range: Tuple[float, float] = (-8.0, 8.0)
    z_range: Tuple[float, float] = (-5.0, 5.0)

    # Voxel grid shape (D, H, W) — for evaluation coordinate conversion
    grid_shape: Tuple[int, int, int] = (64, 128, 128)

    @property
    def v_main(self) -> float:
        """Image distance via thin-lens equation."""
        return 1.0 / (1.0 / self.f_main - 1.0 / self.u_obj)

    @property
    def magnification(self) -> float:
        return self.v_main / self.u_obj

    @property
    def effective_mla_width(self) -> float:
        return self.num_mla_x * self.pitch

    @property
    def effective_mla_height(self) -> float:
        return self.num_mla_y * self.pitch

    @property
    def num_views(self) -> int:
        return self.px_per_mla * self.px_per_mla

    @classmethod
    def from_dict(cls, d: Dict) -> "LFCameraParams":
        """Build from the camera_params dict used in generate_dataset.py."""
        pitch = d.get("microlens_aperture_mm", 0.099)
        pixel_pitch = d.get("ccd_pixel_pitch_mm", 0.0045)
        px = int(round(pitch / pixel_pitch))
        return cls(
            f_main=d.get("main_lens_focal_length_mm", 100.0),
            u_obj=d.get("object_distance_mm", 200.0),
            f_mla=d.get("mla_focal_length_mm", 0.8),
            pitch=pitch,
            pixel_pitch=pixel_pitch,
            num_mla_x=d.get("num_mla_x", 232),
            num_mla_y=d.get("num_mla_y", 186),
            px_per_mla=px,
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def physical_to_voxel(self, centers: torch.Tensor) -> torch.Tensor:
        """Convert (N, 3) physical [x, y, z] -> (N, 3) voxel [z_idx, y_idx, x_idx]."""
        D, H, W = self.grid_shape
        mn_x, mx_x = self.x_range
        mn_y, mx_y = self.y_range
        mn_z, mx_z = self.z_range
        x_idx = (centers[:, 0] - mn_x) / (mx_x - mn_x) * W
        y_idx = (centers[:, 1] - mn_y) / (mx_y - mn_y) * H
        z_idx = (centers[:, 2] - mn_z) / (mx_z - mn_z) * D
        return torch.stack([z_idx, y_idx, x_idx], dim=1)


class DifferentiableLFProjection(nn.Module):
    """
    Differentiable projection: 3D physical coords  -->  continuous SAI spatial coords.

    For each sub-aperture view (s, t) and each 3D point (px, py, pz), computes the
    continuous microlens index (u_cont, v_cont) at which the point appears in that view.

    The derivation follows from the thin-lens + MLA ray-tracing model:
      1. The sub-aperture index determines the local CCD offset within a microlens.
      2. Given that offset and the particle position, solve for the aperture position.
      3. Trace through the main lens to the MLA plane to get the spatial coordinate.
    """

    def __init__(self, params: LFCameraParams) -> None:
        super().__init__()
        self.p = params
        # Register scalars as buffers so they move with .to(device)
        for name, val in [
            ("f_main", params.f_main),
            ("u_obj", params.u_obj),
            ("v_main", params.v_main),
            ("f_mla", params.f_mla),
            ("pitch", params.pitch),
            ("pixel_pitch", params.pixel_pitch),
            ("half_w", params.effective_mla_width / 2.0),
            ("half_h", params.effective_mla_height / 2.0),
            ("half_px", (params.px_per_mla - 1) / 2.0),
        ]:
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32))

    # ------------------------------------------------------------------
    def forward(
        self,
        points: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)  [x, y, z] in mm
            s_idx:  (K,)    angular row indices  (0 .. px_per_mla-1)
            t_idx:  (K,)    angular col indices  (0 .. px_per_mla-1)
        Returns:
            u_proj: (K, N)  continuous spatial **row** index (0-based, num_mla_y)
            v_proj: (K, N)  continuous spatial **col** index (0-based, num_mla_x)
        """
        px = points[:, 0]  # (N,)
        py = points[:, 1]
        pz = points[:, 2]

        # Centered sub-aperture offsets
        s_c = s_idx.float() - self.half_px   # (K,)
        t_c = t_idx.float() - self.half_px

        # Local CCD offset within microlens (mm)
        x_local = (t_c * self.pixel_pitch).unsqueeze(1)   # (K,1) — x direction
        y_local = (s_c * self.pixel_pitch).unsqueeze(1)   # (K,1) — y direction

        px = px.unsqueeze(0)  # (1,N)
        py = py.unsqueeze(0)
        pz = pz.unsqueeze(0)

        d = self.u_obj - pz                                # (1,N)
        A = -1.0 / self.f_main + 1.0 / d                  # (1,N)

        # Solve for main-lens aperture position
        x_ap = (x_local / self.f_mla + px / d) / A        # (K,N)
        y_ap = (y_local / self.f_mla + py / d) / A

        # MLA position
        C = self.v_main * pz / (self.u_obj * d)            # (1,N)
        x_mla = x_ap * C - self.v_main * px / d            # (K,N)
        y_mla = y_ap * C - self.v_main * py / d

        # Continuous spatial indices
        v_proj = (x_mla + self.half_w) / self.pitch         # col
        u_proj = (y_mla + self.half_h) / self.pitch         # row

        return u_proj, v_proj
