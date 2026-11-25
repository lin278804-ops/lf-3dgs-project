"""Dataset utilities for LF-3DGS."""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class LF3DGSDataset(Dataset):
    """Loads .npz samples produced by generate_dataset.py."""
    def __init__(self, root_dir: str, split: Optional[str] = None,
                 normalize: bool = True, limit: Optional[int] = None,
                 intensity_scale: Optional[float] = None) -> None:
        data_dir = os.path.join(root_dir, split) if split else root_dir
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir)
        files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz"))
        if not files:
            raise FileNotFoundError(data_dir)
        if limit is not None:
            files = files[:int(limit)]
        self.files = files
        self.normalize = normalize
        self.intensity_scale = intensity_scale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        with np.load(fp, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        sai_stack = normalize_image(data["sai_stack"], data, self.normalize, self.intensity_scale)
        if "particle_centers" in data:
            centers = data["particle_centers"].astype(np.float32)
        else:
            centers = np.empty((0, 3), dtype=np.float32)
        meta = {}
        for key in ("particle_diameter_um", "ppm_concentration"):
            if key in data:
                meta[key] = float(data[key])
        for key in ("x_range", "y_range", "z_range"):
            if key in data:
                meta[key] = tuple(data[key].astype(float))
        if "grid_shape" in data:
            meta["grid_shape"] = tuple(int(x) for x in data["grid_shape"])
        return sai_stack, centers, meta, fp


def _extract_centers_from_grid(grid, meta):
    """Extract particle centres from a binary voxel grid.
    Converts voxel indices to physical coordinates using the ranges in meta.
    Uses connected-component centroids when scipy is available.
    """
    if grid.max() == 0:
        return np.empty((0, 3), dtype=np.float32)
    from .camera import LFCameraParams
    cam = LFCameraParams()
    xr = meta.get("x_range", cam.x_range)
    yr = meta.get("y_range", cam.y_range)
    zr = meta.get("z_range", cam.z_range)
    gs = meta.get("grid_shape", (grid.shape[0], grid.shape[1], grid.shape[2]))
    try:
        from scipy import ndimage
        labelled, n_obj = ndimage.label(grid > 0)
        if n_obj == 0:
            return np.empty((0, 3), dtype=np.float32)
        coms = ndimage.center_of_mass(grid, labelled, range(1, n_obj + 1))
        voxel_coords = np.array(coms, dtype=np.float32)
    except ImportError:
        nz = np.argwhere(grid > 0).astype(np.float32)
        voxel_coords = nz
    Dz, Dy, Dx = int(gs[0]), int(gs[1]), int(gs[2])
    physical = np.zeros_like(voxel_coords)
    physical[:, 0] = xr[0] + (voxel_coords[:, 2] + 0.5) / Dx * (xr[1] - xr[0])
    physical[:, 1] = yr[0] + (voxel_coords[:, 1] + 0.5) / Dy * (yr[1] - yr[0])
    physical[:, 2] = zr[0] + (voxel_coords[:, 0] + 0.5) / Dz * (zr[1] - zr[0])
    return physical


def normalize_image(image, metadata, normalize=True, intensity_scale=None):
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Non-finite image")
    if normalize:
        scale = intensity_scale if intensity_scale is not None else metadata.get("intensity_scale")
        if scale is None:
            raise ValueError("Missing intensity_scale; legacy 0–255 data require explicit intensity_scale=255")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("intensity_scale must be finite and positive")
        image = image / scale
    return image


def load_single_sample(path, normalize=True, intensity_scale=None, load_labels=True):
    """Load one .npz and return tensors ready for optimisation."""
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files
                if load_labels or k not in ("particle_centers", "particle_grid_gt")}
    sai = normalize_image(data["sai_stack"], data, normalize, intensity_scale)
    sai_t = torch.from_numpy(sai)
    meta = {}
    for key in ("particle_diameter_um", "ppm_concentration"):
        if key in data:
            meta[key] = float(data[key])
    for key in ("x_range", "y_range", "z_range"):
        if key in data:
            meta[key] = tuple(data[key].astype(float))
    if "grid_shape" in data:
        meta["grid_shape"] = tuple(int(x) for x in data["grid_shape"])
    if "particle_centers" in data:
        centers = data["particle_centers"].astype(np.float32)
    elif "particle_grid_gt" in data:
        centers = _extract_centers_from_grid(data["particle_grid_gt"], meta)
    else:
        centers = np.empty((0, 3), dtype=np.float32)
    centers_t = torch.from_numpy(centers)
    return sai_t, centers_t, meta
