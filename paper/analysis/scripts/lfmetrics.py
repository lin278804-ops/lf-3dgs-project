"""Detection scoring used throughout the project, reimplemented for re-analysis.

Mirrors `hungarian_match` in the archived evaluator: one-to-one Hungarian
assignment in anisotropic voxel coordinates with a two-voxel gate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

X_RANGE = (-10.0, 10.0)
Y_RANGE = (-8.0, 8.0)
Z_RANGE = (-5.0, 5.0)
GRID = (64, 128, 128)  # depth, rows (y), columns (x)
GATE_VOX = 2.0


def to_voxel(xyz: np.ndarray) -> np.ndarray:
    """(N, 3) physical [x, y, z] in mm -> (N, 3) voxel [z, y, x] indices."""
    xyz = np.asarray(xyz, dtype=np.float64)
    D, H, W = GRID
    x = (xyz[:, 0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * W
    y = (xyz[:, 1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * H
    z = (xyz[:, 2] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]) * D
    return np.stack([z, y, x], axis=1)


def match(pred_xyz: np.ndarray, gt_xyz: np.ndarray, gate: float = GATE_VOX) -> dict:
    """Return matched index pairs, misses and false positives."""
    pv, gv = to_voxel(pred_xyz).astype(np.float32), to_voxel(gt_xyz).astype(np.float32)
    d = np.sqrt(((pv[:, None, :] - gv[None, :, :]) ** 2).sum(axis=2))
    cost = d.copy()
    cost[d > gate] = 1e6
    ri, ci = linear_sum_assignment(cost)
    keep = d[ri, ci] <= gate
    ri, ci = ri[keep], ci[keep]
    fp = np.setdiff1d(np.arange(len(pred_xyz)), ri)
    fn = np.setdiff1d(np.arange(len(gt_xyz)), ci)
    tp = len(ri)
    p = tp / max(tp + len(fp), 1)
    r = tp / max(tp + len(fn), 1)
    return {
        "pred_idx": ri, "gt_idx": ci, "fp_idx": fp, "fn_idx": fn,
        "tp": tp, "fp": len(fp), "fn": len(fn),
        "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
        "error_mm": np.asarray(pred_xyz)[ri] - np.asarray(gt_xyz)[ci],
    }
