"""
Evaluation metrics for LF-3DGS.

Includes greedy bipartite matching, NMS post-processing, and
Gaussian-specific evaluation utilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


# ------------------------------------------------------------------ #
# NMS post-processing for predicted Gaussians
# ------------------------------------------------------------------ #

def nms_gaussians(
    mu: np.ndarray,
    alpha: np.ndarray,
    merge_radius: float = 1.0,
) -> np.ndarray:
    """Non-maximum suppression on predicted Gaussian centres.

    For each Gaussian (sorted by descending alpha), suppress all
    neighbours within *merge_radius* (in the same coordinate space
    as *mu*).

    Args:
        mu: (N, 3) centre positions.
        alpha: (N,) opacity values (sigmoid-space, 0..1).
        merge_radius: suppression radius.

    Returns:
        Boolean keep mask of shape (N,).
    """
    if mu.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(-alpha)  # descending
    keep = np.ones(len(mu), dtype=bool)

    for idx in order:
        if not keep[idx]:
            continue
        # Suppress nearby lower-alpha Gaussians
        dists = np.sqrt(((mu - mu[idx]) ** 2).sum(axis=1))
        suppress = (dists < merge_radius) & (np.arange(len(mu)) != idx)
        keep[suppress] = False

    return keep


# ------------------------------------------------------------------ #
# Greedy bipartite matching
# ------------------------------------------------------------------ #

def match_centers(
    pred: np.ndarray,
    gt: np.ndarray,
    match_radius: float,
) -> Dict[str, float]:
    """Greedy bipartite matching between predicted and GT centers."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.size == 0 and gt.size == 0:
        return _build_metrics(0, 0, 0, 0.0)
    if gt.size == 0:
        return _build_metrics(0, len(pred), 0, 0.0)
    if pred.size == 0:
        return _build_metrics(0, 0, len(gt), 0.0)

    d2 = ((pred[:, None, :] - gt[None, :, :]) ** 2).sum(axis=2)
    pairs = np.argwhere(d2 <= match_radius ** 2)
    if pairs.size == 0:
        return _build_metrics(0, len(pred), len(gt), 0.0)

    dist = np.sqrt(d2[pairs[:, 0], pairs[:, 1]])
    order = np.argsort(dist)
    pairs = pairs[order]
    dist = dist[order]

    used_p, used_g = set(), set()
    matched_dists = []
    for (pi, gi), di in zip(pairs, dist):
        if int(pi) in used_p or int(gi) in used_g:
            continue
        used_p.add(int(pi))
        used_g.add(int(gi))
        matched_dists.append(float(di))

    tp = len(matched_dists)
    fp = len(pred) - len(used_p)
    fn = len(gt) - len(used_g)
    mean_dist = float(np.mean(matched_dists)) if matched_dists else 0.0
    return _build_metrics(tp, fp, fn, mean_dist)


def _build_metrics(tp, fp, fn, mean_dist):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "mean_dist": mean_dist,
        "precision": precision, "recall": recall, "f1": f1,
        "pred_count": float(tp + fp), "gt_count": float(tp + fn),
    }


# ------------------------------------------------------------------ #
# Gaussian evaluation with optional NMS
# ------------------------------------------------------------------ #

def evaluate_gaussians(
    mu: torch.Tensor,
    logit_alpha: torch.Tensor,
    gt_centers_phys: torch.Tensor,
    cam_params,
    alpha_threshold: float = 0.05,
    match_radius_voxel: float = 2.0,
    use_nms: bool = False,
    nms_radius_voxel: float = 1.5,
) -> Dict[str, float]:
    """Evaluate predicted Gaussians against GT particle centres.

    Args:
        mu: (N, 3) physical positions.
        logit_alpha: (N,) logit opacities.
        gt_centers_phys: (M, 3) GT physical positions.
        cam_params: LFCameraParams with physical_to_voxel.
        alpha_threshold: discard Gaussians below this sigmoid(alpha).
        match_radius_voxel: matching radius in voxel space.
        use_nms: if True, apply NMS before matching.
        nms_radius_voxel: NMS suppression radius in voxel space.
    """
    alpha = torch.sigmoid(logit_alpha).detach().cpu()
    mu_cpu = mu.detach().cpu()
    keep = alpha >= alpha_threshold
    mu_active = mu_cpu[keep]
    alpha_active = alpha[keep].numpy()

    if mu_active.shape[0] > 0:
        pred_voxel = cam_params.physical_to_voxel(mu_active).numpy()
    else:
        pred_voxel = np.empty((0, 3), dtype=np.float32)

    # Optional NMS in voxel space
    if use_nms and pred_voxel.shape[0] > 1:
        nms_keep = nms_gaussians(pred_voxel, alpha_active,
                                 merge_radius=nms_radius_voxel)
        pred_voxel = pred_voxel[nms_keep]

    if gt_centers_phys.shape[0] > 0:
        gt_voxel = cam_params.physical_to_voxel(gt_centers_phys.cpu()).numpy()
    else:
        gt_voxel = np.empty((0, 3), dtype=np.float32)

    result = match_centers(pred_voxel, gt_voxel, match_radius=match_radius_voxel)
    result["n_active"] = float(pred_voxel.shape[0])
    result["n_total"] = float(mu_cpu.shape[0])
    return result
