"""
Loss functions for LF-3DGS.

Includes:
  - Charbonnier (smooth L1) reconstruction loss
  - L1 reconstruction loss
  - Weighted L1 reconstruction loss (signal-aware)
  - Sparsity regularisation on Gaussian opacity
  - Alpha entropy regularisation
  - Scale regularisation on Gaussian sigma
  - Chamfer-like center matching loss (for Stage 2)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reconstruction losses
# --------------------------------------------------------------------------- #

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier (smooth-L1) loss:  mean( sqrt((pred-target)^2 + eps^2) )."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    signal_weight: float = 10.0,
    signal_thr: float = 0.01,
) -> torch.Tensor:
    """Signal-aware L1: up-weight pixels where GT has signal.

    In sparse SAI images (>95% zeros), plain L1 dilutes gradient from
    particle pixels.  This loss gives *signal_weight* times more weight
    to pixels where ``target > signal_thr``.
    """
    weight = torch.where(target > signal_thr, signal_weight, 1.0)
    return (weight * (pred - target).abs()).mean()


# --------------------------------------------------------------------------- #
# Regularisation losses
# --------------------------------------------------------------------------- #

def sparsity_loss(logit_alpha: torch.Tensor) -> torch.Tensor:
    """Encourage most Gaussians to have low opacity  -->  drives unused ones to 0."""
    return torch.mean(torch.sigmoid(logit_alpha))


def alpha_entropy_loss(logit_alpha: torch.Tensor) -> torch.Tensor:
    """Binary-entropy regularisation: push alpha toward 0 or 1 (not middle).

    H(p) = -[p log p + (1-p) log(1-p)]  is maximised at p=0.5.
    Minimising -H encourages binary decisions.
    """
    p = torch.sigmoid(logit_alpha)
    entropy = -(p * torch.log(p + 1e-8) + (1.0 - p) * torch.log(1.0 - p + 1e-8))
    # We want to *minimise* entropy (push to 0/1), so return mean entropy.
    return torch.mean(entropy)


def scale_loss(log_sigma: torch.Tensor) -> torch.Tensor:
    """Penalise large Gaussian scales to prevent blobs from growing unbounded."""
    return torch.mean(torch.exp(log_sigma))


# --------------------------------------------------------------------------- #
# Center matching loss (Stage 2 / optional Stage 1)
# --------------------------------------------------------------------------- #

def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer distance between two point sets.

    Args:
        pred: (M, 3)   predicted centres
        gt:   (N, 3)   ground-truth centres
    Returns:
        Scalar loss = (mean min-dist pred->gt) + (mean min-dist gt->pred)
    """
    if pred.shape[0] == 0 and gt.shape[0] == 0:
        return pred.new_tensor(0.0)
    if pred.shape[0] == 0:
        return pred.new_tensor(0.0)
    if gt.shape[0] == 0:
        return pred.new_tensor(0.0)
    d2 = torch.cdist(pred, gt, p=2.0)       # (M, N)
    d_pred_to_gt = d2.min(dim=1).values      # (M,)
    d_gt_to_pred = d2.min(dim=0).values      # (N,)
    return d_pred_to_gt.mean() + d_gt_to_pred.mean()


# --------------------------------------------------------------------------- #
# Hungarian matching loss (for alpha training stabilisation)
# --------------------------------------------------------------------------- #

def hungarian_matching_loss(
    mu_pred: torch.Tensor,
    logit_alpha: torch.Tensor,
    mu_gt: torch.Tensor,
    match_radius: float = 5.0,
    alpha_target_matched: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """1-to-1 matching loss using the Hungarian algorithm.

    Provides direct particle-level supervision:
      - **position_loss**: mean distance of matched (pred, gt) pairs.
      - **alpha_loss**: BCE encouraging matched Gaussians → alpha=1,
        unmatched Gaussians → alpha=0.

    Args:
        mu_pred: (N, 3) predicted centres (physical coords).
        logit_alpha: (N,) logit alpha values.
        mu_gt: (M, 3) GT centres (physical coords).
        match_radius: max distance for a valid match (mm).
        alpha_target_matched: target alpha for matched Gaussians.
    Returns:
        dict with keys "position", "alpha", "n_matched".
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    N = mu_pred.shape[0]
    M = mu_gt.shape[0]
    device = mu_pred.device
    zero = mu_pred.new_tensor(0.0)

    if N == 0 or M == 0:
        return {"position": zero, "alpha": zero, "n_matched": 0}

    # --- Distance matrix (detach for assignment, keep grad for loss) ---
    with torch.no_grad():
        dist_mat = torch.cdist(mu_pred.float(), mu_gt.float(), p=2.0)  # (N, M)
        cost_full = dist_mat.cpu().numpy()

    # ---- Sparse pre-filter: only keep pred rows that are within radius of
    # at least one GT.  This reduces the assignment problem from (N, M) to
    # (n_candidates, M) where n_candidates << N, making scipy feasible for
    # large N (e.g. topk=3500 vs M=2000 GT particles).
    has_close = (cost_full < match_radius).any(axis=1)   # (N,) bool
    pred_rows = np.where(has_close)[0]                    # indices into N

    if len(pred_rows) == 0:
        # No prediction is close to any GT particle — alpha loss only
        alpha_target = torch.zeros(N, device=device)
        alpha_loss = F.binary_cross_entropy_with_logits(logit_alpha, alpha_target)
        return {"position": zero, "alpha": alpha_loss, "n_matched": 0}

    cost = cost_full[np.ix_(pred_rows, range(M))].copy()
    cost[cost > match_radius] = 1e6

    row_idx_local, col_idx = linear_sum_assignment(cost)

    # Map local row indices back to global pred indices
    row_idx = pred_rows[row_idx_local]

    # Filter out matches beyond radius
    valid = cost[row_idx_local, col_idx] < match_radius
    row_matched = row_idx[valid]
    col_matched = col_idx[valid]
    n_matched = int(valid.sum())

    # --- Position loss ---
    if n_matched > 0:
        pred_matched = mu_pred[row_matched]   # (n_matched, 3)
        gt_matched = mu_gt[col_matched]       # (n_matched, 3)
        pos_loss = torch.sqrt(((pred_matched - gt_matched) ** 2).sum(dim=1) + 1e-8).mean()
    else:
        pos_loss = zero

    # --- Alpha supervision ---
    # All Gaussians: default target = 0 (unmatched)
    alpha_target = torch.zeros(N, device=device)
    if n_matched > 0:
        alpha_target[row_matched] = alpha_target_matched

    alpha_loss = F.binary_cross_entropy_with_logits(logit_alpha, alpha_target)

    return {"position": pos_loss, "alpha": alpha_loss, "n_matched": n_matched}


def volume_peak_hardneg_loss(
    alpha_logits: torch.Tensor,
    gt_centers: torch.Tensor,
    cam,
    match_radius_vox: float = 2.0,
    topk_neg: int = 512,
    target_logit: float | None = None,
) -> torch.Tensor:
    """Push occupancy on local-max voxels that are not near any GT.

    Unlike Hungarian on extracted top-K, this supervises *all* 3x3x3 peaks
    in the occupancy volume so extra ghosts cannot hide below the extract
    cutoff. Official eval uses the same r=2.0 voxel radius.
    """
    if gt_centers is None or gt_centers.numel() == 0:
        return alpha_logits.new_tensor(0.0)

    logits = alpha_logits.float()
    pooled = F.max_pool3d(
        logits.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0)
    is_peak = logits >= (pooled - 1e-6)
    coords = is_peak.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return logits.new_tensor(0.0)

    peak_vox = coords.float() + 0.5  # voxel-center convention (z, y, x)
    gt_v = cam.physical_to_voxel(gt_centers.to(logits.device).float())
    dist = torch.cdist(peak_vox, gt_v, p=2.0)
    min_d = dist.min(dim=1).values
    neg = min_d > float(match_radius_vox)
    if not bool(neg.any()):
        return logits.new_tensor(0.0)

    neg_logits = logits[coords[neg, 0], coords[neg, 1], coords[neg, 2]]
    k = min(int(topk_neg), int(neg_logits.numel()))
    hard = torch.topk(neg_logits, k, dim=0).values
    # Official α=0.05 ↔ logit ≈ −2.944. Softplus margin focuses on peaks
    # that would survive the frozen decode threshold.
    if target_logit is not None:
        return F.softplus(hard - float(target_logit)).mean()
    return F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))


# --------------------------------------------------------------------------- #
# Composite total loss
# --------------------------------------------------------------------------- #

def total_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    logit_alpha: torch.Tensor,
    log_sigma: torch.Tensor,
    cfg: Dict[str, float],
    mu_pred: torch.Tensor | None = None,
    mu_gt: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted combination of all losses.

    cfg keys:
        recon_type:      "charbonnier" | "l1" | "mse" | "weighted_l1"
        recon_weight:    float (default 1.0)
        sparse_weight:   float (default 0.01)
        scale_weight:    float (default 0.001)
        entropy_weight:  float (default 0.0)
        signal_weight:   float (default 10.0)  -- for weighted_l1
        signal_thr:      float (default 0.01)  -- for weighted_l1
        hungarian_pos_weight:   float (default 0.0) -- Hungarian position loss
        hungarian_alpha_weight: float (default 0.0) -- Hungarian alpha BCE loss
        hungarian_match_radius: float (default 5.0) -- max match distance (mm)
    """
    rt = str(cfg.get("recon_type", "charbonnier"))
    if rt == "l1":
        l_recon = l1_loss(rendered, gt)
    elif rt == "mse":
        l_recon = mse_loss(rendered, gt)
    elif rt == "weighted_l1":
        l_recon = weighted_l1_loss(
            rendered, gt,
            signal_weight=float(cfg.get("signal_weight", 10.0)),
            signal_thr=float(cfg.get("signal_thr", 0.01)),
        )
    else:
        l_recon = charbonnier_loss(rendered, gt)

    l_sparse = sparsity_loss(logit_alpha)
    l_scale = scale_loss(log_sigma)
    l_entropy = alpha_entropy_loss(logit_alpha)

    total = (
        float(cfg.get("recon_weight", 1.0)) * l_recon
        + float(cfg.get("sparse_weight", 0.01)) * l_sparse
        + float(cfg.get("scale_weight", 0.001)) * l_scale
        + float(cfg.get("entropy_weight", 0.0)) * l_entropy
    )

    result = {
        "total": total,
        "recon": l_recon,
        "sparse": l_sparse,
        "scale": l_scale,
        "entropy": l_entropy,
    }

    # Hungarian matching loss (optional)
    hung_pos_w = float(cfg.get("hungarian_pos_weight", 0.0))
    hung_alpha_w = float(cfg.get("hungarian_alpha_weight", 0.0))
    if (hung_pos_w > 0 or hung_alpha_w > 0) and mu_pred is not None and mu_gt is not None:
        hung = hungarian_matching_loss(
            mu_pred, logit_alpha, mu_gt,
            match_radius=float(cfg.get("hungarian_match_radius", 5.0)),
        )
        hung_total = hung_pos_w * hung["position"] + hung_alpha_w * hung["alpha"]
        result["total"] = result["total"] + hung_total
        result["hungarian_pos"] = hung["position"]
        result["hungarian_alpha"] = hung["alpha"]
        result["hungarian_n_matched"] = hung["n_matched"]

    return result
