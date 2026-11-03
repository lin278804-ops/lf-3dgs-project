"""Cross-frame particle tracking for LF-PIV.

Matches detected particles between consecutive frames and computes
displacement / velocity vectors.

Algorithms:
  - Hungarian (optimal bipartite matching, scipy.optimize.linear_sum_assignment)
  - Nearest-neighbour (fast greedy fallback)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def hungarian_tracking(
    centers_t1: np.ndarray,
    centers_t2: np.ndarray,
    max_disp: Optional[float] = None,
) -> Dict:
    """Optimal bipartite matching between two frames.

    Args:
        centers_t1: (N, 3) positions at frame 1
        centers_t2: (M, 3) positions at frame 2
        max_disp:   maximum allowed displacement (mm); unmatched beyond this

    Returns:
        dict with matched indices, displacements, and statistics.
    """
    from scipy.optimize import linear_sum_assignment

    N, M = len(centers_t1), len(centers_t2)
    if N == 0 or M == 0:
        return _empty_result(N, M)

    diff = centers_t1[:, None, :] - centers_t2[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=2))

    BIG = 1e9
    if max_disp is not None:
        cost_masked = cost.copy()
        cost_masked[cost > max_disp] = BIG
    else:
        cost_masked = cost

    row_ind, col_ind = linear_sum_assignment(cost_masked)

    valid = cost_masked[row_ind, col_ind] < BIG * 0.5
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    displacements = centers_t2[col_ind] - centers_t1[row_ind]
    dists = np.sqrt((displacements ** 2).sum(axis=1))

    return {
        "idx_t1": row_ind,
        "idx_t2": col_ind,
        "displacements": displacements,
        "match_dists": dists,
        "n_matched": int(len(row_ind)),
        "n_t1": N,
        "n_t2": M,
        "match_rate_t1": float(len(row_ind) / max(N, 1)),
        "match_rate_t2": float(len(row_ind) / max(M, 1)),
    }


def nearest_neighbor_tracking(
    centers_t1: np.ndarray,
    centers_t2: np.ndarray,
    max_disp: Optional[float] = None,
) -> Dict:
    """Greedy nearest-neighbour matching (faster but suboptimal)."""
    N, M = len(centers_t1), len(centers_t2)
    if N == 0 or M == 0:
        return _empty_result(N, M)

    diff = centers_t1[:, None, :] - centers_t2[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=2))

    used_t2 = set()
    idx_t1, idx_t2 = [], []
    for i in np.argsort(cost.min(axis=1)):
        j = int(np.argmin(cost[i]))
        if j in used_t2:
            continue
        d = cost[i, j]
        if max_disp is not None and d > max_disp:
            continue
        idx_t1.append(i)
        idx_t2.append(j)
        used_t2.add(j)

    idx_t1 = np.array(idx_t1, dtype=int)
    idx_t2 = np.array(idx_t2, dtype=int)
    displacements = centers_t2[idx_t2] - centers_t1[idx_t1] if len(idx_t1) else np.empty((0, 3))
    dists = np.sqrt((displacements ** 2).sum(axis=1)) if len(displacements) else np.empty(0)

    return {
        "idx_t1": idx_t1,
        "idx_t2": idx_t2,
        "displacements": displacements,
        "match_dists": dists,
        "n_matched": int(len(idx_t1)),
        "n_t1": N,
        "n_t2": M,
        "match_rate_t1": float(len(idx_t1) / max(N, 1)),
        "match_rate_t2": float(len(idx_t2) / max(M, 1)),
    }


def compute_velocity(displacements: np.ndarray, dt: float) -> np.ndarray:
    """Convert displacement vectors to velocity vectors (mm/s)."""
    return displacements / dt


def _empty_result(n1: int, n2: int) -> Dict:
    return {
        "idx_t1": np.empty(0, dtype=int),
        "idx_t2": np.empty(0, dtype=int),
        "displacements": np.empty((0, 3)),
        "match_dists": np.empty(0),
        "n_matched": 0, "n_t1": n1, "n_t2": n2,
        "match_rate_t1": 0.0, "match_rate_t2": 0.0,
    }
