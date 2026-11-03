"""PIV evaluation metrics - Phase 5 additions included."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


def compute_velocity_metrics(velocity_pred, velocity_gt_all, matched_idx_t1=None):
    if matched_idx_t1 is not None and len(matched_idx_t1) > 0:
        vgt = velocity_gt_all[matched_idx_t1]
    else:
        vgt = velocity_gt_all
    if len(velocity_pred) == 0:
        return _zero_metrics()
    err = velocity_pred - vgt
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err**2).mean(axis=0))
    err_mag = np.sqrt((err**2).sum(axis=1))
    mae_mag = float(err_mag.mean())
    rmse_mag = float(np.sqrt((err_mag**2).mean()))
    dot = (velocity_pred * vgt).sum(axis=1)
    n_pred = np.sqrt((velocity_pred**2).sum(axis=1)).clip(1e-10)
    n_gt = np.sqrt((vgt**2).sum(axis=1)).clip(1e-10)
    cos_a = np.clip(dot / (n_pred * n_gt), -1, 1)
    angle_err = np.degrees(np.arccos(cos_a))
    pred_mag = np.sqrt((velocity_pred**2).sum(axis=1))
    mu_m, sig_m = pred_mag.mean(), pred_mag.std()
    spurious = np.abs(pred_mag - mu_m) > 3 * max(sig_m, 1e-10)
    gt_speed = n_gt.mean()
    rel_err = mae_mag / max(gt_speed, 1e-10)
    return {
        "mae_vx": float(mae[0]), "mae_vy": float(mae[1]), "mae_vz": float(mae[2]),
        "mae_magnitude": mae_mag,
        "rmse_vx": float(rmse[0]), "rmse_vy": float(rmse[1]), "rmse_vz": float(rmse[2]),
        "rmse_magnitude": rmse_mag,
        "angle_error_mean_deg": float(angle_err.mean()),
        "angle_error_median_deg": float(np.median(angle_err)),
        "angle_error_p95_deg": float(np.percentile(angle_err, 95)),
        "spurious_ratio": float(spurious.mean()),
        "relative_error": rel_err,
        "n_vectors": int(len(velocity_pred)),
    }


def compute_displacement_error_per_axis(velocity_pred, velocity_gt, dt, bins=40):
    """Per-axis displacement error PDF in mm."""
    if len(velocity_pred) == 0:
        return {}
    err_disp = (velocity_pred - velocity_gt) * dt
    result = {}
    for i, ax in enumerate("xyz"):
        e = err_disp[:, i]
        hc, he = np.histogram(e, bins=bins)
        result[ax] = {
            "bias_mm": float(e.mean()), "sigma_mm": float(e.std()),
            "mae_mm": float(np.abs(e).mean()),
            "p95_mm": float(np.percentile(np.abs(e), 95)),
            "hist_counts": hc.tolist(), "hist_edges": he.tolist(),
        }
    dist3d = np.sqrt((err_disp**2).sum(1))
    result["magnitude"] = {
        "mean_mm": float(dist3d.mean()), "sigma_mm": float(dist3d.std()),
        "p50_mm": float(np.percentile(dist3d, 50)),
        "p95_mm": float(np.percentile(dist3d, 95)),
    }
    return result


def compute_error_vs_displacement(velocity_pred, velocity_gt, dt, n_bins=10):
    """Relative error as function of displacement magnitude."""
    if len(velocity_pred) == 0:
        return {}
    disp_gt = velocity_gt * dt
    disp_pred = velocity_pred * dt
    disp_mag_gt = np.sqrt((disp_gt**2).sum(1))
    err_mag = np.sqrt(((disp_pred - disp_gt)**2).sum(1))
    edges = np.percentile(disp_mag_gt, np.linspace(0, 100, n_bins + 1))
    bin_centers, rel_err_bins, n_vecs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (disp_mag_gt >= lo) & (disp_mag_gt < hi)
        if not mask.any():
            continue
        mean_disp = float(disp_mag_gt[mask].mean())
        mean_err = float(err_mag[mask].mean())
        bin_centers.append(mean_disp)
        rel_err_bins.append(mean_err / max(mean_disp, 1e-9))
        n_vecs.append(int(mask.sum()))
    return {"bin_centers_mm": bin_centers, "relative_error": rel_err_bins, "n_vectors": n_vecs}


def compute_error_budget(results_by_mode):
    """Decompose PIV error: tracking + detection contributions."""
    def get_rel(mode):
        if mode not in results_by_mode:
            return None
        m = results_by_mode[mode]
        if isinstance(m, list):
            m = m[0] if m else {}
        return float(m.get("relative_error", m.get("RelErr", 0.0)))
    track_err = get_rel("gt")
    noisy_err = get_rel("realistic_noisy") or get_rel("noisy_gt")
    model_err = get_rel("model")
    budget = {}
    if track_err is not None:
        budget["tracking_error_rel"] = track_err
    if noisy_err is not None and track_err is not None:
        det_contrib = max(0.0, noisy_err - track_err)
        budget["detection_error_rel"] = det_contrib
        budget["noisy_total_rel"] = noisy_err
        budget["detection_fraction"] = det_contrib / max(noisy_err, 1e-9)
    if model_err is not None:
        budget["model_error_rel"] = model_err
        if track_err is not None:
            budget["model_overhead_rel"] = max(0.0, model_err - track_err)
    return budget


def aggregate_metrics(per_sample):
    if not per_sample:
        return _zero_metrics()
    keys = [k for k in per_sample[0] if isinstance(per_sample[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_sample if k in m]
        agg[k] = float(np.mean(vals))
    return agg


def format_metrics_table(metrics, title=""):
    lines = []
    if title:
        lines += ["", "=" * 60, "  " + title, "=" * 60]
    lines.append("  " + "Metric".ljust(28) + "Value".rjust(12))
    lines.append("  " + "-" * 28 + " " + "-" * 12)
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append("  " + k.ljust(28) + ("{:.6f}".format(v)).rjust(12))
        else:
            lines.append("  " + k.ljust(28) + str(v).rjust(12))
    return chr(10).join(lines)


def _zero_metrics():
    return {
        "mae_vx": 0.0, "mae_vy": 0.0, "mae_vz": 0.0, "mae_magnitude": 0.0,
        "rmse_vx": 0.0, "rmse_vy": 0.0, "rmse_vz": 0.0, "rmse_magnitude": 0.0,
        "angle_error_mean_deg": 0.0, "angle_error_median_deg": 0.0,
        "angle_error_p95_deg": 0.0, "spurious_ratio": 0.0,
        "relative_error": 0.0, "n_vectors": 0,
    }


def compute_velocity_metrics_xy(velocity_pred, velocity_gt):
    """Evaluate only xy (lateral) velocity components.

    The lateral plane (xy) has much lower detection noise (sigma_xy=0.014mm)
    than the depth axis (sigma_z=0.048mm). This metric isolates the xy
    contribution to understand the LF-PIV lateral accuracy.

    Args:
        velocity_pred: (N, 3) or (N, 2) predicted velocity (mm/s)
        velocity_gt:   (N, 3) or (N, 2) GT velocity (mm/s)

    Returns:
        dict with mae_vx, mae_vy, mae_xy_magnitude, rmse_xy, relative_error_xy,
        angle_error_xy_mean_deg (in-plane angle), n_vectors
    """
    if len(velocity_pred) == 0:
        return _zero_metrics_xy()
    vp = np.asarray(velocity_pred, dtype=np.float32)[:, :2]   # (N,2) xy only
    vg = np.asarray(velocity_gt,   dtype=np.float32)[:, :2]

    err    = vp - vg
    mae    = np.abs(err).mean(axis=0)
    rmse   = np.sqrt((err**2).mean(axis=0))

    err_mag_xy  = np.sqrt((err**2).sum(axis=1))
    mae_xy      = float(err_mag_xy.mean())
    rmse_xy     = float(np.sqrt((err_mag_xy**2).mean()))

    # In-plane angle error
    dot    = (vp * vg).sum(axis=1)
    n_pred = np.sqrt((vp**2).sum(axis=1)).clip(1e-10)
    n_gt   = np.sqrt((vg**2).sum(axis=1)).clip(1e-10)
    cos_a  = np.clip(dot / (n_pred * n_gt), -1, 1)
    angle_err_xy = np.degrees(np.arccos(cos_a))

    gt_speed_xy = n_gt.mean()
    rel_err_xy  = mae_xy / max(gt_speed_xy, 1e-10)

    # Spurious: xy magnitude > 3sigma from mean
    pred_mag = n_pred
    mu_m, sig_m = pred_mag.mean(), pred_mag.std()
    spurious = np.abs(pred_mag - mu_m) > 3 * max(sig_m, 1e-10)

    return {
        "mae_vx":              float(mae[0]),
        "mae_vy":              float(mae[1]),
        "mae_xy_magnitude":    mae_xy,
        "rmse_vx":             float(rmse[0]),
        "rmse_vy":             float(rmse[1]),
        "rmse_xy_magnitude":   rmse_xy,
        "angle_error_xy_mean_deg":   float(angle_err_xy.mean()),
        "angle_error_xy_median_deg": float(np.median(angle_err_xy)),
        "angle_error_xy_p95_deg":    float(np.percentile(angle_err_xy, 95)),
        "spurious_ratio_xy":   float(spurious.mean()),
        "relative_error_xy":   rel_err_xy,
        "n_vectors":           int(len(vp)),
    }


def _zero_metrics_xy():
    return {
        "mae_vx": 0.0, "mae_vy": 0.0, "mae_xy_magnitude": 0.0,
        "rmse_vx": 0.0, "rmse_vy": 0.0, "rmse_xy_magnitude": 0.0,
        "angle_error_xy_mean_deg": 0.0, "angle_error_xy_median_deg": 0.0,
        "angle_error_xy_p95_deg": 0.0, "spurious_ratio_xy": 0.0,
        "relative_error_xy": 0.0, "n_vectors": 0,
    }


def compute_error_budget_xy(results_by_mode: dict) -> dict:
    """Error budget decomposition for xy-only metrics."""
    def get_rel_xy(mode):
        if mode not in results_by_mode:
            return None
        m = results_by_mode[mode]
        if isinstance(m, list):
            m = m[0] if m else {}
        return float(m.get("relative_error_xy", 0.0))

    track_err = get_rel_xy("gt")
    noisy_err = get_rel_xy("realistic_noisy") or get_rel_xy("noisy_gt")
    model_err = get_rel_xy("model")
    budget = {}
    if track_err is not None:
        budget["tracking_error_xy_rel"] = track_err
    if noisy_err is not None and track_err is not None:
        det_contrib = max(0.0, noisy_err - track_err)
        budget["detection_error_xy_rel"] = det_contrib
        budget["noisy_total_xy_rel"]     = noisy_err
        budget["detection_fraction_xy"]  = det_contrib / max(noisy_err, 1e-9)
    if model_err is not None:
        budget["model_error_xy_rel"] = model_err
        if track_err is not None:
            budget["model_overhead_xy_rel"] = max(0.0, model_err - track_err)
    return budget
