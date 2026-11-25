"""Score fitted centres and backprojected peaks (fits/<tag>/*.npz) in the common result format.

  python score_preds.py sealed_std [sealed_std__snr30 ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

from gapcommon import GAP, boot_ci, get_cam, save_json, score, score_lateral

BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def main(tag):
    cam = get_cam()
    files = sorted(glob.glob(str(GAP / "fits" / tag / "*.npz")))
    for method, key in (("fit", "fit_xyz"), ("backprojection", "bp_xyz")):
        rows, preds, secs = [], [], []
        for f in files:
            z = np.load(f)
            xyz, gt = z[key], z["gt"]
            m, lat = score(xyz, gt, cam), score_lateral(xyz, gt, cam)
            az = np.abs(gt[:, 2])
            hit = np.zeros(len(gt), bool)
            hit[m["gt_idx"]] = True
            rows.append({q: m[q] for q in ("tp", "fp", "fn", "precision", "recall", "f1")} |
                        {"file": Path(f).name, "n_pred": len(xyz), "lateral_f1": lat["lateral_f1"],
                         "lateral_recall": lat["lateral_recall"],
                         "lateral_abs_dz_median_mm": float(np.median(lat["abs_dz_mm"])) if len(lat["abs_dz_mm"]) else None,
                         "recall_by_depth_band": [float(hit[(az >= lo) & (az < hi)].mean()) for lo, hi in BANDS]})
            preds.append(dict(xyz=xyz, gt=gt, pred_idx=m["pred_idx"], gt_idx=m["gt_idx"], err_mm=m["err_mm"]))
            secs.append(float(z["seconds_fit" if method == "fit" else "seconds_bp"]))
        f1 = np.array([r["f1"] for r in rows])
        err = np.concatenate([p["err_mm"] for p in preds])
        summary = dict(method=method, data=tag, n_fields=len(rows), f1=float(f1.mean()), f1_ci95=boot_ci(f1),
                       precision=float(np.mean([r["precision"] for r in rows])),
                       recall=float(np.mean([r["recall"] for r in rows])),
                       lateral_f1=float(np.mean([r["lateral_f1"] for r in rows])),
                       lateral_recall=float(np.mean([r["lateral_recall"] for r in rows])),
                       lateral_abs_dz_median_mm=float(np.mean([r["lateral_abs_dz_median_mm"] for r in rows])),
                       err_sd_um=[float(v) for v in 1000 * err.std(0)], err_bias_um=[float(v) for v in 1000 * err.mean(0)],
                       err3d_mm=float(np.linalg.norm(err, axis=1).mean()),
                       recall_by_depth_band=[float(v) for v in np.mean([r["recall_by_depth_band"] for r in rows], 0)],
                       seconds_per_field=float(np.mean(secs)))
        save_json(GAP / "results" / f"{method}__{tag}.json", dict(summary=summary, fields=rows))
        np.savez_compressed(GAP / "results" / f"{method}__{tag}.npz",
                            **{f"{q}_{i:04d}": p[q] for i, p in enumerate(preds) for q in ("xyz", "gt", "pred_idx", "gt_idx")})
        print(method, tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
