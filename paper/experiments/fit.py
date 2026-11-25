"""Per-exposure Gaussian fitting and backprojected peaks on sealed fields, with the archived
observation-only protocol (2,000 steps, hybrid backprojection initialization, no labels).

  python fit.py --data sealed_std --shard 0 --nshards 3 [--snr 30]

Standalone on purpose: it imports the audited fitting code (lf-3dgs-project-repair1), not the
package used for the network. Writes fits/<data>[__snrXX]/<field>.npz with fitted centres,
backprojected peaks and timing; score_preds.py turns these into the common result format.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ.get("LFPIV_PROJECT", Path(__file__).resolve().parents[2]))
GAP = Path(os.environ.get("LFPIV_WORK", PROJECT / "work"))
DS9 = Path(os.environ.get("LFPIV_TRAIN_DATA", GAP / "data" / "lightfield_dataset9"))
# The fit runs from the revision of the package in fitcore/, whose optimizer never loads reference labels
# and whose initial count and pruning floor do not depend on the number of reference particles.
sys.path.insert(0, str(Path(__file__).with_name("fitcore")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lf_3dgs.backproject import backproj_init_centers  # noqa: E402
from lf_3dgs.camera import LFCameraParams  # noqa: E402
from lf_3dgs.metrics import nms_gaussians  # noqa: E402
from lf_3dgs.optimizer import optimise_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    src = {"ds9test": DS9 / "test", "ds9train": DS9 / "train", "ds9val": DS9 / "val"}.get(a.data, GAP / "data" / a.data)
    if (src / "field_0000").is_dir():      # two-exposure set: one folder per field, 11 exposures each
        files = [f for d in sorted(glob.glob(str(src / "field_*")))[: a.limit] for f in sorted(glob.glob(d + "/*.npz"))]
    else:
        files = sorted(glob.glob(str(src / "*.npz")))[: a.limit]
    paired = a.data.startswith("realpairs")        # recorded double frames: sai (2, 484, H, W), no labels
    if paired:
        files = [(f, k) for f in files for k in (0, 1)]
    tag = a.data + ("_fit" if paired else "") + (f"__snr{a.snr:g}" if a.snr is not None else "")
    out = GAP / "fits" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(__file__).with_name("configs") / "e7_star_physics_tuned.yaml").read_text())
    cfg["stage1"].update(diagnostic_oracle=False, intensity_scale=255., n_gaussians_fixed=1500, prune_min_keep=20,
                         log_every=1000, vis_every=0)
    cam = LFCameraParams()
    for i, path in enumerate(files):
        if i % a.nshards != a.shard:
            continue
        if paired:
            path, k = path
            name = f"{Path(path).stem}__f{k}"
        else:
            name = Path(path).stem if Path(path).parent == src else f"{Path(path).parent.name}__{Path(path).stem}"
        dest = out / f"{name}.npz"
        if dest.exists():
            continue
        with np.load(path) as z:
            if paired:
                sai, gt = z["sai"][k].astype(np.float32), np.zeros((0, 3), np.float32)
            else:
                sai = z["sai_stack"].astype(np.float32)
                gt = z["particle_centers"].astype(np.float32)
        if a.snr is not None:
            rng = np.random.default_rng(1000 + i)
            peak = max(float(np.quantile(sai[::8], 0.999)), 0.255)
            sai = np.clip(sai + rng.normal(0, peak / 10 ** (a.snr / 20), sai.shape).astype(np.float32), 0, 255)
        work = out / f"_work_{name}"
        work.mkdir(exist_ok=True)
        obs = work / "observation_only.npz"
        np.savez(obs, sai_stack=sai, x_range=np.array(cam.x_range, np.float32), y_range=np.array(cam.y_range, np.float32),
                 z_range=np.array(cam.z_range, np.float32), grid_shape=np.array(cam.grid_shape, np.int32))
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        t = time.perf_counter()
        bp, _ = backproj_init_centers(torch.from_numpy(sai / 255.), cam, device="cuda", mode="mean", view_stride=2,
                                      peak_threshold=.12, peak_min_distance=2)
        torch.cuda.synchronize()
        tb = time.perf_counter() - t
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        t = time.perf_counter()
        optimise_sample(str(obs), str(work / "stage1"), cfg, "cuda")
        torch.cuda.synchronize()
        ts = time.perf_counter() - t
        with np.load(work / "stage1/gaussians.npz") as z:
            assert len(z["gt_centers"]) == 0
            mu, alpha = z["mu"], 1 / (1 + np.exp(-z["logit_alpha"]))
        keep = alpha >= .2
        pv = cam.physical_to_voxel(torch.tensor(mu[keep])).numpy()
        nk = nms_gaussians(pv, alpha[keep], merge_radius=1.5)
        np.savez_compressed(dest, fit_xyz=mu[keep][nk].astype(np.float32), fit_alpha=alpha[keep][nk].astype(np.float32),
                            bp_xyz=bp.cpu().numpy().astype(np.float32), gt=gt, seconds_fit=ts, seconds_bp=tb)
        for f in list(work.rglob("*"))[::-1]:
            f.unlink() if f.is_file() else f.rmdir()
        work.rmdir()
        print("FIT", tag, name, round(ts, 1), flush=True)


if __name__ == "__main__":
    main()
