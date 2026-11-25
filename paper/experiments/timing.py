"""Time per exposure of network reconstruction and peak extraction (GPU-synchronized, after warm-up).

  python timing.py p7 runs/lift_s0 ...
Images are preloaded; the first three passes are discarded. Writes results/timing.json (appends runs).
The GPU is shared with other jobs of this study; see gpu_utilization in the record.
"""
from __future__ import annotations

import glob
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import run_dir
from gapcommon import GAP, condition, detect, get_cam, load_field, load_json, load_model, save_json


def main(runs):
    dev = "cuda:0"
    files = sorted(glob.glob(str(GAP / "data/sealed_std/*.npz")))[:20]
    fields = [load_field(f)[0].to(dev) for f in files]
    cam = get_cam()
    dest = GAP / "results/timing.json"
    out = load_json(dest) if dest.exists() else {}
    for run in runs:
        model, cfg = load_model(run_dir(run), dev)
        mcam = get_cam(depth=int(cfg.get("grid_depth", 64)))
        for label, pre in (("network", None), ("network_with_conditioning", condition)):
            ts = []
            for i, x in enumerate(fields[:3] + fields):
                torch.cuda.synchronize()
                t = time.perf_counter()
                detect(model, mcam, pre(x) if pre else x, dev, cam_score=cam)
                torch.cuda.synchronize()
                if i >= 3:
                    ts.append(time.perf_counter() - t)
            util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i",
                                   __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0")], capture_output=True, text=True).stdout.strip()
            name = Path(run).name if run not in ("p7", "scratch_s0") else run
            out[f"{name}__{label}"] = dict(ms_mean=float(1000 * np.mean(ts)), ms_sd=float(1000 * np.std(ts)), ms_min=float(1000 * np.min(ts)),
                                           n=len(ts), n_params=int(sum(p.numel() for p in model.parameters())), gpu_utilization=util)
            print(name, label, out[f"{name}__{label}"], flush=True)
    save_json(dest, out)


if __name__ == "__main__":
    main(sys.argv[1:])
