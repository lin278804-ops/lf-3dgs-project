"""Gap A: decode double-frame plenoptic recordings (groups 30/40/50) into the network input format.

Each raw 5120 x 4096 frame is sampled on the archived microlens lattice of its group: 22 x 22 angular
samples (sensor offsets -10.5 ... 10.5 px from the microlens centre) at 184 x 230 microlens positions,
zero-padded to 186 x 232. Per view, a smooth background (Gaussian, sigma 3 px) is subtracted, a noise
floor of three robust standard deviations is removed and the stack is scaled so that its 99.9th
percentile equals that of the simulated fields (94 grey levels of 255). One scale per frame; no label,
velocity or depth information enters.

  ~/Documents/ChatGPT/lf-piv/.venv-real-piv/bin/python experiments/gapA_decode_pairs.py --out /tmp/gapA_pairs [--pilot]

Writes one <pair_id>.npz per pair: sai (2, 484, 186, 232) uint8, plus the preprocessing constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

REAL = Path.home() / "Documents/ChatGPT/lf-piv/real_data_20261108"
HERE = Path(__file__).resolve().parents[1]
EXTRA = {"30": [dict(pair_id="30_023_024", frames=[
    str(Path.home() / "Desktop/光场/30/80010271289-20261105_162712.596683-23.tif"),
    str(Path.home() / "Desktop/80010271289-20261105_162712.596705-24.tif")])]}
TARGET_P999 = 94.0


LOOKUP = sorted(set(np.linspace(0, 483, 32).astype(int).tolist()) | {10 * 22 + 10, 10 * 22 + 11, 11 * 22 + 10, 11 * 22 + 11})


def decode(path, C, views=None):
    a = tifffile.imread(path).astype(np.float32)
    iy, ix = np.meshgrid(np.arange(184), np.arange(230), indexing="ij")
    y = C[0, 0] + ix * C[1, 0] + iy * C[2, 0]
    x = C[0, 1] + ix * C[1, 1] + iy * C[2, 1]
    out = np.empty((484, 184, 230), np.float32)
    for v in (range(484) if views is None else views):
        s, t = divmod(v, 22)
        out[v] = map_coordinates(a, [y + s - 10.5, x + t - 10.5], order=1, mode="constant", cval=0)
    return out if views is None else out[list(views)]


def clean(stack):
    res = stack - np.stack([gaussian_filter(v, 3) for v in stack])
    sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
    res = np.maximum(res - 3 * sigma, 0)
    p = float(np.percentile(res, 99.9)) or float(res.max())
    out = np.clip(res * (TARGET_P999 / p), 0, 255)
    padded = np.zeros((len(stack), 186, 232), np.uint8)
    padded[:, 1:185, 1:231] = np.rint(out).astype(np.uint8)
    return padded, dict(noise_sigma_counts=float(sigma), p999_counts=p, peak_snr=float((p + 3 * sigma) / sigma),
                        nonzero=float((padded > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true", help="two pairs per group with the clearest correlation peak")
    ap.add_argument("--lookup-only", action="store_true",
                    help="store only the 32 views read by the network and the four innermost views (reference)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = json.loads((HERE / "source_data/gapA/bench_speed.json").read_text())["groups"]
    for g in ("30", "40", "50"):
        C = np.array(json.loads((REAL / f"analysis/lightfield_preflight/lattice_{g}.json").read_text())["affine_yx_from_1_ix_iy"])
        pairs = [json.loads(l) for l in (REAL / f"manifests/{g}_pairs.jsonl").read_text().splitlines() if l.strip()]
        pairs += EXTRA.get(g, [])
        if a.pilot:
            best = sorted(bench[g]["per_pair"], key=lambda r: -r["peak_to_second"])[:2]
            pairs = [p for p in pairs if p["pair_id"] in {b["pair"] for b in best}]
        for p in pairs:
            dest = out / f"{p['pair_id']}.npz"
            if dest.exists():
                continue
            frames, meta = [], []
            for f in p["frames"]:
                s, m = clean(decode(f, C, LOOKUP if a.lookup_only else None))
                frames.append(s)
                meta.append(m)
            np.savez_compressed(dest, sai=np.stack(frames), group=g, pair_id=p["pair_id"],
                                view_idx=np.array(LOOKUP if a.lookup_only else range(484)),
                                noise_sigma_counts=[m["noise_sigma_counts"] for m in meta],
                                p999_counts=[m["p999_counts"] for m in meta], peak_snr=[m["peak_snr"] for m in meta])
            print(p["pair_id"], [round(m["peak_snr"], 1) for m in meta], [round(m["nonzero"], 3) for m in meta], round(dest.stat().st_size / 1e6, 1), "MB", flush=True)


if __name__ == "__main__":
    main()
