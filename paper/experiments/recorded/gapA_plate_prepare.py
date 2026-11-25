"""Gap A: package the perforated-plate recording (32 stage positions, 100 holes each) for the network.

Keeps the 32 views read by the network and the four innermost views, zero-padded to 186 x 232, with the
supplier's hole coordinates (mm; hole centre distance 1 mm, stage step 0.5 mm). Raw 8-bit values are
kept; the input conditioning is applied by the evaluation script.

  .venv/bin/python experiments/gapA_plate_prepare.py /tmp/gapA_plate
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

PLATE = Path.home() / "Desktop/光场/标定板解码sai及真值npz"
LOOKUP = sorted(set(np.linspace(0, 483, 32).astype(int).tolist()) | {10 * 22 + 10, 10 * 22 + 11, 11 * 22 + 10, 11 * 22 + 11})


def stage_of(name):
    m = re.search(r"_(m?)(\d+)(p5)?_sai", name)
    return (-1 if m.group(1) else 1) * (int(m.group(2)) + (0.5 if m.group(3) else 0.0))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(str(PLATE / "*_sai.npz"))):
        z = np.load(f)
        sai = z["sai"].reshape(484, 184, 230)[LOOKUP]
        pad = np.zeros((len(LOOKUP), 186, 232), np.uint8)
        pad[:, 1:185, 1:231] = sai
        st = stage_of(Path(f).name)
        np.savez_compressed(out / f"plate_{st:+05.1f}.npz".replace("+", "p").replace("-", "m"), sai=pad,
                            view_idx=np.array(LOOKUP), holes_mm=z["gt"].astype(np.float32), stage_mm=np.float32(st),
                            source=Path(f).name)
    print(len(list(out.glob("*.npz"))), "planes", sum(p.stat().st_size for p in out.glob("*.npz")) / 1e6, "MB")


if __name__ == "__main__":
    main(sys.argv[1])
