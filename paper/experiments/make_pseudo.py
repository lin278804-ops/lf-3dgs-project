"""Training set whose targets are the particles fitted to each exposure, without reference coordinates.

  python make_pseudo.py

For every training field of the static archive, the per-exposure Gaussian fit (fits/ds9train, produced
by fit.py from the images alone) replaces the reference particle list. Images are unchanged. The
validation folder is linked to the archive only so that the training loop can report progress; no
checkpoint of a network trained on this set is selected with reference coordinates (the final weights
are evaluated).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from gapcommon import DS9, GAP, save_json

out = GAP / "data" / "ds9_pseudo"
(out / "train").mkdir(parents=True, exist_ok=True)
if not (out / "val").exists():
    os.symlink(DS9 / "val", out / "val")
n, counts, ref = 0, [], []
for f in sorted(glob.glob(str(DS9 / "train" / "*.npz"))):
    fit = GAP / "fits" / "ds9train" / Path(f).name
    if not fit.exists():
        continue
    dest = out / "train" / Path(f).name
    z = np.load(fit)
    if not dest.exists():
        with np.load(f) as src:
            keep = {k: src[k] for k in src.files if k not in ("particle_centers",)}
        np.savez_compressed(dest, particle_centers=z["fit_xyz"].astype(np.float32), **keep)
    counts.append(len(z["fit_xyz"]))
    n += 1
save_json(out / "manifest.json", dict(kind="pseudo-labels from per-exposure fitting", n_train=n,
                                      particles_per_field_mean=float(np.mean(counts)), particles_per_field_sd=float(np.std(counts)),
                                      source="fits/ds9train (fit.py --data ds9train); reference coordinates not used"))
print("pseudo-label set:", n, "fields;", round(float(np.mean(counts)), 1), "fitted particles per field")
