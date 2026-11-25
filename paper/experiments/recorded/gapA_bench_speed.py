"""Gap A: do the 30/40/50 plenoptic recordings carry the bench speeds named by their labels?

Whole-field ensemble cross-correlation of the two exposures of each pair in the decoded central
sub-aperture view (archived provisional decoding, 145 x 180 microlens positions). A 32-px single-pass
window, as used in the earlier preflight, cannot follow displacements of 8-13 px; the whole-field
correlation plane has no such limit. Controls: deliberately mismatched pairs (first exposure of pair k,
second exposure of pair k+1) and the four off-centre views.

Supplier's conversion (as clarified by the author): 1 sub-aperture pixel per frame = 3.9209 m/s
(39.209 um per pixel, 10 us between exposures).

Writes source_data/gapA/bench_speed.json and bench_speed_planes.npz.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parents[1]
REAL = Path.home() / "Documents/ChatGPT/lf-piv/real_data_20261108"
MPS_PER_PX = 3.9209          # supplier's sentence, read as metres per second per sub-aperture pixel
DT_S = 1e-5                  # pulse separation, confirmed by the author (21 Nov 2026)
R = 30                     # half-width of the stored correlation plane (px)
VIEWS = ["centre", "dx-6", "dx+6", "dy-6", "dy+6"]


def load(path):
    with np.load(path) as z:
        key = [k for k in z.files if z[k].ndim == 3][0]
        return z[key].astype(np.float64)


def decode_raw(path, group):
    """Archived provisional decoding (five angular samples, 145 x 180 microlens positions) of one raw frame."""
    import tifffile
    from scipy.ndimage import map_coordinates
    C = np.array(json.loads((REAL / f"analysis/lightfield_preflight/lattice_{group}.json").read_text())["affine_yx_from_1_ix_iy"])
    a = tifffile.imread(path).astype(np.float32)
    iy, ix = np.meshgrid(np.arange(20, 165), np.arange(25, 205), indexing="ij")
    y = C[0, 0] + ix * C[1, 0] + iy * C[2, 0]
    x = C[0, 1] + ix * C[1, 1] + iy * C[2, 1]
    return np.stack([map_coordinates(a, [y + dy, x + dx], order=1, mode="constant", cval=0)
                     for dx, dy in ((0, 0), (-6, 0), (6, 0), (0, -6), (0, 6))]).astype(np.float64)


# Frame 24 of group 30 was missing from the first delivery and was supplied later (21 Nov 2026).
EXTRA = {"30": [("30_023_024", Path.home() / "Desktop/光场/30/80010271289-20261105_162712.596683-23.tif",
                 Path.home() / "Desktop/80010271289-20261105_162712.596705-24.tif")]}


def prep(a, static=None):
    if static is not None:
        a = a - static                       # temporal median of the group: stationary structure
    a = np.maximum(a - gaussian_filter(a, 3), 0)
    a = a - a.mean()
    return a / (a.std() + 1e-12)


def xcorr(a, b):
    """Correlation plane c[dy, dx]: b(y + dy, x + dx) ~ a(y, x); normalized by overlap."""
    H, W = a.shape
    fa = np.fft.rfft2(a, s=(2 * H, 2 * W))
    fb = np.fft.rfft2(b, s=(2 * H, 2 * W))
    c = np.fft.fftshift(np.fft.irfft2(np.conj(fa) * fb, s=(2 * H, 2 * W)))
    c = c[H - R:H + R + 1, W - R:W + R + 1]
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    return c / ((H - np.abs(dy)) * (W - np.abs(dx)))


def peak(c):
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    sub = []
    for axis, i in ((0, iy), (1, ix)):
        if 0 < i < c.shape[axis] - 1:
            lo, mid, hi = (c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) if axis == 0 else (c[iy, ix - 1], c[iy, ix], c[iy, ix + 1])
            den = lo - 2 * mid + hi
            sub.append(i + (0.5 * (lo - hi) / den if den < 0 else 0.0))
        else:
            sub.append(float(i))
    masked = c.copy()
    masked[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -np.inf
    second = masked.max()
    noise = np.median(np.abs(c - np.median(c))) * 1.4826
    return dict(dy=float(sub[0] - R), dx=float(sub[1] - R), height=float(c[iy, ix]),
                peak_to_second=float(c[iy, ix] / max(second, 1e-12)), peak_over_noise=float(c[iy, ix] / max(noise, 1e-12)))


def main():
    um_px = json.loads((HERE / "source_data/gapA/camera_geometry.json").read_text())["plate_june"]["um_per_sai_px"]
    plate_mps = um_px * 1e-6 / DT_S        # camera's own plate: metres per second per sub-aperture pixel
    out, planes = {"conversion_mps_per_sai_px": MPS_PER_PX, "plate_um_per_sai_px": um_px, "dt_s": DT_S,
                   "plate_mps_per_sai_px": plate_mps, "groups": {}}, {}
    for g in ("30", "40", "50"):
        pairs = [json.loads(l) for l in (REAL / f"manifests/{g}_pairs.jsonl").read_text().splitlines() if l.strip()]
        stacks = []
        for p in pairs:
            fa, fb = [REAL / "decoded_provisional" / f"{g}_{Path(f).stem}.npz" for f in p["frames"]]
            stacks.append((load(fa), load(fb)))
        for pid, fa, fb in EXTRA.get(g, []):
            if fa.exists() and fb.exists():
                check = decode_raw(Path.home() / "Desktop/光场" / g / Path(pairs[0]["frames"][0]).name, g)
                assert np.allclose(check, stacks[0][0], atol=1e-2), "decoding differs from the archive"
                stacks.append((decode_raw(fa, g), decode_raw(fb, g)))
                pairs.append({"pair_id": pid})
        rec = {"label": int(g), "pairs": len(pairs), "views": {}, "per_pair": []}
        expected = int(g) / MPS_PER_PX
        med_a = np.median([a for a, _ in stacks], 0)
        med_b = np.median([b for _, b in stacks], 0)
        for v, name in enumerate(VIEWS):
            right = np.mean([xcorr(prep(a[v], med_a[v]), prep(b[v], med_b[v])) for a, b in stacks], 0)
            wrong = np.mean([xcorr(prep(stacks[k][0][v], med_a[v]), prep(stacks[k + 1][1][v], med_b[v]))
                             for k in range(len(stacks) - 1)], 0)
            pk, pw = peak(right), peak(wrong)
            pk["speed_px"] = float(np.hypot(pk["dx"], pk["dy"]))
            pk["speed_mps"] = pk["speed_px"] * MPS_PER_PX
            rec["views"][name] = dict(ensemble=pk, mismatched_pairs=pw)
            planes[f"g{g}_{name}"] = right.astype(np.float32)
            planes[f"g{g}_{name}_mismatched"] = wrong.astype(np.float32)
        for p, (a, b) in zip(pairs, stacks):
            pk = peak(xcorr(prep(a[0], med_a[0]), prep(b[0], med_b[0])))
            rec["per_pair"].append(dict(pair=p["pair_id"], dx=pk["dx"], dy=pk["dy"], speed_px=float(np.hypot(pk["dx"], pk["dy"])),
                                        peak_to_second=pk["peak_to_second"]))
        sp = np.array([r["speed_px"] for r in rec["per_pair"]])
        ens = rec["views"]["centre"]["ensemble"]
        near = np.abs(sp - ens["speed_px"]) < 1.0
        # is there any correlation peak where the labelled speed would put it?
        c, w = planes[f"g{g}_centre"].astype(float), planes[f"g{g}_centre_mismatched"].astype(float)
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rad = np.hypot(xx, yy)
        noise = 1.4826 * np.median(np.abs(w - np.median(w)))
        rec["radial_max_over_noise"] = [float(c[(rad >= r - 0.5) & (rad < r + 0.5)].max() / noise) for r in range(26)]
        rec["mismatched_max_over_noise"] = float(w.max() / noise)
        rec["peak_over_noise_at_expected_radius"] = float(c[np.abs(rad - expected) <= 1.5].max() / noise)
        plate_r = int(g) / plate_mps
        rec["expected_px_if_label_is_mps_plate_scale"] = plate_r
        rec["peak_over_noise_at_expected_radius_plate_scale"] = float(c[np.abs(rad - plate_r) <= 1.0].max() / noise)
        vx = rec["views"]
        rec["parallax_change_px_over_12_sensor_px"] = [
            float(vx["dx+6"]["ensemble"]["dx"] - vx["dx-6"]["ensemble"]["dx"]),
            float(vx["dy+6"]["ensemble"]["dy"] - vx["dy-6"]["ensemble"]["dy"])]
        clear = np.array([r["peak_to_second"] >= 1.5 for r in rec["per_pair"]])
        dxy = np.array([[r["dx"], r["dy"]] for r in rec["per_pair"]])[clear]
        spc = np.hypot(*dxy.T)
        rec["per_pair_clear"] = dict(
            n=int(clear.sum()), speed_px_median=float(np.median(spc)), speed_px_iqr=[float(v) for v in np.percentile(spc, [25, 75])],
            displacement_um_median=float(np.median(spc) * um_px),
            speed_mps_median=float(np.median(spc) * plate_mps), speed_mps_iqr=[float(v * plate_mps) for v in np.percentile(spc, [25, 75])],
            speed_mps_median_supplier_conversion=float(np.median(spc) * MPS_PER_PX),
            mean_vector_px=[float(v) for v in dxy.mean(0)], resultant_over_mean_speed=float(np.hypot(*dxy.mean(0)) / spc.mean()),
            pairs_above_3px=int((spc > 3).sum()))
        rec.update(expected_px_if_label_is_mps=expected, per_pair_speed_px_median=float(np.median(sp)),
                   per_pair_within_1px_of_ensemble=int(near.sum()),
                   per_pair_speed_px_sd_of_those=float(sp[near].std()) if near.sum() > 1 else None,
                   ensemble_speed_mps=ens["speed_mps"], ratio_measured_to_label=ens["speed_mps"] / int(g))
        out["groups"][g] = rec
        print(g, "expected px", round(expected, 2), "| ensemble centre", {k: round(v, 3) for k, v in ens.items()},
              "| mismatched", {k: round(v, 3) for k, v in rec["views"]["centre"]["mismatched_pairs"].items()},
              "| per-pair near", int(near.sum()), "/", len(sp), "| at expected radius",
              round(rec["peak_over_noise_at_expected_radius"], 1), "x noise; plate-scale radius",
              round(rec["peak_over_noise_at_expected_radius_plate_scale"], 1), "| parallax change",
              [round(v, 3) for v in rec["parallax_change_px_over_12_sensor_px"]])
        print("   per pair:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec["per_pair_clear"].items()
                               if not isinstance(v, list)}, "| label at plate scale would be", round(plate_r, 2), "px")
        print("   other views:", {n: (round(rec["views"][n]["ensemble"]["dx"], 2), round(rec["views"][n]["ensemble"]["dy"], 2),
                                      round(rec["views"][n]["ensemble"]["peak_to_second"], 2)) for n in VIEWS[1:]})
    dest = HERE / "source_data/gapA"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bench_speed.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(dest / "bench_speed_planes.npz", **planes)


if __name__ == "__main__":
    main()
