"""Gap A: is there a mean jet in the 30/40/50 plenoptic recordings, and does it scale with the label?

According to the supplier (reply of 24 Nov 2026) the recordings show a jet from an upright nozzle and the
series labels are speeds in m/s. Whole-field analysis gave about 10 m/s in every series with a direction
that changed from pair to pair. This script resolves the field: ensemble cross-correlation of the two
exposures in interrogation windows of the central sub-aperture view, accumulated over all pairs of a
series (correlation planes are summed before the peak is taken), with mismatched pairs as control.

Writes source_data/gapA/jet_field.json and jet_field.npz (mean images, vector fields, correlation planes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from gapA_bench_speed import EXTRA, REAL, decode_raw, load

HERE = Path(__file__).resolve().parents[1]
WIN, STEP, R = 32, 16, 12          # window, spacing and search radius in sub-aperture pixels
DT_S = 1e-5


def prep(a, static):
    a = a - static
    a = np.maximum(a - gaussian_filter(a, 3), 0)
    return a


def window_corr(a, b):
    """Correlation plane c[dy, dx] of two windows, b(y + dy, x + dx) ~ a(y, x), zero-mean, search radius R."""
    a = a - a.mean()
    b = b - b.mean()
    n = a.shape[0]
    fa = np.fft.rfft2(a, s=(2 * n, 2 * n))
    fb = np.fft.rfft2(b, s=(2 * n, 2 * n))
    c = np.fft.fftshift(np.fft.irfft2(np.conj(fa) * fb, s=(2 * n, 2 * n)))
    return c[n - R:n + R + 1, n - R:n + R + 1]


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
    m = c.copy()
    m[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -np.inf
    return float(sub[0] - R), float(sub[1] - R), float(c[iy, ix]), float(c[iy, ix] / max(m.max(), 1e-12))


def main():
    um_px = json.loads((HERE / "source_data/gapA/camera_geometry.json").read_text())["plate_june"]["um_per_sai_px"]
    mps = um_px * 1e-6 / DT_S
    out, arrays = {"um_per_sai_px": um_px, "mps_per_px": mps, "window_px": WIN, "step_px": STEP, "search_px": R, "groups": {}}, {}
    for g in ("30", "40", "50"):
        pairs = [json.loads(l) for l in (REAL / f"manifests/{g}_pairs.jsonl").read_text().splitlines() if l.strip()]
        A, B = [], []
        for p in pairs:
            fa, fb = [REAL / "decoded_provisional" / f"{g}_{Path(f).stem}.npz" for f in p["frames"]]
            A.append(load(fa)[0]), B.append(load(fb)[0])
        for pid, fa, fb in EXTRA.get(g, []):
            if fa.exists() and fb.exists():
                A.append(decode_raw(fa, g)[0]), B.append(decode_raw(fb, g)[0])
        A, B = np.array(A), np.array(B)
        med_a, med_b = np.median(A, 0), np.median(B, 0)
        PA = np.array([prep(a, med_a) for a in A])
        PB = np.array([prep(b, med_b) for b in B])
        H, W = A.shape[1:]
        ys = list(range(0, H - WIN + 1, STEP))
        xs = list(range(0, W - WIN + 1, STEP))
        field = np.zeros((len(ys), len(xs), 4))
        ctrl = np.zeros((len(ys), len(xs), 4))
        planes = np.zeros((len(ys), len(xs), 2 * R + 1, 2 * R + 1), np.float32)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                c = sum(window_corr(PA[k, y:y + WIN, x:x + WIN], PB[k, y:y + WIN, x:x + WIN]) for k in range(len(PA)))
                w = sum(window_corr(PA[k, y:y + WIN, x:x + WIN], PB[k + 1, y:y + WIN, x:x + WIN]) for k in range(len(PA) - 1))
                field[i, j] = peak(c)
                ctrl[i, j] = peak(w)
                planes[i, j] = c
        ok = (field[:, :, 3] > 1.3) & (field[:, :, 2] > 3 * np.median(ctrl[:, :, 2]))
        sp = np.hypot(field[:, :, 0], field[:, :, 1])
        out["groups"][g] = dict(
            pairs=int(len(PA)), windows=int(ok.size), windows_valid=int(ok.sum()),
            mean_dy_px=float(field[:, :, 0][ok].mean()) if ok.any() else None, mean_dx_px=float(field[:, :, 1][ok].mean()) if ok.any() else None,
            speed_px_median=float(np.median(sp[ok])) if ok.any() else None, speed_px_max=float(sp[ok].max()) if ok.any() else None,
            speed_mps_median=float(np.median(sp[ok]) * mps) if ok.any() else None, speed_mps_max=float(sp[ok].max() * mps) if ok.any() else None,
            expected_px_if_label_is_mps=float(int(g) / mps),
            brightness_first=float(A.mean()), brightness_second=float(B.mean()))
        arrays[f"g{g}_mean_first"] = A.mean(0).astype(np.float32)
        arrays[f"g{g}_median_first"] = med_a.astype(np.float32)
        arrays[f"g{g}_particles"] = PA.mean(0).astype(np.float32)
        arrays[f"g{g}_field"] = field.astype(np.float32)
        arrays[f"g{g}_control"] = ctrl.astype(np.float32)
        arrays[f"g{g}_valid"] = ok
        arrays[f"g{g}_planes"] = planes
        arrays[f"g{g}_ys"], arrays[f"g{g}_xs"] = np.array(ys), np.array(xs)
        print(g, json.dumps(out["groups"][g]))
        print("  dy (px):\n", np.array2string(np.where(ok, field[:, :, 0], np.nan), precision=1, suppress_small=True, max_line_width=200))
        print("  dx (px):\n", np.array2string(np.where(ok, field[:, :, 1], np.nan), precision=1, suppress_small=True, max_line_width=200))
    (HERE / "source_data/gapA/jet_field.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(HERE / "source_data/gapA/jet_field.npz", **arrays)


if __name__ == "__main__":
    main()
