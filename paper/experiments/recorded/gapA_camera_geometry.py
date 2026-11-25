"""Gap A: object-space scale of the plenoptic camera (serial ...289) from its own perforated plate.

The supplier's README for the plate recording gives the hole centre distance (1 mm), the hole diameter
(0.2 mm) and the stage step (0.5 mm, 32 positions from 7.5 to -8 mm, z = 0 at the focal plane). The
author states that the November flow recordings (30/40/50) kept this configuration.

1. Lateral scale: least-squares lattice through the hole centres of the central view, on the sharply
   imaged planes within 2.5 mm of the focus.
2. Angular check: shift of the hole pattern between adjacent views against stage position. For a thin
   main lens and the documented microlens array (pitch 100 um, focal length 0.8 mm, pixel 4.5 um) the
   shift per view index per millimetre is (pixel / (f_mla * pitch)) * M^2. It tests whether a proposed
   lateral scale is compatible with the depth response of the same camera.

Writes source_data/gapA/camera_geometry.json.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
PLATE = Path.home() / "Desktop/光场/标定板解码sai及真值npz"
RAW = Path.home() / "Desktop/光场"
REAL = Path.home() / "Documents/ChatGPT/lf-piv/real_data_20261108"
PIX_MM, PITCH_MM, FMLA_MM, HOLE_PITCH_MM = 0.0045, 0.100, 0.800, 1.0


def stage_of(name):
    m = re.search(r"_(m?)(\d+)(p5)?_sai", name)
    return (-1 if m.group(1) else 1) * (int(m.group(2)) + (0.5 if m.group(3) else 0.0))


def hp(a):
    a = a.astype(float)
    return ndi.gaussian_filter(a, 1) - ndi.gaussian_filter(a, 8)


def shift(a, b):
    """Sub-pixel displacement of b relative to a (small shifts), by cross-correlation."""
    a, b = hp(a), hp(b)
    a, b = a - a.mean(), b - b.mean()
    c = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b), s=a.shape))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = c[cy - 3:cy + 4, cx - 3:cx + 4]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    out = []
    for axis, i in ((0, iy), (1, ix)):
        if 0 < i < 6:
            lo, mid, hi = (win[iy - 1, ix], win[iy, ix], win[iy + 1, ix]) if axis == 0 else \
                (win[iy, ix - 1], win[iy, ix], win[iy, ix + 1])
            out.append(i - 3 + 0.5 * (lo - hi) / (lo - 2 * mid + hi))
        else:
            out.append(float(i - 3))
    return out[0], out[1]              # dy, dx


def hole_lattice(im):
    """Least-squares lattice through the hole centres of one view: spacing along both axes (px), rotation,
    angle between the axes, holes used, median residual (px)."""
    h = hp(im)
    lab, n = ndi.label(h > h.mean() + 2 * h.std())
    idx = range(1, n + 1)
    c = np.array(ndi.center_of_mass(np.maximum(h, 0), lab, idx))[ndi.sum(h > 0, lab, idx) >= 3][:, ::-1]   # x, y
    if len(c) < 20:
        return None
    _, i = cKDTree(c).query(c, k=5)
    v = (c[i[:, 1:]] - c[:, None]).reshape(-1, 2)
    r = np.hypot(*v.T)
    v = v[(r > 8) & (r < 16)]
    ax = v[np.abs(v[:, 0]) > np.abs(v[:, 1])]
    ay = v[np.abs(v[:, 1]) >= np.abs(v[:, 0])]
    a, b = np.median(ax * np.sign(ax[:, :1]), 0), np.median(ay * np.sign(ay[:, 1:]), 0)
    o = c[np.argmin(np.hypot(*(c - c.mean(0)).T))]
    ij = np.rint(np.linalg.solve(np.stack([a, b], 1), (c - o).T).T)
    X = np.column_stack([np.ones(len(c)), ij])
    coef = np.linalg.lstsq(X, c, rcond=None)[0]
    g = np.hypot(*(c - X @ coef).T) < 1.0
    if g.sum() < 20:
        return None
    coef = np.linalg.lstsq(X[g], c[g], rcond=None)[0]
    a, b = coef[1], coef[2]
    return dict(pitch_x_px=float(np.hypot(*a)), pitch_y_px=float(np.hypot(*b)),
                rotation_deg=float(np.degrees(np.arctan2(a[1], a[0]))),
                axes_angle_deg=float(np.degrees(np.arccos(a @ b / np.hypot(*a) / np.hypot(*b)))),
                holes=int(g.sum()), residual_px=float(np.median(np.hypot(*(c[g] - X[g] @ coef).T))))


def plate():
    rows = []
    for f in sorted(glob.glob(str(PLATE / "*_sai.npz"))):
        z = np.load(f)
        sai = z["sai"].astype(np.float32)
        gt = z["gt"]
        lat = hole_lattice(sai[10:12, 10:12].mean((0, 1)))
        # adjacent-view shifts along the two angular axes, views well inside the illuminated disc
        dx = [shift(sai[s, t], sai[s, t + 1])[1] for s in (9, 10, 11, 12) for t in range(5, 16)]
        dy = [shift(sai[s, t], sai[s + 1, t])[0] for t in (9, 10, 11, 12) for s in range(5, 16)]
        rows.append(dict(file=Path(f).name, stage=stage_of(Path(f).name), gt_z=float(gt[0, 2]), lattice=lat,
                         shift_x_px_per_view=float(np.median(dx)), shift_y_px_per_view=float(np.median(dy))))
    st = np.array([r["stage"] for r in rows])
    sx = np.array([r["shift_x_px_per_view"] for r in rows])
    sy = np.array([r["shift_y_px_per_view"] for r in rows])
    kx, bx = np.polyfit(st, sx, 1)
    ky, by = np.polyfit(st, sy, 1)
    k = 0.5 * (abs(kx) + abs(ky))                     # px per view index per mm
    # lateral scale from the sharply imaged planes around the focus (holes blur and merge farther out)
    near = [r for r in rows if abs(r["stage"]) <= 2.5 and r["lattice"] and r["lattice"]["holes"] >= 30]
    pitch = np.array([0.5 * (r["lattice"]["pitch_x_px"] + r["lattice"]["pitch_y_px"]) for r in near])
    zn = np.array([r["stage"] for r in near])
    lp, l0 = np.polyfit(zn, pitch, 1)
    at0 = [r for r in near if r["stage"] == 0][0]["lattice"]
    M = l0 * PITCH_MM / HOLE_PITCH_MM                 # magnification at z = 0
    gain = PIX_MM / (FMLA_MM * PITCH_MM)              # thin-lens parallax per view index per mm at M = 1
    return dict(planes=len(rows), stage_range_mm=[float(st.min()), float(st.max())],
                hole_pitch_mm=HOLE_PITCH_MM, hole_pitch_source="supplier README of the plate recording (centre distance 1 mm, "
                "hole diameter 0.2 mm, planes 0.5 mm apart); see plate_readme_from_supplier.txt",
                planes_used_for_scale=len(near), hole_pitch_px_at_z0=float(l0), hole_pitch_px_per_mm_depth=float(lp),
                hole_pitch_px_xy_at_z0=[at0["pitch_x_px"], at0["pitch_y_px"]], lattice_axes_angle_deg_at_z0=at0["axes_angle_deg"],
                um_per_sai_px=float(1000 * HOLE_PITCH_MM / l0),
                um_per_sai_px_range_within_2p5mm=[float(1000 * HOLE_PITCH_MM / pitch.max()), float(1000 * HOLE_PITCH_MM / pitch.min())],
                magnification=float(M),
                parallax_px_per_view_per_mm=dict(x=float(kx), y=float(ky), mean_abs=float(k)),
                parallax_fit_residual_px=float(np.std(sx - (kx * st + bx))),
                zero_parallax_plane_mm=float(np.mean([-bx / kx, -by / ky])),
                thin_lens_parallax_at_measured_magnification=float(gain * M ** 2),
                measured_over_thin_lens_parallax=float(k / (gain * M ** 2)),
                if_sai_px_were_39p209um=dict(magnification=PITCH_MM / 0.039209, hole_pitch_mm=float(l0 * 0.039209),
                                             thin_lens_parallax_px_per_view_per_plane_step=float(
                                                 gain * (PITCH_MM / 0.039209) ** 2 * l0 * 0.039209),
                                             measured_over_expected=float(k / (gain * (PITCH_MM / 0.039209) ** 2 * l0 * 0.039209))),
                rows=rows)


if __name__ == "__main__":
    res = dict(plate_june=plate())
    dest = HERE / "source_data/gapA"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "camera_geometry.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res["plate_june"].items() if k != "rows"}, indent=1, ensure_ascii=False))
