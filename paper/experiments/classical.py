"""Classical baselines given the same calibrated projection: multiplicative algebraic
reconstruction (MART/SMART) and the archived plane sweep.

  python classical.py tune                       # choose MART settings on 8 validation fields
  python classical.py run --data sealed_std      # frozen settings -> results/mart*__<data>.json etc.

MART: voxel intensities E on the 64 x 128 x 128 grid, bilinear projection weights from the same
closed-form camera, simultaneous multiplicative updates E <- E * exp(mu * mean_k log(I_k / P_k E)),
after the usual Gaussian pre-blur of the recordings. Particles are local maxima of E above a relative
threshold (at most 1,500) within a 3 x 3 x zwin window, refined by an intensity centroid over that window
and suppressed within 1.5 voxels, i.e. the same extraction budget and scoring as every other method.
Views, pre-blur, iterations, threshold and the axial window are tuned on validation fields.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, uniform_filter

from gapcommon import DS9, GAP, boot_ci, get_cam, load_field, nms_gaussians, save_json, score, score_lateral
from lf_3dgs.camera import DifferentiableLFProjection

VIEWS32 = torch.linspace(0, 483, 32).long()


def bilinear_matrix(coord, n):
    """coord (..., M) continuous pixel coordinates -> dense weights (..., M, n)."""
    c0 = torch.floor(coord)
    w1 = coord - c0
    out = torch.zeros(coord.shape + (n,), device=coord.device)
    for off, w in ((0, 1 - w1), (1, w1)):
        idx = (c0 + off).long()
        ok = (idx >= 0) & (idx < n)
        out.scatter_add_(-1, idx.clamp(0, n - 1).unsqueeze(-1), (w * ok).unsqueeze(-1))
    return out


class Mart:
    def __init__(self, cam, views, device, chunk=32, up=1):
        """`up` refines the reconstruction grid by an integer factor per axis (up = 2 gives lateral
        voxels of 0.78 image pixels); peaks are always returned in millimetres."""
        self.cam, self.dev, self.chunk, self.up = cam, device, max(1, chunk // up ** 2), up
        D, H, W = self.shape = tuple(int(n) * up for n in cam.grid_shape)
        z = cam.z_range[0] + (torch.arange(D) + 0.5) / D * (cam.z_range[1] - cam.z_range[0])
        y = cam.y_range[0] + (torch.arange(H) + 0.5) / H * (cam.y_range[1] - cam.y_range[0])
        x = cam.x_range[0] + (torch.arange(W) + 0.5) / W * (cam.x_range[1] - cam.x_range[0])
        proj = DifferentiableLFProjection(cam)
        s_idx, t_idx = views // cam.px_per_mla, views % cam.px_per_mla
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        _, col = proj(torch.stack((xx, torch.zeros_like(xx), zz), -1).reshape(-1, 3), s_idx, t_idx)
        zz, yy = torch.meshgrid(z, y, indexing="ij")
        row, _ = proj(torch.stack((torch.zeros_like(yy), yy, zz), -1).reshape(-1, 3), s_idx, t_idx)
        self.views = views
        self.row = row.reshape(len(views), D, H)          # separable projection: rows depend on (y, z)
        self.col = col.reshape(len(views), D, W)

    def _weights(self, k0, k1):
        R = bilinear_matrix(self.row[k0:k1].to(self.dev), self.cam.num_mla_y)   # (k, D, H, Hi)
        C = bilinear_matrix(self.col[k0:k1].to(self.dev), self.cam.num_mla_x)   # (k, D, W, Wi)
        return R, C

    @torch.no_grad()
    def reconstruct(self, sai01, iters=10, mu=1.0, blur=1.0, init="mlos"):
        img = sai01[self.views].to(self.dev)
        if blur > 0:
            r = max(1, int(3 * blur))
            g = torch.exp(-torch.arange(-r, r + 1, device=self.dev).float() ** 2 / (2 * blur ** 2))
            g = g / g.sum()
            img = F.conv2d(img[:, None], g.view(1, 1, 1, -1), padding=(0, r))
            img = F.conv2d(img, g.view(1, 1, -1, 1), padding=(r, 0))[:, 0]
        K = len(self.views)
        D, H, W = self.shape
        eps = 1e-6
        # multiplicative line-of-sight initialization
        logE = torch.zeros(D, H, W, device=self.dev)
        for k0 in range(0, K, self.chunk):
            R, C = self._weights(k0, min(K, k0 + self.chunk))
            back = torch.einsum("kdhi,kij,kdwj->kdhw", R, img[k0:k0 + self.chunk], C)
            logE += torch.log(back + eps).sum(0)
        E = torch.exp(logE / K) if init == "mlos" else torch.ones(D, H, W, device=self.dev)
        for _ in range(iters):
            upd = torch.zeros_like(E)
            for k0 in range(0, K, self.chunk):
                R, C = self._weights(k0, min(K, k0 + self.chunk))
                projd = torch.einsum("kdiw,kdwj->kij", torch.einsum("kdhi,dhw->kdiw", R, E), C)
                ratio = (img[k0:k0 + self.chunk] + eps) / (projd + eps)
                upd += torch.log(torch.einsum("kdhi,kij,kdwj->kdhw", R, ratio, C).clamp(1e-3, 1e3)).sum(0)
            E = E * torch.exp(mu * upd / K)
        return E

    def peaks(self, E, rel_thr, cap=1500, zwin=3):
        """Local maxima in a 3 x 3 x zwin window (reconstructed particles are elongated along the
        optical axis), intensity centroid over the same window, then the common suppression."""
        vol = E.cpu().numpy() if torch.is_tensor(E) else E
        peak = (vol == maximum_filter(vol, size=(zwin, 3, 3), mode="constant", cval=0)) & (vol > rel_thr * vol.max())
        idx = np.column_stack(np.nonzero(peak))
        if len(idx) == 0:
            return np.empty((0, 3), np.float32), np.empty(0, np.float32)
        strength = vol[tuple(idx.T)]
        order = np.argsort(-strength, kind="stable")[:cap]
        idx, strength = idx[order], strength[order]
        hz = zwin // 2
        pad = np.pad(vol, ((hz, hz), (1, 1), (1, 1)))
        pos = np.zeros((len(idx), 3), np.float32)
        wsum = np.zeros(len(idx), np.float32)
        for dz, dy, dx in itertools.product(range(-hz, hz + 1), (-1, 0, 1), (-1, 0, 1)):
            w = pad[idx[:, 0] + hz + dz, idx[:, 1] + 1 + dy, idx[:, 2] + 1 + dx]
            pos += w[:, None] * np.array([dz, dy, dx], np.float32)
            wsum += w
        vox = idx + 0.5 + pos / np.maximum(wsum, 1e-12)[:, None]           # centre convention, [z, y, x]
        sc = strength / strength.max()
        D, H, W = vol.shape
        c = self.cam
        xyz = np.stack([c.x_range[0] + vox[:, 2] / W * (c.x_range[1] - c.x_range[0]),
                        c.y_range[0] + vox[:, 1] / H * (c.y_range[1] - c.y_range[0]),
                        c.z_range[0] + vox[:, 0] / D * (c.z_range[1] - c.z_range[0])], 1).astype(np.float32)
        keep = nms_gaussians(c.physical_to_voxel(torch.from_numpy(xyz)).numpy(), sc, 1.5)   # scoring grid
        return xyz[keep], sc[keep].astype(np.float32)


class PlaneSweep:
    """Archived calibrated plane sweep (local contrast with cross-view agreement), 32 views."""

    def __init__(self, cam, device):
        self.cam, self.dev = cam, device
        d, h, w = cam.grid_shape
        z, y, x = [torch.linspace(*lim, n) for lim, n in zip((cam.z_range, cam.y_range, cam.x_range), (d, h, w))]
        proj = DifferentiableLFProjection(cam)
        view = VIEWS32
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        _, col = proj(torch.stack((xx, torch.zeros_like(xx), zz), -1).reshape(-1, 3), view // 22, view % 22)
        zz, yy = torch.meshgrid(z, y, indexing="ij")
        row, _ = proj(torch.stack((torch.zeros_like(yy), yy, zz), -1).reshape(-1, 3), view // 22, view % 22)
        self.x = (2 * col.reshape(32, d, 1, w) / (cam.num_mla_x - 1) - 1).to(device)
        self.y = (2 * row.reshape(32, d, h, 1) / (cam.num_mla_y - 1) - 1).to(device)

    @torch.no_grad()
    def volume(self, sai01):
        im = sai01[VIEWS32].numpy()
        bg = uniform_filter(im, size=(1, 5, 5), mode="reflect")
        var = np.maximum(uniform_filter(im ** 2, size=(1, 5, 5), mode="reflect") - bg ** 2, 0)
        contrast = np.maximum(im - bg, 0) / np.sqrt(var + (1 / 255) ** 2)
        inp = torch.from_numpy(np.stack((im, contrast), 1)).to(self.dev)
        d, h, w = self.cam.grid_shape
        out = torch.empty(d, h, w, device=self.dev)
        for a in range(0, d, 8):
            b = min(d, a + 8)
            gx, gy = torch.broadcast_tensors(self.x[:, a:b], self.y[:, a:b])
            grid = torch.stack((gx, gy), -1).reshape(32, (b - a) * h, w, 2)
            vals = F.grid_sample(inp, grid, align_corners=True, padding_mode="zeros", mode="bilinear")
            vals = vals.reshape(32, 2, b - a, h, w)
            out[a:b] = vals.mean(0)[1].square() / (vals[:, 1].square().mean(0).sqrt() + 1e-7)
        return out.cpu().numpy()

    def peaks(self, vol, rel_thr=0.24, cap=1500):
        peak = (vol == maximum_filter(vol, size=3, mode="constant", cval=0)) & (vol > 0)
        idx = np.column_stack(np.nonzero(peak))
        strength = vol[tuple(idx.T)]
        order = np.argsort(-strength, kind="stable")[:cap]
        idx, strength = idx[order], strength[order]
        pos = idx.astype(np.float32)
        for axis in range(3):
            inner = np.flatnonzero((idx[:, axis] > 0) & (idx[:, axis] < vol.shape[axis] - 1))
            lo, hi = idx[inner].copy(), idx[inner].copy()
            lo[:, axis] -= 1
            hi[:, axis] += 1
            a, b, c = vol[tuple(lo.T)], strength[inner], vol[tuple(hi.T)]
            den = a - 2 * b + c
            off = np.divide(0.5 * (a - c), den, out=np.zeros_like(a), where=den < -1e-12)
            pos[inner, axis] += off.clip(-.5, .5)
        lim = np.array([self.cam.x_range, self.cam.y_range, self.cam.z_range], np.float32)
        xyz = lim[:, 0] + pos[:, ::-1] / (np.array(vol.shape[::-1], np.float32) - 1) * np.diff(lim, axis=1).ravel()
        vox = self.cam.physical_to_voxel(torch.from_numpy(xyz)).numpy()
        sc = strength / strength.max()
        keep = nms_gaussians(vox, sc, 1.5) & (sc >= rel_thr)
        return xyz[keep].astype(np.float32), sc[keep].astype(np.float32)


def files_of(name):
    d = {"ds9val": DS9 / "val", "ds9test": DS9 / "test"}.get(name, GAP / "data" / name)
    return sorted(glob.glob(str(d / "*.npz")))


THRS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
ZWINS = (3, 7, 15, 31)


def tune(device, up=1):
    """Grid over views, pre-blur, iterations (0 = multiplicative line-of-sight only), peak threshold and
    axial peak window on 8 validation fields; the best setting per view count is frozen."""
    cam = get_cam()
    files = files_of("ds9val")[:8]
    fields = [load_field(f) for f in files]
    grid = []
    blurs, its = ((0.0, 0.5, 0.7, 1.0), (0, 2, 5, 10)) if up == 1 else ((0.0, 0.5), (0, 2, 5))
    for nviews, blur, iters, mu in itertools.product((32, 484), blurs, its, (1.0,)):
        views = VIEWS32 if nviews == 32 else torch.arange(484)
        mart = Mart(cam, views, device, chunk=32 if nviews == 32 else 44, up=up)
        vols = [mart.reconstruct(sai, iters=iters, mu=mu, blur=blur).cpu().numpy() for sai, _ in fields]
        for thr, zwin in itertools.product(THRS, [z * up + (1 - up % 2) for z in ZWINS]):
            f1 = float(np.mean([score(mart.peaks(E, thr, zwin=zwin)[0], gt, cam)["f1"] for E, (_, gt) in zip(vols, fields)]))
            grid.append(dict(views=nviews, blur=blur, iters=iters, mu=mu, thr=thr, zwin=zwin, val_f1=f1))
        best = max((g for g in grid if g["views"] == nviews and g["blur"] == blur and g["iters"] == iters),
                   key=lambda g: g["val_f1"])
        print(best, flush=True)
    chosen = {str(n): max((g for g in grid if g["views"] == n), key=lambda g: g["val_f1"]) for n in (32, 484)}
    save_json(GAP / "results" / f"mart_tuning{'' if up == 1 else f'_up{up}'}.json",
              dict(validation_files=[Path(f).name for f in files], up=up, grid=grid, chosen=chosen))
    print("chosen", chosen)


def run(device, data, limit=None, up=1):
    cam = get_cam()
    sfx = "" if up == 1 else f"_up{up}"
    chosen = __import__("json").loads((GAP / "results" / f"mart_tuning{sfx}.json").read_text())["chosen"]
    sweep = PlaneSweep(cam, device)
    marts = {n: Mart(cam, VIEWS32 if n == "32" else torch.arange(484), device, chunk=32 if n == "32" else 44, up=up)
             for n in chosen}
    files = files_of(data)[:limit]
    rows = {"planesweep": [], f"mart32{sfx}": [], f"mart484{sfx}": []}
    preds = {k: [] for k in rows}
    secs = {k: 0.0 for k in rows}
    for f in files:
        sai, gt = load_field(f)
        t0 = time.time()
        xyz, sc = sweep.peaks(sweep.volume(sai))
        secs["planesweep"] += time.time() - t0
        out = {"planesweep": (xyz, sc)}
        for n, m in marts.items():
            c = chosen[n]
            t0 = time.time()
            E = m.reconstruct(sai, iters=c["iters"], mu=c["mu"], blur=c["blur"])
            out[f"mart{n}{sfx}"] = m.peaks(E, c["thr"], zwin=c["zwin"])
            torch.cuda.synchronize()
            secs[f"mart{n}{sfx}"] += time.time() - t0
        for k, (xyz, sc) in out.items():
            m = score(xyz, gt, cam)
            lat = score_lateral(xyz, gt, cam)
            rows[k].append({q: m[q] for q in ("tp", "fp", "fn", "precision", "recall", "f1")} |
                           {"file": Path(f).name, "n_pred": len(xyz), "lateral_f1": lat["lateral_f1"],
                            "lateral_recall": lat["lateral_recall"],
                            "lateral_abs_dz_median_mm": float(np.median(lat["abs_dz_mm"])) if len(lat["abs_dz_mm"]) else None,
                            "axial_mae_um": float(1000 * np.abs(m["err_mm"][:, 2]).mean()) if m["tp"] else None})
            preds[k].append(dict(xyz=xyz, alpha=sc, gt=gt, pred_idx=m["pred_idx"], gt_idx=m["gt_idx"]))
    for k in rows:
        f1 = np.array([r["f1"] for r in rows[k]])
        summary = dict(method=k, data=data, n_fields=len(files), f1=float(f1.mean()), f1_ci95=boot_ci(f1),
                       precision=float(np.mean([r["precision"] for r in rows[k]])),
                       recall=float(np.mean([r["recall"] for r in rows[k]])), seconds_per_field=secs[k] / len(files),
                       lateral_f1=float(np.mean([r["lateral_f1"] for r in rows[k]])),
                       lateral_recall=float(np.mean([r["lateral_recall"] for r in rows[k]])),
                       lateral_abs_dz_median_mm=float(np.mean([r["lateral_abs_dz_median_mm"] for r in rows[k]])),
                       settings=chosen.get(k.replace("mart", "").replace(sfx, ""), {"rel_thr": 0.24}))
        save_json(GAP / "results" / f"{k}__{data}.json", dict(summary=summary, fields=rows[k]))
        np.savez_compressed(GAP / "results" / f"{k}__{data}.npz",
                            **{f"{q}_{i:04d}": p[q] for i, p in enumerate(preds[k]) for q in
                               ("xyz", "alpha", "gt", "pred_idx", "gt_idx")})
        print(k, data, {q: round(v, 4) if isinstance(v, float) else v for q, v in summary.items() if q != "settings"},
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "run"])
    ap.add_argument("--data", nargs="*", default=["sealed_std"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--up", type=int, default=1, help="refinement factor of the MART grid")
    a = ap.parse_args()
    if a.mode == "tune":
        tune("cuda:0", a.up)
    else:
        for d in a.data:
            run("cuda:0", d, a.limit, a.up)
