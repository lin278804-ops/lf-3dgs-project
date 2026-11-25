"""Shared pieces for the gap-closing experiments (B: baselines, C: independence and
robustness, D: depth resolution). Imports the frozen project package `lf_3dgs`
unchanged; every modification lives here as a subclass or a patched function.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Locations. The defaults assume this file sits in <lf-3dgs-project>/paper/experiments/.
PROJECT = Path(os.environ.get("LFPIV_PROJECT", Path(__file__).resolve().parents[2]))    # checkout of lf-3dgs-project
GAP = Path(os.environ.get("LFPIV_WORK", PROJECT / "work"))                              # runs/, data/, results/, fits/, logs/
DS9 = Path(os.environ.get("LFPIV_TRAIN_DATA", GAP / "data" / "lightfield_dataset9"))    # training archive: train/, val/, test/
P7 = GAP / "runs" / "p7_margin_s0"            # pooled network from the staged schedule (principal network)
RECIPE = Path(__file__).with_name("recipe.json")
SCRATCH_S0 = GAP / "runs" / "scratch_s0"      # pooled network, single-stage schedule, first initialization

sys.path.insert(0, str(PROJECT))
from lf_3dgs import run_s2 as s  # noqa: E402
from lf_3dgs.camera import LFCameraParams  # noqa: E402
from lf_3dgs.metrics import nms_gaussians  # noqa: E402
from lf_3dgs.model_v3 import Decoder3DV3, DifferentiableLFProjection, LF3DGSNetV3  # noqa: E402

MODEL_KEYS = ["base_channels", "topk", "n_views_cost", "encode_native", "extract_mode", "bound_offset",
              "view_sample", "skip_mode"]
ALPHA_THR, NMS_VOX, GATE_VOX = 0.05, 1.5, 2.0
OFFSET_BOUND_NODE = 1.5


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def get_cam(f_mla: float = 0.8, depth: int = 64):
    """Camera and reconstruction grid; `depth` is the number of depth planes of the network."""
    cam = s.get_cam()
    cam.f_mla = float(f_mla)
    cam.grid_shape = (int(depth), cam.grid_shape[1], cam.grid_shape[2])
    return cam


def load_state_flexible(model, state):
    """Load weights; depth-dependent skip parameters are interpolated when the depth differs."""
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    own = model.state_dict()
    fixed = {}
    for k, v in state.items():
        if k in own and own[k].shape != v.shape and v.dim() == 5 and v.shape[3:] == (1, 1):
            v = F.interpolate(v.float(), size=(own[k].shape[2], 1, 1), mode="trilinear", align_corners=True)
        fixed[k] = v
    return model.load_state_dict(fixed, strict=True)


# --------------------------------------------------------------------------- models
class NodeOffsetNet(LF3DGSNetV3):
    """Same network; sub-voxel offsets are decoded in the convention of the occupancy label.

    The occupancy target is a Gaussian centred at the continuous voxel coordinate c and
    evaluated at integer nodes k, so the detected peak is the node nearest to c. The
    original decoder returns k + 0.5 + offset with |offset| <= 0.5 and supervises the offset
    at floor(c); for half of the particles the detected node is floor(c) + 1, where the
    required offset lies outside the bound. Here the centre is k + offset with |offset| <= 1.5
    and the offset is supervised at the nodes around round(c) (see offset_node_l1).
    """

    def __init__(self, *a, **kw):
        kw["bound_offset"] = False
        super().__init__(*a, **kw)

    def _extract_topk(self, gauss, cam_params=None):
        gauss["offsets"] = torch.tanh(gauss["offsets"]) * OFFSET_BOUND_NODE
        out = super()._extract_topk(gauss, cam_params)
        if cam_params is not None:
            D, H, W = self.out_shape
            half = out["mu"].new_tensor([
                0.5 * (cam_params.x_range[1] - cam_params.x_range[0]) / W,
                0.5 * (cam_params.y_range[1] - cam_params.y_range[0]) / H,
                0.5 * (cam_params.z_range[1] - cam_params.z_range[0]) / D])
            out["mu"] = out["mu"] - half
        return out


_NEIGH = torch.tensor([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def offset_node_l1(pred_off, centers, cam):
    """L1 on (dx, dy, dz) at the 27 nodes around the node nearest to each centre."""
    if centers.shape[0] == 0:
        return pred_off.new_tensor(0.0)
    D, H, W, _ = pred_off.shape
    vox = cam.physical_to_voxel(centers.to(pred_off.device))            # (N, 3) [z, y, x]
    base = torch.round(vox).long()
    k = base[:, None, :] + _NEIGH.to(base.device)[None]                  # (N, 27, 3)
    lim = torch.tensor([D, H, W], device=base.device)
    valid = ((k >= 0) & (k < lim)).all(-1)
    kc = torch.minimum(k.clamp(min=0), lim - 1)
    target = (vox[:, None, :] - k.float()).flip(-1).clamp(-OFFSET_BOUND_NODE, OFFSET_BOUND_NODE)
    pred = pred_off[kc[..., 0], kc[..., 1], kc[..., 2]]                  # (N, 27, 3) [dx, dy, dz]
    err = (pred - target).abs().sum(-1) * valid.float()
    return err.sum() / (3.0 * valid.float().sum().clamp(min=1.0))


class LiftNet(LF3DGSNetV3):
    """Learned baseline without the optical projection.

    The 32 encoded views are brought onto the object grid by a single focal-plane lateral
    registration, concatenated in a fixed order and mapped to depth planes by learned
    two-dimensional convolutions (views-to-depth lifting, as in learned light-field volume
    reconstruction). The view-dependent parallax is never given to the network. Encoder, three-dimensional decoder, skip stream, heads, losses and
    decoding are those of the main network; the camera calibration never enters the network.
    """

    LIFT_C = 16
    HIDDEN = 512
    K2 = 3            # kernel of the second lifting convolution

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        bc = self.head_3d.in_channels
        D = self.out_shape[0]
        self.lift = nn.Sequential(
            nn.Conv2d(self.n_views_cost * bc, self.HIDDEN, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.HIDDEN), nn.ReLU(inplace=True),
            nn.Conv2d(self.HIDDEN, self.LIFT_C * D, self.K2, padding=self.K2 // 2, bias=False),
            nn.BatchNorm2d(self.LIFT_C * D), nn.ReLU(inplace=True))
        self.decoder = Decoder3DV3(self.LIFT_C, bc, 2 * bc)

    def _build_cost_volume(self, feat_down, s_idx, t_idx, cam_params, cv_shape):
        B, K, C, Hf, Wf = feat_down.shape
        D, H, W = cv_shape
        dev = feat_down.device
        # Focal-plane registration only: one lateral map (image inversion and magnification at
        # z = 0, identical for every view) brings the features onto the object grid. No
        # view-dependent or depth-dependent geometry is used.
        yy, xx = torch.meshgrid(torch.linspace(*cam_params.y_range, H, device=dev),
                                torch.linspace(*cam_params.x_range, W, device=dev), indexing="ij")
        pts = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(-1, 3)
        with torch.no_grad():
            u, v = DifferentiableLFProjection(cam_params).to(dev)(pts, s_idx[:1], t_idx[:1])
        grid = torch.stack([2.0 * v / max(cam_params.num_mla_x - 1, 1) - 1.0,
                            2.0 * u / max(cam_params.num_mla_y - 1, 1) - 1.0], -1).reshape(1, H, W, 2)
        reg = F.grid_sample(feat_down.reshape(B * K, C, Hf, Wf), grid.expand(B * K, H, W, 2).to(feat_down.dtype),
                            mode="bilinear", padding_mode="zeros", align_corners=True)
        y = self.lift(reg.reshape(B, K * C, H, W))
        return y.reshape(B, self.LIFT_C, D, H, W)


class ProjLiftNet(LF3DGSNetV3):
    """Projection-guided network with the capacity of the lifting baseline.

    The projection lookup of the main network builds the mean/variance volume (2C channels on D
    planes). Its planes are then stacked as channels and processed by learned two-dimensional
    convolutions of the same widths as in LiftNet (512 and 16 x D channels), and the result enters
    the same decoder. The only difference from LiftNet is the input of the learned lifting:
    features sampled along the calibrated parallax instead of features registered at the focal plane.
    """

    LIFT_C = 16
    HIDDEN = 512

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        bc = self.head_3d.in_channels
        D = self.out_shape[0]
        self.lift = nn.Sequential(
            nn.Conv2d(2 * bc * D, self.HIDDEN, 1, bias=False), nn.BatchNorm2d(self.HIDDEN), nn.ReLU(inplace=True),
            nn.Conv2d(self.HIDDEN, self.HIDDEN, 3, padding=1, bias=False), nn.BatchNorm2d(self.HIDDEN), nn.ReLU(inplace=True),
            nn.Conv2d(self.HIDDEN, self.LIFT_C * D, 3, padding=1, bias=False), nn.BatchNorm2d(self.LIFT_C * D),
            nn.ReLU(inplace=True))
        self.decoder = Decoder3DV3(self.LIFT_C, bc, 2 * bc)

    def _build_cost_volume(self, feat_down, s_idx, t_idx, cam_params, cv_shape):
        cv = super()._build_cost_volume(feat_down, s_idx, t_idx, cam_params, cv_shape)      # (B, 2C, D, H, W)
        B, C2, D, H, W = cv.shape
        y = self.lift(cv.reshape(B, C2 * D, H, W))
        return y.reshape(B, self.LIFT_C, D, H, W)


class ViewLiftNet(LF3DGSNetV3):
    """Projection-aligned fusion that keeps the views apart.

    The lookup samples every encoded view at the projection of each candidate centre, as in the
    main network, but the views are not reduced to their mean and variance. On each depth plane the
    aligned features of all views are concatenated in a fixed order and fused by two-dimensional
    convolutions whose weights are shared by all planes. The geometry stays in the lookup (so a
    calibration can be exchanged) and the fusion can use which view saw what.
    """

    LIFT_C = 16
    HIDDEN = 128

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        bc = self.head_3d.in_channels
        h = self.HIDDEN
        self.fuse = nn.Sequential(
            nn.Conv2d(self.n_views_cost * bc, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            nn.Conv2d(h, self.LIFT_C, 3, padding=1, bias=False), nn.BatchNorm2d(self.LIFT_C), nn.ReLU(inplace=True))
        self.decoder = Decoder3DV3(self.LIFT_C, bc, 2 * bc)

    def _build_cost_volume(self, feat_down, s_idx, t_idx, cam_params, cv_shape):
        B, K, C, Hf, Wf = feat_down.shape
        D, H, W = cv_shape
        dev = feat_down.device
        z = torch.linspace(*cam_params.z_range, D, device=dev)
        y = torch.linspace(*cam_params.y_range, H, device=dev)
        x = torch.linspace(*cam_params.x_range, W, device=dev)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        with torch.no_grad():
            u, v = DifferentiableLFProjection(cam_params).to(dev)(pts, s_idx, t_idx)            # (K, N)
        grid = torch.stack([2.0 * v / max(cam_params.num_mla_x - 1, 1) - 1.0,
                            2.0 * u / max(cam_params.num_mla_y - 1, 1) - 1.0], -1)              # (K, N, 2)
        out = []
        for b in range(B):                                   # batch of one in practice
            samp = F.grid_sample(feat_down[b], grid.reshape(K, 1, -1, 2).to(feat_down.dtype), mode="bilinear",
                                 padding_mode="zeros", align_corners=True)                      # (K, C, 1, N)
            samp = samp.reshape(K * C, D, H, W).permute(1, 0, 2, 3)                              # (D, K*C, H, W)
            out.append(self.fuse(samp).permute(1, 0, 2, 3))                                      # (LIFT_C, D, H, W)
        return torch.stack(out, 0)


class LiftCalNet(LiftNet):
    """Lifting baseline that is told the calibration: the microlens focal length modulates the lifted
    features through learned per-channel scales and shifts (the camera never enters as geometry)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        D = self.out_shape[0]
        self.cal = nn.Sequential(nn.Linear(1, 64), nn.ReLU(inplace=True), nn.Linear(64, 2 * (self.HIDDEN + self.LIFT_C * D)))
        nn.init.zeros_(self.cal[-1].weight)
        nn.init.zeros_(self.cal[-1].bias)
        self.cam_f = 0.8

    def _build_cost_volume(self, feat_down, s_idx, t_idx, cam_params, cv_shape):
        B, K, C, Hf, Wf = feat_down.shape
        D, H, W = cv_shape
        dev = feat_down.device
        yy, xx = torch.meshgrid(torch.linspace(*cam_params.y_range, H, device=dev),
                                torch.linspace(*cam_params.x_range, W, device=dev), indexing="ij")
        pts = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(-1, 3)
        with torch.no_grad():
            u, v = DifferentiableLFProjection(cam_params).to(dev)(pts, s_idx[:1], t_idx[:1])
        grid = torch.stack([2.0 * v / max(cam_params.num_mla_x - 1, 1) - 1.0,
                            2.0 * u / max(cam_params.num_mla_y - 1, 1) - 1.0], -1).reshape(1, H, W, 2)
        reg = F.grid_sample(feat_down.reshape(B * K, C, Hf, Wf), grid.expand(B * K, H, W, 2).to(feat_down.dtype),
                            mode="bilinear", padding_mode="zeros", align_corners=True).reshape(B, K * C, H, W)
        f = torch.full((1, 1), (float(cam_params.f_mla) - 0.95) / 0.15, device=dev)
        m = self.cal(f)[0]
        n1, n2 = self.HIDDEN, self.LIFT_C * D
        g1, b1, g2, b2 = m[:n1], m[n1:2 * n1], m[2 * n1:2 * n1 + n2], m[2 * n1 + n2:]
        h = self.lift[2](self.lift[1](self.lift[0](reg)) * (1 + g1.view(1, -1, 1, 1)) + b1.view(1, -1, 1, 1))
        h = self.lift[5](self.lift[4](self.lift[3](h)) * (1 + g2.view(1, -1, 1, 1)) + b2.view(1, -1, 1, 1))
        return h.reshape(B, self.LIFT_C, D, H, W)


def build_model(kind, cfg, cam):
    cls = {"v3": LF3DGSNetV3, "node": NodeOffsetNet, "lift": LiftNet, "projlift": ProjLiftNet, "viewlift": ViewLiftNet,
           "liftcal": LiftCalNet}[kind]
    if kind == "viewlift" and cfg.get("lift_hidden"):
        cls = type("ViewLiftNetSized", (ViewLiftNet,), {"HIDDEN": int(cfg["lift_hidden"])})
    if kind == "lift" and cfg.get("lift_hidden"):          # size-matched variant
        cls = type("LiftNetSized", (LiftNet,), {"HIDDEN": int(cfg["lift_hidden"]), "K2": int(cfg.get("lift_k2", 3))})
    return cls(out_shape=tuple(cam.grid_shape), **{k: cfg[k] for k in MODEL_KEYS})


def load_model(run_dir, device, which="best.pt", topk=None):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    kind = cfg.get("model_kind", "v3")
    for k, v in (("encode_native", True), ("extract_mode", "peak"), ("bound_offset", True),
                 ("view_sample", "uniform"), ("skip_mode", "broadcast")):
        cfg.setdefault(k, v)
    if topk is not None:
        cfg["topk"] = int(topk)
    cam = get_cam(depth=int(cfg.get("grid_depth", 64)))
    model = build_model(kind, cfg, cam).to(device)
    load_state_flexible(model, torch.load(run_dir / which, map_location=device, weights_only=False))
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- noise
def peak_level(sai):
    """99.9th percentile of the brightest eighth of views (cheap proxy for the peak grey level)."""
    v = sai.reshape(sai.shape[0], -1)[:, :: max(1, sai[0].numel() // 2_000_000)]
    return torch.quantile(v.float(), 0.999, dim=1).clamp(min=1e-3)


def add_noise(sai, snr_db, gen, gain=1.0, background=0.0):
    """Additive Gaussian noise at a peak SNR (p99.9 grey level / sigma), after gain and offset."""
    out = sai * gain + background
    if snr_db is not None:
        sigma = peak_level(out) / (10.0 ** (snr_db / 20.0))
        out = out + torch.randn(out.shape, generator=gen, device=out.device) * sigma.view(-1, 1, 1, 1)
    return out.clamp(0.0, 1.0)


def condition(sai, target=94.0 / 255.0, k=3.0, blur=3.0):
    """Input conditioning used for recorded frames: per view, subtract a smooth background (Gaussian,
    sigma `blur` px), remove a noise floor of k robust standard deviations of the stack, and scale the
    stack so that its 99.9th percentile equals that of the simulated fields. sai: (V, H, W) in [0, 1]."""
    r = max(1, int(3 * blur))
    g = torch.exp(-torch.arange(-r, r + 1, device=sai.device, dtype=sai.dtype) ** 2 / (2 * blur ** 2))
    g = g / g.sum()
    x = sai[:, None]
    bg = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    bg = F.conv2d(F.pad(bg, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))
    res = (x - bg)[:, 0]
    sub = res.reshape(-1)[:: max(1, res.numel() // 2_000_000)]
    sigma = 1.4826 * (sub - sub.median()).abs().median()
    res = (res - k * sigma).clamp(min=0)
    p = torch.quantile(res.reshape(-1)[:: max(1, res.numel() // 2_000_000)], 0.999)
    return (res * (target / p.clamp(min=1e-6))).clamp(0, 1)


def augment(sai, gen):
    """Training-time randomization: gain, uniform background and sensor noise."""
    u = torch.rand(4, generator=gen, device=sai.device)
    gain = 0.7 + 0.6 * float(u[0])
    background = 0.02 * float(u[1]) if float(u[2]) < 0.5 else 0.0
    snr = None if float(u[3]) < 0.15 else 20.0 + 30.0 * float(torch.rand(1, generator=gen, device=sai.device))
    return add_noise(sai, snr, gen, gain, background)


# --------------------------------------------------------------------------- inference and scoring
@torch.no_grad()
def detect(model, cam, sai01, device, alpha_thr=ALPHA_THR, nms=NMS_VOX, return_volume=False, cam_score=None):
    """sai01: float tensor (484, H, W) in [0, 1]. Returns xyz (mm), voxel coordinates, alpha.

    `cam` carries the calibration and grid of the network; suppression always uses the
    64-plane scoring grid (`cam_score`) so that the rule is identical for every model.
    """
    cam_score = cam_score or get_cam()
    V = cam.px_per_mla
    lf = sai01.to(device).reshape(1, V, V, cam.num_mla_y, cam.num_mla_x)
    out = model(lf, out_hw=(cam.grid_shape[1], cam.grid_shape[2]), cam_params=cam)
    mu = out["mu"][0].float().cpu()
    alpha = torch.sigmoid(out["logit_alpha"][0].float().cpu()).numpy()
    keep = alpha >= alpha_thr
    mu, alpha = mu[keep], alpha[keep]
    vox = cam_score.physical_to_voxel(mu).numpy().astype(np.float32)
    if len(vox) > 1:
        k = nms_gaussians(vox, alpha, merge_radius=nms)
        mu, vox, alpha = mu[k], vox[k], alpha[k]
    res = (mu.numpy().astype(np.float32), vox, alpha.astype(np.float32))
    return res + (out["volume"],) if return_volume else res


def to_vox(xyz, cam=None):
    cam = cam or get_cam()
    return cam.physical_to_voxel(torch.as_tensor(np.asarray(xyz), dtype=torch.float32)).numpy()


def hungarian(pred_vox, gt_vox, gate=GATE_VOX):
    if len(pred_vox) == 0 or len(gt_vox) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.sqrt(((pred_vox[:, None, :] - gt_vox[None, :, :]) ** 2).sum(2))
    c = d.copy()
    c[d > gate] = 1e6
    ri, ci = linear_sum_assignment(c)
    k = d[ri, ci] <= gate
    return ri[k], ci[k]


def score(pred_xyz, gt_xyz, cam=None):
    cam = cam or get_cam()
    pv, gv = to_vox(pred_xyz, cam).astype(np.float32), to_vox(gt_xyz, cam).astype(np.float32)
    ri, ci = hungarian(pv, gv)
    tp, fp, fn = len(ri), len(pv) - len(ri), len(gv) - len(ri)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    err = np.asarray(pred_xyz)[ri] - np.asarray(gt_xyz)[ci] if tp else np.zeros((0, 3), np.float32)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "pred_idx": ri, "gt_idx": ci, "err_mm": err}


def score_lateral(pred_xyz, gt_xyz, cam=None, lateral_gate=GATE_VOX):
    """Detection scored on lateral position only (depth ignored), with the depth error of matches."""
    cam = cam or get_cam()
    pv, gv = to_vox(pred_xyz, cam), to_vox(gt_xyz, cam)
    if len(pv) == 0:
        return {"lateral_f1": 0.0, "lateral_recall": 0.0, "lateral_precision": 0.0, "abs_dz_mm": np.zeros(0)}
    dl = np.sqrt(((pv[:, None, 1:] - gv[None, :, 1:]) ** 2).sum(2))
    c = dl + 0.05 * np.abs(pv[:, None, 0] - gv[None, :, 0])
    c[dl > lateral_gate] = 1e6
    ri, ci = linear_sum_assignment(c)
    k = dl[ri, ci] <= lateral_gate
    ri, ci = ri[k], ci[k]
    p, r = len(ri) / max(len(pv), 1), len(ri) / max(len(gv), 1)
    return {"lateral_f1": 2 * p * r / max(p + r, 1e-12), "lateral_recall": r, "lateral_precision": p,
            "abs_dz_mm": np.abs(np.asarray(pred_xyz)[ri, 2] - np.asarray(gt_xyz)[ci, 2])}


def load_field(path):
    with np.load(path, allow_pickle=False) as z:
        sai = z["sai_stack"].astype(np.float32)
        gt = z["particle_centers"].astype(np.float32)
    if sai.max() > 1.0:
        sai = sai / 255.0
    return torch.from_numpy(sai), gt


def boot_ci(values, n=10000, seed=20261121):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    m = values[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
