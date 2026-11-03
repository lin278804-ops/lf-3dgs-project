"""CPU smoke check for the released LF-3DGS pipeline.

Covers the four pieces of the mature scheme, on a tiny camera so it finishes
in seconds and does not need a GPU or a dataset:

1. Analytic light-field projection.
2. Stage-1 differentiable Gaussian splatting, including one backward step.
3. Stage-2 LF3DGSNetV3 forward (cost volume, peak extraction, physical centres).
4. Frozen Hungarian detection match and Hungarian two-frame tracking.

Exit code 0 prints SMOKE_OK. Any failed assertion exits non-zero.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lf_3dgs.camera import LFCameraParams
from lf_3dgs.model_v3 import LF3DGSNetV3
from lf_3dgs.nmi_sprint.bootstrap_ci import bootstrap_mean_ci
from lf_3dgs.nmi_sprint.evaluator import hungarian_match, metrics_from_match
from lf_3dgs.piv.piv_tracking import hungarian_tracking
from lf_3dgs.renderer import GaussianSplatRenderer


def tiny_camera() -> LFCameraParams:
    return LFCameraParams(
        num_mla_x=32,
        num_mla_y=32,
        px_per_mla=4,
        grid_shape=(4, 32, 32),
    )


def check_projection(cam: LFCameraParams) -> None:
    from lf_3dgs.camera import DifferentiableLFProjection

    proj = DifferentiableLFProjection(cam)
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]], dtype=torch.float32)
    s = torch.tensor([2], dtype=torch.long)
    t = torch.tensor([2], dtype=torch.long)
    u, v = proj(pts, s, t)
    assert u.shape == (1, 2) and v.shape == (1, 2)
    assert torch.isfinite(u).all() and torch.isfinite(v).all()
    vox = cam.physical_to_voxel(pts)
    assert vox.shape == (2, 3)
    print("  projection ok", float(u[0, 0]), float(v[0, 0]))


def check_stage1_renderer(cam: LFCameraParams) -> None:
    renderer = GaussianSplatRenderer(cam)
    n = 4
    mu = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-1.0, 0.2, 0.4], [0.3, -0.8, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    log_sigma = torch.full((n,), 0.5, requires_grad=True)
    logit_alpha = torch.full((n,), 1.0, requires_grad=True)
    s_idx = torch.tensor([1, 2], dtype=torch.long)
    t_idx = torch.tensor([1, 2], dtype=torch.long)
    images = renderer(mu, log_sigma, logit_alpha, s_idx, t_idx)
    assert images.shape[0] == 2
    assert images.shape[-2] == cam.num_mla_y
    assert images.shape[-1] == cam.num_mla_x
    assert torch.isfinite(images).all()
    assert float(images.max()) > 0.0
    images.mean().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    print("  stage1 renderer ok", tuple(images.shape), float(images.max()))


def check_stage2(cam: LFCameraParams) -> None:
    torch.manual_seed(0)
    net = LF3DGSNetV3(
        out_shape=(4, 32, 32),
        base_channels=4,
        topk=8,
        n_views_cost=2,
        encode_native=True,
        extract_mode="peak",
        bound_offset=True,
        view_sample="uniform",
        skip_mode="depth_aware",
    )
    net.eval()
    lf4d = torch.rand(1, 4, 4, 32, 32)
    with torch.no_grad():
        out = net(lf4d, cam_params=cam)
    mu = out["mu"]
    assert mu.shape == (1, 8, 3), mu.shape
    assert torch.isfinite(mu).all()
    assert out["logit_alpha"].shape == (1, 8)
    assert "volume" in out
    print("  stage2 forward ok", tuple(mu.shape), float(mu.abs().mean()))


def check_hungarian() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    pred = gt + np.array([[0.1, 0.0, 0.0], [-0.1, 0.1, 0.0], [0.0, -0.1, 0.05]], dtype=np.float32)
    matched = metrics_from_match(hungarian_match(pred, gt, match_radius=1.0))
    assert matched["tp"] == 3 and matched["f1"] == 1.0, matched
    missed = metrics_from_match(hungarian_match(pred + 5.0, gt, match_radius=1.0))
    assert missed["tp"] == 0 and missed["fn"] == 3, missed
    ci = bootstrap_mean_ci([0.8, 0.9, 0.85, 0.88])
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    print("  hungarian ok", matched["f1"], round(ci["mean"], 4))


def check_tracking() -> None:
    t1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    shift = np.array([0.2, -0.1, 0.0])
    t2 = t1 + shift
    tr = hungarian_tracking(t1, t2, max_disp=1.0)
    assert tr["n_matched"] == 2, tr
    disp = np.asarray(tr["displacements"])
    assert np.allclose(disp, shift, atol=1e-6), disp
    print("  tracking ok", tr["n_matched"], disp[0].tolist())


def main() -> None:
    print("LF-3DGS smoke check (CPU)")
    cam = tiny_camera()
    check_projection(cam)
    check_stage1_renderer(cam)
    check_stage2(cam)
    check_hungarian()
    check_tracking()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
