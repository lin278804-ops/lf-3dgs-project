"""Training driver for the gap experiments.

  python train.py --name seed1 --seed 1                      # from scratch, archived 40-epoch recipe
  python train.py --name lift_s0 --seed 0 --model lift       # learned baseline without the projection
  python train.py --name p7_ctrl --init p7 --epochs 20 --lr 3e-5
  python train.py --name p7_node --init p7 --epochs 20 --lr 3e-5 --model node
  python train.py --name p7_aug  --init p7 --epochs 20 --lr 3e-5 --augment

From-scratch runs use the recipe of the archived seed-0 run; fine-tunes use the configuration of
the principal network. Resumable; writes config.json, history.json, best.pt (validation F1), last.pt.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gapcommon import (DS9, GAP, MODEL_KEYS, P7, RECIPE, augment, build_model, get_cam, load_json,
                       load_state_flexible, offset_node_l1, s, save_json, sha)
from lf_3dgs.dataset import LF3DGSDataset
from lf_3dgs.renderer import GaussianSplatRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["v3", "node", "lift", "projlift", "viewlift", "liftcal"], default="v3")
    ap.add_argument("--multi", default=None, help="comma-separated microlens focal lengths: train on data/<prefix><f*100>/")
    ap.add_argument("--multi-prefix", default="mc_f")
    ap.add_argument("--chunk", type=int, default=24, help="multi-camera training: fields per camera between switches")
    ap.add_argument("--init", default=None, help="'p7' or a run directory to warm-start from")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: number of training fields")
    ap.add_argument("--train-fields", type=int, default=None, help="low-data regime: first N training fields")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--lift-hidden", type=int, default=None, help="width of the lifting layer (default 512)")
    ap.add_argument("--lift-k2", type=int, default=3)
    ap.add_argument("--data", default=str(DS9), help="dataset root with train/ and val/")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--depth", type=int, default=64, help="number of depth planes of the network grid")
    args = ap.parse_args()

    dest = GAP / "runs" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "trained.json").exists():
        print("already trained", args.name)
        return

    init_dir = P7 if args.init == "p7" else (Path(args.init) if args.init else None)
    cfg = load_json(init_dir / "config.json") if init_dir else load_json(RECIPE)
    for k, v in (("encode_native", True), ("extract_mode", "peak"), ("bound_offset", True),
                 ("view_sample", "uniform")):
        cfg.setdefault(k, v)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.topk:
        cfg["topk"] = args.topk
    cfg.update(seed=args.seed, model_kind=args.model, augment=bool(args.augment), grid_depth=args.depth,
               data=args.data, init=str(init_dir) if init_dir else None, train_fields=args.train_fields, multi=args.multi)
    if args.lift_hidden:
        cfg.update(lift_hidden=args.lift_hidden, lift_k2=args.lift_k2)
    save_json(dest / "config.json", cfg)

    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda:0")
    cam = get_cam(depth=args.depth)
    model = build_model(args.model, cfg, cam).to(dev)
    if init_dir:
        print("init", init_dir, load_state_flexible(
            model, torch.load(init_dir / "best.pt", map_location=dev, weights_only=False)))
    if args.model == "node":
        s.offset_peak_l1 = offset_node_l1          # consistent supervision for the node convention
    n_params = sum(p.numel() for p in model.parameters())
    renderer = GaussianSplatRenderer(cam).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=0.01 * cfg["lr"])
    scaler = torch.cuda.amp.GradScaler()

    cams = None
    if args.multi:
        cams = []
        for ftxt in args.multi.split(","):
            fv = float(ftxt)
            root = GAP / "data" / f"{args.multi_prefix}{int(round(fv * 100)):03d}"
            c = get_cam(fv, depth=args.depth)
            cams.append(dict(f=fv, cam=c, renderer=GaussianSplatRenderer(c).to(dev), train=LF3DGSDataset(str(root), "train"),
                             val=LF3DGSDataset(str(root), "val")))
        args.data = str(GAP / "data" / f"{args.multi_prefix}{int(round(cams[0]['f'] * 100)):03d}")
    train_ds = LF3DGSDataset(args.data, "train")
    val_ds = LF3DGSDataset(args.data, "val")
    if args.train_fields:
        train_ds.files = train_ds.files[: args.train_fields]
    if args.limit:
        train_ds.files = train_ds.files[: args.limit]
        val_ds.files = val_ds.files[: max(4, args.limit // 4)]

    def loader(ds, workers):
        return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=s.collate_fn, num_workers=workers,
                          pin_memory=True, persistent_workers=False, prefetch_factor=2)

    val_loader = loader(val_ds, 2)
    history, best, start = [], -1.0, 1
    resume = dest / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        start = len(history) + 1
        best = max(h["val"]["f1"] for h in history)
    save_json(dest / "meta.json", dict(name=args.name, n_params=n_params, n_train=len(train_ds), n_val=len(val_ds),
                                       gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__,
                                       code_sha256={f: sha(Path(__file__).parent / f) for f in
                                                    ("train.py", "gapcommon.py")},
                                       started=time.strftime("%Y-%m-%d %H:%M:%S")))
    gen = torch.Generator(device=dev)
    for ep in range(start, cfg["epochs"] + 1):
        t0 = time.monotonic()
        rng_seed = 47000 + 1000 * args.seed + ep
        order = np.random.default_rng(rng_seed).permutation(len(train_ds)).tolist()
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        gen.manual_seed(rng_seed)
        base = loader(Subset(train_ds, order), 4)

        def batches():
            for sai, centers, meta, paths in base:
                if args.augment:
                    sai = augment(sai.to(dev), gen)
                yield sai, centers, meta, paths

        validated = ep % args.val_every == 0 or ep == cfg["epochs"]
        if cams is None:
            tm = s.train_one_epoch(model, renderer, batches(), opt, cam, cfg, dev, scaler)
            vm = s.validate_model(model, val_loader, cam, dev) if validated else dict(history[-1]["val"]) if history else \
                dict(precision=0.0, recall=0.0, f1=0.0, mean_dist=0.0)
        else:
            rng = np.random.default_rng(rng_seed)
            orders = [rng.permutation(len(c["train"])).tolist() for c in cams]
            n_chunks = max(len(o) for o in orders) // args.chunk
            parts = []
            for k in range(n_chunks):
                for ci in rng.permutation(len(cams)):
                    c = cams[ci]
                    idx = orders[ci][k * args.chunk:(k + 1) * args.chunk]
                    if not idx:
                        continue
                    model.cam_f = c["f"]
                    parts.append((len(idx), s.train_one_epoch(model, c["renderer"], loader(Subset(c["train"], idx), 2), opt, c["cam"],
                                                              cfg, dev, scaler)))
            tot = sum(n for n, _ in parts)
            tm = {key: float(sum(n * m[key] for n, m in parts) / tot) for key in parts[0][1]}
            if validated:
                vms = []
                for c in cams:
                    model.cam_f = c["f"]
                    vms.append(s.validate_model(model, loader(c["val"], 2), c["cam"], dev))
                vm = {key: float(np.mean([m[key] for m in vms])) for key in vms[0]}
                vm["f1_by_camera"] = [float(m["f1"]) for m in vms]
            else:
                vm = dict(history[-1]["val"]) if history else dict(precision=0.0, recall=0.0, f1=0.0, mean_dist=0.0)
        lr_now = opt.param_groups[0]["lr"]
        sched.step()
        rec = dict(epoch=ep, lr=lr_now, train=tm, val=vm, validated=validated, seconds=time.monotonic() - t0)
        history.append(rec)
        improved = validated and vm["f1"] > best
        best = max(best, vm["f1"])
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), history=history), str(resume) + ".tmp")
        os.replace(str(resume) + ".tmp", resume)
        if improved:
            torch.save(model.state_dict(), str(dest / "best.pt") + ".tmp")
            os.replace(str(dest / "best.pt") + ".tmp", dest / "best.pt")
        save_json(dest / "history.json", history)
        print("EPOCH", args.name, ep, round(rec["seconds"]), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vm.items()},
              {k: round(v, 4) for k, v in tm.items()}, flush=True)
    torch.save(model.state_dict(), dest / "last.pt")
    save_json(dest / "trained.json", dict(best_epoch=max(history, key=lambda h: h["val"]["f1"])["epoch"],
                                          best_val_f1=best, epochs=len(history),
                                          best_sha256=sha(dest / "best.pt"), last_sha256=sha(dest / "last.pt")))
    resume.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
