#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets_py.patch_dataset import PatchDataset
from models import build_loss, build_priornet
from train.common import (Logger, MetricWriter, binary_metrics_from_logits,
                          load_cfg, print_env_and_sanity, save_cfg_snapshot,
                          save_ckpt, set_seed, setup_perf, smoke_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prior_v1.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 个 patch（smoke）")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--proc", default=None, help="覆盖 config 里的 processed 目录")
    ap.add_argument("--raw", default=None, help="覆盖 config 里的原始图像目录")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.proc:
        cfg["proc"] = args.proc
    if args.raw:
        cfg["raw"] = args.raw
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["amp"] = cfg.get("amp", False) or args.amp

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    name = args.name or cfg.get("name", "prior_v1")
    out_dir = Path(cfg.get("out_dir", "runs/priornet")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train.log")
    log.log("=" * 70)
    log.log("train_prior  name=%s  config=%s" % (name, args.config))
    log.log("=" * 70)
    save_cfg_snapshot(cfg, out_dir)

    proc = cfg.get("proc", "datasets/EVD4UAV_processed")
    raw = cfg.get("raw", "datasets/EVD4UAV/raw")

    manifest = cfg.get("patch_manifest", "patch_manifest.csv")
    log.log("patch manifest = %s" % manifest)

    tr = PatchDataset(proc=proc, raw=raw, split="train", task="prior",
                      augment=cfg.get("augment", True), seed=seed,
                      class_aware=bool(cfg.get("class_aware", False)),
                      manifest=manifest, soft_dir=cfg.get("soft_dir"))
    va = PatchDataset(proc=proc, raw=raw, split="val", task="prior",
                      augment=False, seed=seed,
                      class_aware=bool(cfg.get("class_aware", False)),
                      manifest=manifest, soft_dir=cfg.get("soft_dir"))
    if args.limit:
        tr = Subset(tr, range(min(args.limit, len(tr))))
        va = Subset(va, range(min(max(32, args.limit // 5), len(va))))

    bs = int(cfg.get("batch_size", 8))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0)

    model = build_priornet(cfg.get("model", {"arch": "litehr"}))
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        log.log("resumed from %s" % args.resume)
    FMT = setup_perf(log)
    dev = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if dev == "cuda":
        model = model.to(memory_format=FMT)

    loss_fn = build_loss(cfg.get("loss", {"main": "focal", "lambda_dice": 1.0}))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.get("epochs", 30)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp")) and dev == "cuda")

    print_env_and_sanity(log, cfg, model, tr, va, bs, 3, args.num_workers)
    smoke_test(log, model, tl, loss_fn, dev, amp=bool(cfg.get("amp")), in_channels=3)

    mw = MetricWriter(out_dir / "metrics.csv",
                      ["epoch", "lr", "train_loss", "val_loss", "precision",
                       "recall", "iou", "dice", "f1", "epoch_sec"])
    best = -1.0
    epochs = int(cfg.get("epochs", 30))
    log_interval = int(cfg.get("log_interval", 20))
    _loss_cfg = cfg.get("loss", {}) or {}
    lambda_bg = float(_loss_cfg.get("lambda_bg", 1.0))
    lambda_balance = float(_loss_cfg.get("lambda_balance", 0.0))
    class_aware = bool(cfg.get("class_aware", False))
    lambda_cls = float(_loss_cfg.get("lambda_cls", 0.5))
    cls_weights = None
    if class_aware:
        from collections import Counter
        _cnt = Counter(int(v) for v in tr.cls_map.values() if v >= 0)
        _ncls = int(getattr(model, "N_CLASS", 3))
        _w = torch.tensor([1.0 / (_cnt.get(c, 1) ** 0.5) for c in range(_ncls)],
                          device=dev)
        cls_weights = _w / _w.min()
        print(f"[class-balanced] cls_weights={cls_weights.tolist()} "
              f"(car-dominant mitigation; freq={dict(_cnt)})", flush=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for i, batch in enumerate(tl):
            if class_aware:
                x, y, alt, yc = batch
                yc = yc.to(dev, non_blocking=True)
            else:
                x, y, alt = batch
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if dev == "cuda":
                x = x.contiguous(memory_format=FMT)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",
                                enabled=bool(cfg.get("amp")) and dev == "cuda"):
                if getattr(model, "dual", False):
                    res = model.forward_dual(
                        x, alt if getattr(model, "accepts_altitude", False) else None)
                    fg = res[0]
                    loss = loss_fn(fg, y)
                    if len(res) >= 2 and res[1] is not None:
                        bg = res[1]
                        loss = loss + lambda_bg * loss_fn(bg, 1.0 - y)
                if getattr(model, "class_aware", False) and class_aware and len(res) >= 3:
                    cls = res[2]
                    fg_mask = (y > 0.5).view(y.size(0), -1).sum(1) > 0
                    known = (yc >= 0) & (yc < getattr(model, "N_CLASS", 3))
                    cls_mask = fg_mask & known
                    if cls_mask.any():
                        loss = loss + lambda_cls * F.cross_entropy(
                            cls[cls_mask], yc[cls_mask].long(),
                            weight=cls_weights)
                else:
                    out = model(x, alt) if getattr(model, "accepts_altitude", False) else model(x)
                    loss = loss_fn(out, y)
                if lambda_balance > 0 and hasattr(model, "load_balance_loss"):
                    lb = model.load_balance_loss()
                    if lb is not None:
                        loss = loss + lambda_balance * lb
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if i % log_interval == 0:
                log.log("  ep%d it%4d/%4d loss=%.6f" % (ep, i, len(tl), loss.item()))
        sched.step()

        model.eval()
        vloss, acc = [], {"precision": [], "recall": [], "iou": [], "dice": [], "f1": []}
        with torch.no_grad():
            for batch in vl:
                if class_aware:
                    x, y, alt, _ = batch
                else:
                    x, y, alt = batch
                x, y = x.to(dev), y.to(dev)
                if getattr(model, "dual", False):
                    res = model.forward_dual(
                        x, alt if getattr(model, "accepts_altitude", False) else None)
                    out = res[0]
                else:
                    out = model(x, alt) if getattr(model, "accepts_altitude", False) else model(x)
                vloss.append(float(loss_fn(out, y)))
                m = binary_metrics_from_logits(out, (y > 0.5).float())
                for k in acc:
                    acc[k].append(m[k])
        row = {
            "epoch": ep,
            "lr": sched.get_last_lr()[0],
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(vloss)),
            **{k: float(np.mean(v)) for k, v in acc.items()},
            "epoch_sec": round(time.time() - t0, 1),
        }
        mw.add(row)
        log.log("EPOCH %d  train=%.6f  val=%.6f  iou=%.4f  dice=%.4f  rec=%.4f  (%.0fs)"
                % (ep, row["train_loss"], row["val_loss"], row["iou"],
                   row["dice"], row["recall"], row["epoch_sec"]))

        save_ckpt(model, out_dir / "last.pth", {"epoch": ep, "cfg": cfg})
        if row["dice"] > best:
            best = row["dice"]
            save_ckpt(model, out_dir / "best.pth",
                      {"epoch": ep, "dice": best, "cfg": cfg})
            log.log("  -> new best dice=%.5f" % best)

    mw.close()
    log.log("训练结束。best dice=%.5f  输出目录=%s" % (best, out_dir))
    log.close()

if __name__ == "__main__":
    main()
