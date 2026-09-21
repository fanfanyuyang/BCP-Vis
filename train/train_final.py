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
from torch.utils.data import DataLoader, Subset

from datasets_py.patch_dataset import PatchDataset
from models import build_final_cnn, build_loss
from train.common import (Logger, MetricWriter, binary_metrics_from_logits,
                          load_cfg, print_env_and_sanity, save_cfg_snapshot,
                          save_ckpt, set_seed, setup_perf, smoke_test)

PRIOR_MODES = {
    "E0": ("none", 3), "E1": ("zero", 4), "E2": ("random", 4),
    "E3": ("predicted", 4), "E4": ("predicted", 4), "E5": ("predicted", 4),
    "E6": ("predicted", 4), "E7": ("oracle", 4),
    "E8": ("predicted", 4),
    "E9": ("predicted", 4),
    "E10": ("predicted", 4),
    "E11": ("predicted", 4),
    "E12": ("predicted", 4),
    "E13": ("predicted", 4), "E14": ("predicted", 4), "E13b": ("predicted", 4), "E16": ("predicted", 4), "E17": ("predicted", 4), "E18": ("predicted", 4),
             "E19": ("predicted", 4), "E20": ("predicted2", 5), "E21": ("predicted", 4), "E22": ("predicted2", 5),
             "E23": ("predicted", 4), "E24": ("predicted", 4), "E25": ("predicted", 4), "E26": ("predicted", 4), "E28": ("predicted2", 5), "E29": ("predicted", 4), "E30": ("predicted2", 5), "E31": ("predicted", 4), "E32": ("predicted", 4), "E33": ("predicted", 4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--exp", default=None, choices=sorted(PRIOR_MODES),
                    help="显式指定 E0~E7，覆盖 config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prior-dir", default=None)
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

    exp = args.exp or cfg.get("exp", "E5")
    prior_mode, in_ch = PRIOR_MODES[exp]
    if exp == "E7":
        print("!! 警告：E7 使用 GT mask 作为第 4 通道，仅为理论上界，"
              "严禁当作真实部署方法。", flush=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    name = args.name or cfg.get("name", exp + "_final")
    out_dir = Path(cfg.get("out_dir", "runs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train.log")
    log.log("=" * 70)
    log.log("train_final  exp=%s  prior_mode=%s  in_channels=%d  name=%s"
            % (exp, prior_mode, in_ch, name))
    log.log("config=%s" % args.config)
    log.log("=" * 70)
    cfg = {**cfg, "exp": exp, "prior_mode": prior_mode, "in_channels": in_ch}
    save_cfg_snapshot(cfg, out_dir)

    proc = cfg.get("proc", "datasets/EVD4UAV_processed")
    raw = cfg.get("raw", "datasets/EVD4UAV/raw")
    pd = args.prior_dir or cfg.get("prior_dir")
    manifest = cfg.get("patch_manifest", "patch_manifest.csv")
    log.log("patch manifest = %s" % manifest)

    tr = PatchDataset(proc=proc, raw=raw, split="train", task="final",
                      prior_mode=prior_mode, prior_dir=pd,
                      prior_dir2=cfg.get("prior_dir2"),
                      augment=cfg.get("augment", True), seed=seed,
                      manifest=manifest)
    va = PatchDataset(proc=proc, raw=raw, split="val", task="final",
                      prior_mode=prior_mode, prior_dir=pd,
                      prior_dir2=cfg.get("prior_dir2"),
                      augment=False, seed=seed, manifest=manifest)
    if args.limit:
        tr = Subset(tr, range(min(args.limit, len(tr))))
        va = Subset(va, range(min(max(32, args.limit // 5), len(va))))

    bs = int(cfg.get("batch_size", 8))
    nw_tr = max(1, args.num_workers, 12)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw_tr,
                    pin_memory=True, drop_last=True, prefetch_factor=6,
                    persistent_workers=nw_tr > 0)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw_tr,
                    pin_memory=True, prefetch_factor=2,
                    persistent_workers=False)

    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch}))
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        log.log("resumed from %s" % args.resume)
    FMT = setup_perf(log)
    dev = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if dev == "cuda":
        model = model.to(memory_format=FMT)
    assert model.in_channels == in_ch, \
        "模型 in_channels=%d 与实验要求的 %d 不一致（公平性检查失败）" \
        % (model.in_channels, in_ch)

    loss_fn = build_loss(cfg.get("loss", {"main": "focal", "lambda_dice": 1.0}))
    lam_pc = float(cfg.get("lambda_pc", 0.0))
    if lam_pc > 0:
        log.log("prior-consistency loss ON: lambda_pc=%.3g" % lam_pc)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.get("epochs", 30)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp")) and dev == "cuda")

    print_env_and_sanity(log, cfg, model, tr, va, bs, in_ch, args.num_workers)
    smoke_test(log, model, tl, loss_fn, dev, amp=bool(cfg.get("amp")),
               in_channels=in_ch)

    mw = MetricWriter(out_dir / "metrics.csv",
                      ["epoch", "lr", "train_loss", "val_loss", "precision",
                       "recall", "iou", "dice", "f1", "epoch_sec"])
    best = -1.0
    epochs = int(cfg.get("epochs", 30))

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for i, batch in enumerate(tl):
            x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            if dev == "cuda":
                x = x.contiguous(memory_format=FMT)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",
                                enabled=bool(cfg.get("amp")) and dev == "cuda"):
                out = model(x, batch[2].get("altitude") if (len(batch) > 2 and isinstance(batch[2], dict)) else None)
                loss = loss_fn(out, y)
                if lam_pc > 0 and x.shape[1] > 3:
                    s = torch.sigmoid(out).float()
                    p4 = x[:, 3:4].float()
                    loss = loss + lam_pc * (1.0 - 2 * (s * p4).sum() / (s.sum() + p4.sum() + 1e-7))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
            if i % 20 == 0:
                log.log("  ep%d it%4d/%4d loss=%.6f" % (ep, i, len(tl), loss.item()))
        sched.step()

        model.eval()
        vloss, acc = [], {"precision": [], "recall": [], "iou": [], "dice": [], "f1": []}
        with torch.no_grad():
            for batch in vl:
                x, y = batch[0].to(dev), batch[1].to(dev)
                if dev == "cuda":
                    x = x.contiguous(memory_format=FMT)
                out = model(x, batch[2].get("altitude") if (len(batch) > 2 and isinstance(batch[2], dict)) else None)
                vloss.append(float(loss_fn(out, y)))
                m = binary_metrics_from_logits(out, y)
                for k in acc:
                    acc[k].append(m[k])
        row = {
            "epoch": ep, "lr": sched.get_last_lr()[0],
            "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vloss)),
            **{k: float(np.mean(v)) for k, v in acc.items()},
            "epoch_sec": round(time.time() - t0, 1),
        }
        mw.add(row)
        log.log("EPOCH %d  train=%.6f  val=%.6f  iou=%.4f  dice=%.4f  rec=%.4f  prec=%.4f (%.0fs)"
                % (ep, row["train_loss"], row["val_loss"], row["iou"], row["dice"],
                   row["recall"], row["precision"], row["epoch_sec"]))
        save_ckpt(model, out_dir / "last.pth", {"epoch": ep, "cfg": cfg})
        if row["dice"] > best:
            best = row["dice"]
            save_ckpt(model, out_dir / "best.pth", {"epoch": ep, "dice": best, "cfg": cfg})
            log.log("  -> new best dice=%.5f" % best)

    mw.close()
    log.log("训练结束 exp=%s best dice=%.5f  输出=%s" % (exp, best, out_dir))
    log.close()

if __name__ == "__main__":
    main()
