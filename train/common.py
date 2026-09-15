# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def setup_perf(log=None):
    torch.backends.cudnn.benchmark = True
    fmt = torch.channels_last
    if log is not None:
        log.log("perf: cudnn.benchmark=True, memory_format=channels_last")
    return fmt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path):
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def save_cfg_snapshot(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


class Logger:

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class MetricWriter:

    def __init__(self, path, fields):
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def add(self, row):
        self.w.writerow({k: row.get(k, "") for k in self.fields})
        self.f.flush()

    def close(self):
        self.f.close()


def print_env_and_sanity(log, cfg, model, train_ds, val_ds, batch_size,
                         in_channels, num_workers):
    log.log("-" * 60)
    log.log("训练前自动检查 (第 31 节)")
    log.log("torch version   : %s" % torch.__version__)
    log.log("cuda available  : %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        log.log("GPU name        : %s" % torch.cuda.get_device_name(0))
        log.log("GPU count       : %d" % torch.cuda.device_count())
        log.log("CUDA memory     : %.1f / %.1f GB"
                % (torch.cuda.memory_allocated() / 1e9,
                   torch.cuda.get_device_properties(0).total_memory / 1e9))
    log.log("dataset length  : train=%d val=%d" % (len(train_ds), len(val_ds)))
    log.log("train batches   : %d" % (len(train_ds) // max(1, batch_size)))
    log.log("val batches     : %d" % (len(val_ds) // max(1, batch_size)))
    log.log("input channels  : %d" % in_channels)
    log.log("seed            : %s" % cfg.get("seed", 42))
    log.log("num_workers     : %d" % num_workers)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.log("trainable params: %d (%.2f M)" % (n_param, n_param / 1e6))
    log.log("config          : %s" % json.dumps(cfg, ensure_ascii=False)[:800])
    log.log("-" * 60)


def smoke_test(log, model, loader, loss_fn, device, amp=False, in_channels=3):
    log.log("Smoke test: 1 batch forward/backward ...")
    model.train()
    batch = next(iter(loader))
    x = batch[0].to(device)
    y = batch[1].to(device)
    alt = batch[2] if (len(batch) > 2 and getattr(model, "accepts_altitude", False)) else None
    assert x.shape[1] == in_channels, \
        "input channels 不匹配: 期望 %d 得到 %d" % (in_channels, x.shape[1])
    with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
        out = model(x, alt) if alt is not None else model(x)
        loss = loss_fn(out, y)
    assert out.shape[0] == x.shape[0] and out.shape[1] == 1, \
        "输出形状异常: %s (期望 [B,1,H,W])" % (tuple(out.shape),)
    assert out.shape[-2:] == x.shape[-2:], \
        "输出分辨率未恢复到输入尺寸: %s vs %s" % (tuple(out.shape[-2:]), tuple(x.shape[-2:]))
    loss.backward()
    grads = [p.grad.abs().mean().item() for p in model.parameters()
             if p.grad is not None]
    log.log("  input  : %s" % (tuple(x.shape),))
    log.log("  output : %s" % (tuple(out.shape),))
    log.log("  target : %s" % (tuple(y.shape),))
    lv = float(loss.detach())
    log.log("  loss   : %.6f  (finite=%s)" % (lv, bool(torch.isfinite(loss.detach()))))
    log.log("  grad   : n=%d mean=%.3e" % (len(grads), float(np.mean(grads)) if grads else 0.0))
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke test 失败：loss 非有限值，请检查 mask/prior 值域与 LR")
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    log.log("Smoke test 通过")


def save_ckpt(model, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra or {}}, str(path))


def binary_metrics_from_logits(logits, target, thr=0.5, eps=1e-7):
    p = (torch.sigmoid(logits) > thr).float()
    t = (target > 0.5).float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    f1 = dice
    return {"precision": float(prec), "recall": float(rec),
            "iou": float(iou), "dice": float(dice), "f1": float(f1),
            "tp": float(tp), "fp": float(fp), "fn": float(fn)}
