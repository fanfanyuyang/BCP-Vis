# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_final_cnn, build_loss
from train.common import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rgb_baseline.yaml")
    ap.add_argument("--exp", default="E0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cudnn-benchmark", type=int, default=0)
    ap.add_argument("--channels-last", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    mem_fmt = torch.channels_last if a.channels_last else torch.contiguous_format

    cfg = load_cfg(a.config)
    in_ch = 3 if a.exp == "E0" else 4
    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch})).cuda()
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    loss_fn = build_loss(cfg.get("loss", {"main": "focal", "lambda_dice": 1.0}))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print("config=%s  exp=%s  in_ch=%d  amp=%s  bs=%d  size=%d"
          % (a.config, a.exp, in_ch, amp, a.batch_size, a.size))

    x = torch.randn(a.batch_size, in_ch, a.size, a.size, device="cuda")
    y = (torch.rand(a.batch_size, 1, a.size, a.size, device="cuda") > 0.97).float()
    if a.channels_last:
        x = x.to(memory_format=torch.channels_last)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.iters
    print("纯 GPU 训练 step: %.4f s/iter  (%.1f it/s)" % (dt, 1.0 / dt))
    print("峰值显存: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print()
    print("对照：训练日志的实测 s/iter = epoch_sec / iters_per_epoch")
    print("若实测 >> 本值，则瓶颈在数据加载（应优化 dataloader）；")
    print("若两者接近，则瓶颈在 GPU（应调 batch size / AMP / 分辨率）。")

if __name__ == "__main__":
    main()
