#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets_py.patch_dataset import PatchDataset
from models import build_final_cnn
from train.common import load_cfg, set_seed


def to_bgr(a):
    if a.ndim == 2:
        return cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="E5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="reports/visual_samples")
    ap.add_argument("--proc", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--prior-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.proc:
        cfg["proc"] = args.proc
    if args.raw:
        cfg["raw"] = args.raw
    set_seed(args.seed)

    modes = {"E0": ("none", 3), "E1": ("zero", 4), "E2": ("random", 4),
             "E3": ("predicted", 4), "E4": ("predicted", 4),
             "E5": ("predicted", 4), "E6": ("predicted", 4),
             "E7": ("oracle", 4)}
    mode, in_ch = modes[args.exp]
    dev = args.device if torch.cuda.is_available() else "cpu"

    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch})).to(dev)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    ds = PatchDataset(proc=cfg["proc"], raw=cfg["raw"], split=args.split,
                      task="final", prior_mode=mode,
                      prior_dir=args.prior_dir or cfg.get("prior_dir"),
                      augment=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = min(args.n, len(ds))
    idxs = list(range(n))
    with torch.no_grad():
        for i in idxs:
            item = ds[i]
            x, y = item[0], item[1]
            meta = item[2] if len(item) > 2 else {}
            pid = meta.get("patch_id", "sample%03d" % i)
            xi = x.unsqueeze(0).to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                logit = model(xi)
            pred = (torch.sigmoid(logit)[0, 0].float().cpu().numpy() > args.thr)
            gt = y[0].numpy().astype(np.float32)

            rgb = (x[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            tiles = [rgb, to_bgr(gt)]
            if in_ch == 4:
                prior = x[3].numpy()
                tiles.append(cv2.applyColorMap((np.clip(prior, 0, 1) * 255
                                                ).astype(np.uint8),
                                               cv2.COLORMAP_JET))
            tiles.append(to_bgr(pred.astype(np.float32)))
            ov = rgb.copy()
            ov[pred] = (0.4 * ov[pred] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            tiles.append(ov)
            canvas = np.concatenate(tiles, axis=1)
            cv2.imwrite(str(out / ("%s_%s.jpg" % (args.exp, pid))), canvas)
    print("写出 %d 张 -> %s" % (n, out))

if __name__ == "__main__":
    main()
