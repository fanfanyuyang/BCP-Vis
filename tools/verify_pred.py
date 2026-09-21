# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_final_cnn
from train.common import load_cfg

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw/EVD4UAV")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-fg", type=int, default=None,
                    help="只评估前景像素 <= 该值的 patch（用于只测 tiny）")
    ap.add_argument("--images-file", default=None,
                    help="只评估该 csv 中列出的图像（列 image_id），用于干净子集复核")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    exp = cfg.get("exp", "E0")
    in_ch = 3 if exp == "E0" else 4
    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch}))
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    model.load_state_dict(sd)
    model.eval()

    dev = a.device
    try:
        model = model.to(dev)
        torch.zeros(1, in_ch, 64, 64, device=dev)
    except Exception as e:
        print("GPU 不可用(%s)，回退 CPU" % type(e).__name__)
        dev = "cpu"
        model = model.to(dev)

    proc, raw = Path(a.proc).resolve(), Path(a.raw).resolve()
    idx = {}
    for p in raw.rglob("*"):
        if p.suffix in IMG_EXT:
            idx.setdefault(p.stem, p)

    rows = [r for r in csv.DictReader((proc / "patch_manifest.csv").open(encoding="utf-8"))
            if r["split"] == a.split]
    if a.max_fg is not None:
        rows = [r for r in rows if int(r["foreground_pixels"]) <= a.max_fg]
        print("限定前景 <= %d 的 patch: %d 个" % (a.max_fg, len(rows)))
    if a.images_file:
        keep = set(pd.read_csv(a.images_file).iloc[:, 0].astype(str))
        rows = [r for r in rows if str(r["source_image"]) in keep]
        print("限定图像子集: %d 张 -> %d 个 patch" % (len(keep), len(rows)))
        step = 1
    else:
        step = max(1, len(rows) // a.n)
    rows = rows[::step][:max(a.n, len(rows))]

    import cv2
    print("%-28s %8s %8s %7s %7s %7s %7s" %
          ("patch", "gt_fg%", "pd_fg%", "IoU", "Dice", "Recall", "Prec"))
    print("-" * 76)
    tot = np.zeros(5)
    for r in rows:
        stem = r["source_image"]
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        p = idx.get(stem)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(proc / "binary_masks" / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb = img[y1:y2, x1:x2]
        gt = (m[y1:y2, x1:x2] > 0)
        x = torch.from_numpy(np.ascontiguousarray(
            rgb.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
        if in_ch == 4:
            z = torch.zeros(1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, z], dim=1)
        with torch.no_grad():
            lg = model(x.to(dev))
        pr = torch.sigmoid(lg.float().cpu())[0, 0].numpy() > a.thr

        tp = float((pr & gt).sum())
        fp = float((pr & ~gt).sum())
        fn = float((~pr & gt).sum())
        iou = tp / (tp + fp + fn + 1e-7)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        tot += np.array([tp, fp, fn, iou, dice])
        print("%-28s %7.3f%% %7.3f%% %7.4f %7.4f %7.4f %7.4f" %
              (r["patch_id"][:28], 100 * gt.mean(), 100 * pr.mean(),
               iou, dice, rec, prec))

    tp, fp, fn = tot[:3]
    print("-" * 76)
    print("汇总(逐 patch 平均) IoU=%.4f Dice=%.4f" % (tot[3] / max(1, len(rows)),
                                                    tot[4] / max(1, len(rows))))
    print("汇总(像素级微平均) IoU=%.4f Dice=%.4f  Recall=%.4f  Prec=%.4f"
          % (tp / (tp + fp + fn + 1e-7), 2 * tp / (2 * tp + fp + fn + 1e-7),
             tp / (tp + fn + 1e-7), tp / (tp + fp + 1e-7)))

if __name__ == "__main__":
    main()
