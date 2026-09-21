# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import cv2

PROC = "<PROJECT_ROOT>/datasets/EVD4UAV_processed"
GT = os.path.join(PROC, "binary_masks")

rows = [r for r in csv.DictReader(open(os.path.join(PROC, "patch_manifest.csv"), encoding="utf-8"))
        if r.get("split") == "val"]
imgs, seen = [], set()
for r in rows:
    if r["source_image"] not in seen:
        seen.add(r["source_image"]); imgs.append(r["source_image"])
    if len(imgs) >= 6:
        break
gts = {}
for img in imgs:
    g = cv2.imread(os.path.join(GT, img + ".png"), cv2.IMREAD_GRAYSCALE)
    if g is not None:
        gts[img] = g - g + 0
gts = {img: cv2.imread(os.path.join(GT, img + ".png"), cv2.IMREAD_GRAYSCALE) for img in imgs}
gts = {k: v for k, v in gts.items() if v is not None}

dirs = sorted(d for d in os.listdir(PROC)
              if os.path.isdir(os.path.join(PROC, d)) and
              (d.startswith("priors") or d.startswith("oof_priors") or d.startswith("soft_priors")))

print("%-24s %-10s %-10s %-10s %s" % ("prior_dir", "目标区", "背景区", "对比度", "样本"))
for d in dirs:
    fs = os.listdir(os.path.join(PROC, d))
    if not fs:
        continue
    fg_l, bg_l = [], []
    used = 0
    for img in gts:
        p = os.path.join(PROC, d, img + ".png")
        if not os.path.exists(p):
            continue
        pr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        gt = gts[img]
        if pr is None:
            continue
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        fgm = gt > 127
        bgm = ~fgm
        if fgm.any():
            fg_l.append(float(pr[fgm].mean()))
            bg_l.append(float(pr[bgm].mean()))
            used += 1
    if used:
        f, b = np.mean(fg_l), np.mean(bg_l)
        flag = "  <== 疑似被压扁" if f < 150 else ""
        print("%-24s %-10.1f %-10.1f %-10.1f n=%d%s" % (d, f, b, f - b, used, flag))
    else:
        print("%-24s %s" % (d, "（无对应 GT 文件名，跳过）"))
