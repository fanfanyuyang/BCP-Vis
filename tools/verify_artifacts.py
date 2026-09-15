#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROC = Path(sys.argv[1] if len(sys.argv) > 1 else
            "datasets/EVD4UAV_processed")
RAW = Path(sys.argv[2] if len(sys.argv) > 2 else
           "datasets/EVD4UAV/raw/EVD4UAV")

issues = []

mf = PROC / "binary_masks" / "manifest.csv"
rows = list(csv.DictReader(mf.open(encoding="utf-8")))
ok = [r for r in rows if r["status"] == "OK"]
print("binary manifest: %d 条，OK=%d" % (len(rows), len(ok)))

empty = 0
bad_shape = 0
sample = ok[:200]
for r in sample:
    p = PROC / r["mask_path"]
    if not p.exists():
        issues.append("mask 缺失: %s" % r["mask_path"])
        continue
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        issues.append("mask 读取失败: %s" % p)
        continue
    if (m.shape[0], m.shape[1]) != (int(r["height"]), int(r["width"])):
        bad_shape += 1
    vals = set(np.unique(m).tolist())
    if not vals <= {0, 255}:
        issues.append("mask 值域异常 %s: %s" % (r["mask_path"], sorted(vals)[:5]))
    if (m > 0).sum() == 0:
        empty += 1
print("  抽检 %d 张：形状不符 %d，全空 %d，值域异常 %d"
      % (len(sample), bad_shape, empty, len([i for i in issues if "值域" in i])))

sp = PROC / "soft_priors"
pngs = sorted(sp.glob("*.png"))
print("soft priors: %d 个 png" % len(pngs))
bad = 0
for p in pngs[:200]:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        bad += 1
        continue
    if m.max() != 255 or m.min() < 0:
        issues.append("soft prior 值域异常: %s max=%d" % (p.name, m.max()))
print("  抽检 200 个：读取失败 %d" % bad)

pm = list(csv.DictReader((PROC / "patch_manifest.csv").open(encoding="utf-8")))
sp_rows = list(csv.DictReader((PROC / "split.csv").open(encoding="utf-8")))
split_of = {r["image_id"]: r["split"] for r in sp_rows}
mismatch = sum(1 for r in pm if split_of.get(r["source_image"]) != r["split"])
print("patch manifest: %d 条，与 split.csv 不一致 %d" % (len(pm), mismatch))
if mismatch:
    issues.append("patch 与 split 不一致 %d 条" % mismatch)

img_splits = {}
for r in pm:
    img_splits.setdefault(r["source_image"], set()).add(r["split"])
leak = [k for k, v in img_splits.items() if len(v) > 1]
print("同一原图出现在多个 split 的数量: %d" % len(leak))
if leak:
    issues.append("同图跨 split: %d 张，例 %s" % (len(leak), leak[:3]))

alt = Counter(r.get("altitude", "") for r in pm)
print("patch 高度分布:", dict(alt.most_common()))

print()
if issues:
    print("!!! 发现 %d 个问题:" % len(issues))
    for i in issues[:20]:
        print("   -", i)
    sys.exit(1)
print("INTEGRITY CHECK PASSED")
