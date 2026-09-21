#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np


def sliding_positions(dim: int, patch: int, stride: int):
    if dim <= patch:
        return [0]
    pos = list(range(0, dim - patch + 1, stride))
    if pos[-1] != dim - patch:
        pos.append(dim - patch)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/EVD4UAV/raw")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--patch", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--empty-keep-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-vis", type=int, default=12)
    ap.add_argument("--out-csv", default="patch_manifest.csv",
                    help="输出 manifest 文件名（相对 --proc）。"
                         "OOF 每折需写各自的 patch_manifest_foldK.csv，"
                         "避免覆盖主 manifest。")
    args = ap.parse_args()

    proc = Path(args.proc).resolve()
    root = Path(args.root).resolve()
    split_csv = proc / "split.csv"
    mask_manifest = proc / "binary_masks" / "manifest.csv"
    if not split_csv.exists():
        raise SystemExit("ERROR: 缺少 split.csv，请先运行 split_dataset.py（先划分再 patch）")
    if not mask_manifest.exists():
        raise SystemExit("ERROR: 缺少 binary_masks/manifest.csv")

    splits = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    masks = {r["image_id"]: r for r in
             csv.DictReader(mask_manifest.open(encoding="utf-8")) if r.get("status") == "OK"}

    print("=" * 72)
    print("make_patches (只生成索引，不落盘像素)")
    print("patch=%d overlap=%d stride=%d  empty_keep_ratio=%.2f"
          % (args.patch, args.overlap, args.patch - args.overlap,
             args.empty_keep_ratio))
    print("=" * 72)

    rng = random.Random(args.seed)
    rows = []
    stat = {"total": 0, "kept_fg": 0, "kept_empty": 0, "dropped_empty": 0}
    per_split = {}

    for r in splits:
        iid = r["image_id"]
        m = masks.get(iid)
        if m is None:
            continue
        H, W = int(m["height"]), int(m["width"])
        sp = r["split"]
        mp = proc / m["mask_path"]
        if not mp.exists():
            continue
        import cv2
        bm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if bm is None:
            continue
        if bm.shape != (H, W):
            bm = cv2.resize(bm, (W, H), interpolation=cv2.INTER_NEAREST)

        stride = args.patch - args.overlap
        for y1 in sliding_positions(H, args.patch, stride):
            for x1 in sliding_positions(W, args.patch, stride):
                y2, x2 = y1 + args.patch, x1 + args.patch
                sub = bm[y1:y2, x1:x2]
                fg = int((sub > 0).sum())
                stat["total"] += 1
                if fg == 0 and rng.random() > args.empty_keep_ratio:
                    stat["dropped_empty"] += 1
                    continue
                if fg == 0:
                    stat["kept_empty"] += 1
                else:
                    stat["kept_fg"] += 1
                rows.append({
                    "patch_id": "%s__%04d_%04d" % (iid, y1, x1),
                    "source_image": iid,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "altitude": r.get("altitude", ""),
                    "snow": r.get("snow", ""),
                    "foreground_pixels": fg,
                    "fg_ratio": round(fg / float(args.patch * args.patch), 8),
                    "split": sp,
                    "group": r.get("group", ""),
                })
                per_split[sp] = per_split.get(sp, 0) + 1

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = proc / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tr_img = {r["source_image"] for r in rows if r["split"] == "train"}
    va_img = {r["source_image"] for r in rows if r["split"] == "val"}
    te_img = {r["source_image"] for r in rows if r["split"] == "test"}
    leak_va = tr_img & va_img
    leak_te = tr_img & te_img

    print("候选 patch 总数  :", stat["total"])
    print("保留(含前景)     :", stat["kept_fg"])
    print("保留(纯背景采样) :", stat["kept_empty"])
    print("丢弃(纯背景)     :", stat["dropped_empty"])
    print("最终 patch 数    :", len(rows))
    print("各 split patch 数:", per_split)
    print("泄露自检 train∩val :", len(leak_va), " train∩test:", len(leak_te),
          "->", "OK" if not (leak_va or leak_te) else "!!! 存在泄露 !!!")

    if args.n_vis > 0:
        vis = proc / "visual_examples" / "patches"
        vis.mkdir(parents=True, exist_ok=True)
        import cv2
        fg_rows = [r for r in rows if r["foreground_pixels"] > 0]
        rng.shuffle(fg_rows)
        for r in fg_rows[:args.n_vis]:
            img_p = None
            for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
                for c in root.rglob(r["source_image"] + ext):
                    img_p = c
                    break
                if img_p:
                    break
            if img_p is None:
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            crop = img[y1:y2, x1:x2].copy()
            bm = cv2.imread(str(proc / masks[r["source_image"]]["mask_path"]),
                            cv2.IMREAD_GRAYSCALE)
            if bm is None:
                continue
            bm = cv2.resize(bm, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            ov = crop.copy()
            sub = bm[y1:y2, x1:x2] > 0
            ov[sub] = (0.4 * ov[sub] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(vis / (r["patch_id"] + ".jpg")),
                        np.concatenate([crop, ov], axis=1))
        print("可视化:", vis)

    print("输出:", out_csv)
    print("=" * 72)

if __name__ == "__main__":
    main()
