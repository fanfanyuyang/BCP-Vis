#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, ".")
from models import build_priornet
from scripts.infer_prior import load_model, infer_image

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-fg", required=True, help="主先验（如 cascade）")
    ap.add_argument("--arch-fg", default="cascade")
    ap.add_argument("--cfg-fg", default=None)
    ap.add_argument("--ckpt-ind", required=True, help="独立先验（如 litehr）")
    ap.add_argument("--arch-ind", default="litehr")
    ap.add_argument("--cfg-ind", default=None)
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw")
    ap.add_argument("--split", default=None)
    ap.add_argument("--fold-images", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="U 的缩放：P*=(1-alpha*U)*P_f")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    proc = Path(args.proc).resolve()
    raw = Path(args.raw).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    m_fg = load_model(args.ckpt_fg, args.arch_fg, dev, args.cfg_fg)
    m_ind = load_model(args.ckpt_ind, args.arch_ind, dev, args.cfg_ind)
    print("loaded fg :", args.ckpt_fg)
    print("loaded ind:", args.ckpt_ind)

    if args.fold_images:
        targets = [s for s in args.fold_images.split(",") if s]
    else:
        rows = list(csv.DictReader((proc / "split.csv").open(encoding="utf-8")))
        targets = [r["image_id"] for r in rows
                   if args.split is None or r["split"] == args.split]
    print("待推理图像:", len(targets))

    alt_of = {}
    sp = proc / "split.csv"
    if sp.exists():
        for r in csv.DictReader(sp.open(encoding="utf-8")):
            alt_of[r["image_id"]] = r.get("altitude", "")

    idx = {}
    for p in raw.rglob("*"):
        if p.suffix in IMG_EXT:
            idx.setdefault(p.stem, p)

    for i, stem in enumerate(targets):
        p = idx.get(stem)
        if p is None:
            hits = [h for h in raw.rglob(stem + ".*") if h.suffix in IMG_EXT]
            if not hits:
                print("  ! 找不到图像", stem)
                continue
            p = hits[0]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("  ! 读取失败", stem)
            continue
        alt = alt_of.get(stem)
        pf = infer_image(m_fg, img, dev, tile=args.tile, overlap=args.overlap,
                         altitude=alt, mode="raw")
        pi = infer_image(m_ind, img, dev, tile=args.tile, overlap=args.overlap,
                         altitude=None, mode="raw")
        U = np.abs(pf - pi)
        P = np.clip((1.0 - args.alpha * U) * pf, 0.0, 1.0)
        cv2.imwrite(str(out / (stem + ".png")), (P * 255).astype(np.uint8))
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, len(targets)))
    print("完成，输出目录:", out)

if __name__ == "__main__":
    main()
