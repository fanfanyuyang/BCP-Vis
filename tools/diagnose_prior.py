# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_final_cnn, build_priornet
from train.common import load_cfg, set_seed

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
TINY, SMALL = 32 * 32, 96 * 96


def size_group(fg):
    if fg <= 0:
        return "empty"
    if fg < TINY:
        return "tiny"
    if fg < SMALL:
        return "small"
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="baseline (E0) 权重")
    ap.add_argument("--config", default="configs/rgb_baseline.yaml")
    ap.add_argument("--prior-ckpt", default="runs/priornet/v3_litehr/best.pth")
    ap.add_argument("--prior-arch", default="litehr")
    ap.add_argument("--prior-dir", default=None,
                    help="已生成的先验目录；不给则现场用 PriorNet 推理")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw/EVD4UAV")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-patches", type=int, default=600)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--p-thr", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    set_seed(42)
    proc, raw = Path(a.proc).resolve(), Path(a.raw).resolve()
    dev = a.device if torch.cuda.is_available() else "cpu"

    cfg = load_cfg(a.config)
    model = build_final_cnn(cfg.get("model", {"in_channels": 3})).to(dev)
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    pnet = None
    if not a.prior_dir:
        pnet = build_priornet({"arch": a.prior_arch}).to(dev)
        pck = torch.load(a.prior_ckpt, map_location="cpu")
        pnet.load_state_dict(pck.get("model", pck), strict=False)
        pnet.eval()

    rows = [r for r in csv.DictReader((proc / "patch_manifest.csv").open(encoding="utf-8"))
            if r["split"] == a.split]
    import random as _rnd
    rng = _rnd.Random(42)
    by_g = defaultdict(list)
    for r in rows:
        by_g[size_group(int(r["foreground_pixels"]))].append(r)
    sel = list(by_g["tiny"])
    sel += rng.sample(by_g["small"], min(200, len(by_g["small"])))
    sel += rng.sample(by_g["medium"], min(100, len(by_g["medium"])))
    sel = sel[:a.n_patches]
    cnt = defaultdict(int)
    for r in sel:
        cnt[size_group(int(r["foreground_pixels"]))] += 1
    print("诊断 patch: %d  构成=%s" % (len(sel), dict(cnt)))

    import cv2
    idx = {}
    for p in raw.rglob("*"):
        if p.suffix in IMG_EXT:
            idx.setdefault(p.stem, p)

    prior_cache = {}

    def get_prior(stem, shape):
        if stem in prior_cache:
            return prior_cache[stem]
        pr = None
        if a.prior_dir:
            for cand in (Path(a.prior_dir) / (stem + ".png"),
                         Path(a.prior_dir) / (stem + ".npy")):
                if cand.exists():
                    pr = (cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                          if cand.suffix == ".png" else np.load(str(cand)).astype(np.float32))
                    break
        if pr is None and pnet is not None:
            img = cv2.imread(str(idx.get(stem)), cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pr = tiled_prior(pnet, img, dev, a.tile, a.overlap)
        if pr is None:
            return None
        if pr.shape[:2] != shape:
            pr = cv2.resize(pr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        prior_cache[stem] = pr
        return pr

    agg = defaultdict(lambda: {"miss": 0.0, "hit": 0.0, "gt": 0.0, "tp": 0.0, "n": 0,
                               "p_tp": 0.0, "p_fp": 0.0, "p_fn": 0.0,
                               "area": 0.0, "pix": 0.0})

    for i, r in enumerate(sel):
        stem = r["source_image"]
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        p = idx.get(stem)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m = cv2.imread(str(proc / "binary_masks" / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = (m[y1:y2, x1:x2] > 0)
        x = torch.from_numpy(np.ascontiguousarray(
            img[y1:y2, x1:x2].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            lg = model(x.to(dev))
        pred = torch.sigmoid(lg.float().cpu())[0, 0].numpy() > a.thr

        pr = get_prior(stem, m.shape)
        if pr is None:
            continue
        hit = pr[y1:y2, x1:x2] > a.p_thr

        miss = gt & ~pred
        g = agg[size_group(int(r["foreground_pixels"]))]
        g["gt"] += gt.sum()
        g["tp"] += (gt & pred).sum()
        g["miss"] += miss.sum()
        g["hit"] += (miss & hit).sum()
        g["n"] += 1
        g["p_tp"] += (gt & hit).sum()
        g["p_fp"] += (~gt & hit).sum()
        g["p_fn"] += (gt & ~hit).sum()
        g["area"] += hit.sum()
        g["pix"] += hit.size
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(sel)), flush=True)

    print()
    print("=" * 74)
    print("先验互补性诊断  (baseline=%s, prior=%s)" % (Path(a.ckpt).parent.name,
                                                   "cached" if a.prior_dir else "PriorNet"))
    print("=" * 74)
    print("%-7s %5s %9s %9s %9s %8s %10s" %
          ("组", "n", "baseRec", "priorRec", "priorPrec", "先验覆盖", "可召回率"))
    print("-" * 74)
    for k in ["tiny", "small", "medium", "empty"]:
        if k not in agg:
            continue
        g = agg[k]
        rec = g["tp"] / (g["gt"] + 1e-7)
        prec = g["p_tp"] / (g["p_tp"] + g["p_fp"] + 1e-7)
        prec_ = g["p_tp"] / (g["p_tp"] + g["p_fn"] + 1e-7)
        cov = g["area"] / (g["pix"] + 1e-7)
        recov = g["hit"] / (g["miss"] + 1e-7)
        print("%-7s %5d %9.4f %9.4f %9.4f %8.3f%% %10.4f" %
              (k, g["n"], rec, prec_, prec, 100 * cov, recov))
    print()
    print("列含义：")
    print("  baseRec   = baseline(E0) 对 GT 前景的召回")
    print("  priorRec  = 先验(阈值 %.2f) 对 GT 前景的召回" % a.p_thr)
    print("  priorPrec = 先验自身的精度（低 = 先验很慷慨，靠大面积堆召回）")
    print("  先验覆盖  = 先验判为正的面积占全图比例（对照 priorRec 判断是否靠面积堆出来的）")
    print("  可召回率  = 先验命中 且 baseline 漏检 的像素 / baseline 漏检像素")
    print()
    print("判读：若 priorRec 显著高于其覆盖率（即精度不太低），")
    print("      说明先验确实携带位置信息；此时高可召回率才意味着")
    print("      先验能补上 baseline 的漏检，BCP 有望提升 Recall。")

@torch.no_grad()


def tiled_prior(net, img, dev, tile=1024, overlap=192):
    H, W = img.shape[:2]
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    for y in ys:
        for x in xs:
            patch = img[y:y + tile, x:x + tile]
            t = torch.from_numpy(np.ascontiguousarray(
                patch.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
            out = net(t.to(dev))
            p = torch.sigmoid(out.float().cpu())[0, 0].numpy()
            ph, pw = p.shape
            acc[y:y + ph, x:x + pw] += p
            cnt[y:y + ph, x:x + pw] += 1.0
    return acc / np.maximum(cnt, 1.0)

if __name__ == "__main__":
    main()
