#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_priornet

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def load_model(ckpt, arch, device, model_config=None):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck.get("cfg") or {}
    mcfg = dict(cfg.get("model") or {})
    if model_config:
        import yaml
        ycfg = yaml.safe_load(Path(model_config).read_text(encoding="utf-8")) or {}
        mcfg = dict(mcfg)
        mcfg.update(ycfg.get("model") or {})
        print("  用 --model-config 构建模型:", mcfg)
    elif cfg.get("model"):
        print("  用 checkpoint 内配置构建模型:", mcfg)
    mcfg.setdefault("arch", arch)
    m = build_priornet(mcfg)
    sd = ck.get("model", ck)
    missing = m.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        print("  ! 缺失 key:", missing.missing_keys[:5])
    m = m.to(device).eval()
    return m

@torch.no_grad()


def infer_image(model, img_bgr, device, tile=None, overlap=192, amp=True,
                altitude=None, mode="mul", alpha=0.3):
    import cv2
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = x.to(device)
    H, W = x.shape[-2:]

    want_alt = altitude is not None and getattr(model, "accepts_altitude", False)

    dual = getattr(model, "dual", False)

    def _fwd(t):
        if dual:
            res = model.forward_dual(t, altitude if want_alt else None)
            return torch.sigmoid(res[0]), torch.sigmoid(res[1])
        return model(t, altitude) if want_alt else model(t)

    def _post(v):
        if not dual:
            return torch.sigmoid(v)
        pf, pb = v
        if mode == "raw":
            return pf
        if mode == "raw_bg":
            return pb
        if mode == "mul":
            return pf * (1.0 - pb)
        if mode == "gated":
            return torch.clamp(pf - alpha * pb, 0.0, 1.0)
        if mode == "logit":
            pf_c = pf.clamp(1e-4, 1.0 - 1e-4)
            nb_c = (1.0 - pb).clamp(1e-4, 1.0 - 1e-4)
            lf = torch.log(pf_c / (1.0 - pf_c))
            lb = torch.log(nb_c / (1.0 - nb_c))
            return torch.sigmoid(alpha * lf + lb)
        U = (pf - (1.0 - pb)).abs()
        return (1.0 - U) * pf

    if tile is None or (H <= tile and W <= tile):
        with torch.autocast(device_type="cuda",
                            enabled=amp and device == "cuda"):
            out = _post(_fwd(x))
        return out[0, 0].float().cpu().numpy()

    stride = tile - overlap
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    for y in ys:
        for xx in xs:
            sub = x[:, :, y:y + tile, xx:xx + tile]
            with torch.autocast(device_type="cuda",
                                enabled=amp and device == "cuda"):
                p = _post(_fwd(sub))
            p = p[0, 0].float().cpu().numpy()
            acc[y:y + tile, xx:xx + tile] += p
            cnt[y:y + tile, xx:xx + tile] += 1.0
    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="litehr")
    ap.add_argument("--model-config", default=None,
                    help="训练该 ckpt 用的 yaml（configs/*.yaml），用于复现 model 宽度/专家数；"
                         "ckpt 未内嵌 cfg 时必须显式传入")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw")
    ap.add_argument("--split", default=None, help="按 split.csv 过滤")
    ap.add_argument("--fold-images", default=None, help="逗号分隔的 image_id 列表")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--save-npy", action="store_true",
                    help="额外保存 float32 .npy（8.3MB/张，仅调试用；默认只存 uint8 png）")
    ap.add_argument("--prior-mode", default="mul",
                    choices=["mul", "gated", "uap", "raw", "raw_bg", "logit"],
                    help="双分支先验的组合方式：mul=相乘(默认) / gated=温和相减 / "
                         "uap=不确定性感知(第3个创新点)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="gated 模式的背景抑制系数")
    args = ap.parse_args()

    proc = Path(args.proc).resolve()
    raw = Path(args.raw).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.arch, dev, args.model_config)
    print("loaded:", args.ckpt, "arch:", args.arch, "device:", dev)

    if args.fold_images:
        targets = [s for s in args.fold_images.split(",") if s]
    else:
        rows = list(csv.DictReader((proc / "split.csv").open(encoding="utf-8")))
        targets = [r["image_id"] for r in rows
                   if args.split is None or r["split"] == args.split]
    print("待推理图像:", len(targets))

    idx = {}
    for p in raw.rglob("*"):
        if p.suffix in IMG_EXT:
            idx.setdefault(p.stem, p)

    alt_of = {}
    sp = proc / "split.csv"
    if sp.exists():
        for r in csv.DictReader(sp.open(encoding="utf-8")):
            alt_of[r["image_id"]] = r.get("altitude", "")
    if getattr(model, "accepts_altitude", False):
        miss = sum(1 for s_ in targets if not alt_of.get(s_))
        print("altitude 可用 %d/%d（缺失 %d 将退化为 0）"
              % (len(targets) - miss, len(targets), miss))

    import cv2
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
        pr = infer_image(model, img, dev, tile=args.tile, overlap=args.overlap,
                         altitude=alt_of.get(stem),
                         mode=args.prior_mode, alpha=args.alpha)
        cv2.imwrite(str(out / (stem + ".png")), (pr * 255).astype(np.uint8))
        if args.save_npy:
            np.save(str(out / (stem + ".npy")), pr)
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, len(targets)))
    print("完成，输出目录:", out)

if __name__ == "__main__":
    main()
