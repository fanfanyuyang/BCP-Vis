#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_py.patch_dataset import PatchDataset
from models import build_final_cnn, build_priornet
from train.common import load_cfg, set_seed, setup_perf

FMT = torch.channels_last
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
AREA_S, AREA_M = 32 * 32, 96 * 96
IOU_THRS = np.arange(0.5, 1.0, 0.05)


def load_gt_polygons(xml_path: Path, want_stems: set):
    tree = ET.parse(str(xml_path))
    gt = {}
    for im in tree.getroot().findall("image"):
        stem = Path(im.attrib.get("name", "")).stem
        if stem not in want_stems:
            continue
        w = int(im.attrib.get("width", 1920))
        h = int(im.attrib.get("height", 1080))
        polys = []
        for p in im.findall("polygon"):
            pts = p.attrib.get("points", "")
            if pts:
                polys.append(pts)
        gt[stem] = (h, w, polys)
    return gt


def rasterize_instances(h, w, polys):
    import cv2
    lab = np.zeros((h, w), dtype=np.int32)
    for i, pts in enumerate(polys, start=1):
        arr = []
        for pair in pts.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            xy = pair.split(",")
            if len(xy) < 2:
                continue
            try:
                arr.append([float(xy[0]), float(xy[1])])
            except ValueError:
                continue
        if len(arr) < 3:
            continue
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        a[:, 0] = np.clip(a[:, 0], 0, w - 1)
        a[:, 1] = np.clip(a[:, 1], 0, h - 1)
        cv2.fillPoly(lab, [a.astype(np.int32)], i)
    areas = np.bincount(lab.ravel(), minlength=len(polys) + 1)
    return lab, areas

@torch.no_grad()


def tiled_prior(net, img_rgb, dev, tile=768, overlap=192):
    H, W = img_rgb.shape[:2]
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride)) + ([max(0, H - tile)] if H > tile else [])
    xs = list(range(0, max(1, W - tile + 1), stride)) + ([max(0, W - tile)] if W > tile else [])
    ys, xs = sorted(set(ys)), sorted(set(xs))
    for y in ys:
        for x in xs:
            ph, pw = min(tile, H - y), min(tile, W - x)
            patch = img_rgb[y:y + ph, x:x + pw]
            t = torch.from_numpy(np.ascontiguousarray(
                patch.transpose(2, 0, 1))).float().unsqueeze(0).to(dev) / 255.0
            if dev == "cuda":
                t = t.contiguous(memory_format=FMT)
            p = torch.sigmoid(net(t).float())[0, 0].cpu().numpy()
            acc[y:y + ph, x:x + pw] += p
            cnt[y:y + ph, x:x + pw] += 1.0
    return acc / np.maximum(cnt, 1.0)

@torch.no_grad()


def tiled_final(net, img_rgb, prior, dev, in_ch, thr=0.5, tile=768, overlap=192):
    H, W = img_rgb.shape[:2]
    prob = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride)) + ([max(0, H - tile)] if H > tile else [])
    xs = list(range(0, max(1, W - tile + 1), stride)) + ([max(0, W - tile)] if W > tile else [])
    ys, xs = sorted(set(ys)), sorted(set(xs))
    for y in ys:
        for x in xs:
            ph, pw = min(tile, H - y), min(tile, W - x)
            rgb = img_rgb[y:y + ph, x:x + pw]
            x_in = torch.from_numpy(np.ascontiguousarray(
                rgb.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
            if in_ch == 4:
                pp = torch.from_numpy(prior[y:y + ph, x:x + pw]).float().unsqueeze(0).unsqueeze(0)
                x_in = torch.cat([x_in, pp], dim=1)
            x_in = x_in.to(dev)
            if dev == "cuda":
                x_in = x_in.contiguous(memory_format=FMT)
            p = torch.sigmoid(net(x_in).float())[0, 0].cpu().numpy()
            prob[y:y + ph, x:x + pw] += p
            cnt[y:y + ph, x:x + pw] += 1.0
    prob = prob / np.maximum(cnt, 1.0)
    return prob >= thr, prob


def pred_instances(prob, thr_mask, min_area=10):
    import cv2
    n, lab = cv2.connectedComponents(thr_mask.astype(np.uint8), connectivity=8)
    areas = np.bincount(lab.ravel(), minlength=n)
    scores = np.zeros(n, dtype=np.float32)
    if n > 1:
        flat = lab.ravel()
        s = np.bincount(flat, weights=prob.ravel(), minlength=n)
        cnt = np.maximum(areas, 1)
        scores = s / cnt
    keep = (areas >= min_area) & (np.arange(n) > 0)
    return lab, areas, scores, keep


def match_and_collect(g_lab, g_areas, p_lab, p_areas, p_scores, p_keep, t):
    G, P = len(g_areas), len(p_areas)
    if G <= 1 or P <= 1:
        tps, fps = [], list(p_scores[p_keep])
        return tps, fps
    K = P + 1
    combo = (g_lab.astype(np.int64) * K + p_lab.astype(np.int64)).ravel()
    inter = np.bincount(combo, minlength=G * K)[:G * K].reshape(G, K)[:, 1:]
    ga = g_areas[:G].astype(np.float64)[:, None]
    pa = p_areas[:P].astype(np.float64)[None, :]
    iou = inter[1:, 1:] / np.maximum(ga[1:] + pa[:, 1:] - inter[1:, 1:], 1e-9)
    order = np.argsort(-p_scores[1:])
    used = np.zeros(G, dtype=bool)
    tps, fps = [], []
    for j in order:
        if not p_keep[j + 1]:
            continue
        col = iou[:, j]
        best, bi = -1.0, -1
        for i in np.argsort(-col):
            if used[i]:
                continue
            best, bi = col[i], i
            break
        if best >= t and bi >= 0:
            used[bi] = True
            tps.append(float(p_scores[j + 1]))
        else:
            fps.append(float(p_scores[j + 1]))
    return tps, fps


def ap_from_records(records, n_gt):
    if n_gt == 0:
        return float("nan")
    rec = sorted(records, key=lambda r: -r[0])
    tp = np.array([r[1] for r in rec], dtype=np.float64)
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    rcl = ctp / n_gt
    rpoints = np.linspace(0, 1, 101)
    pinterp = np.zeros(101)
    for i, rp in enumerate(rpoints):
        m = rcl >= rp
        pinterp[i] = prec[m].max() if m.any() else 0.0
    return float(pinterp.mean())


def evaluate_set(all_recs, n_gt, t):
    sel = [(s, ok) for (s, ok, _a, mt) in all_recs if mt == t]
    return ap_from_records(sel, n_gt)


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="E0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw/EVD4UAV")
    ap.add_argument("--xml", default="datasets/EVD4UAV/raw/EVD4UAV/mask_attribute.xml")
    ap.add_argument("--prior-ckpt", default="runs/priornet/v3_litehr/best.pth")
    ap.add_argument("--prior-dir", default=None, help="已缓存先验目录（优先）")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    set_seed(42)
    setup_perf()
    dev = a.device if torch.cuda.is_available() else "cpu"
    tag = a.tag or a.exp

    proc, raw = Path(a.proc).resolve(), Path(a.raw).resolve()
    cfg = load_cfg(a.config)
    in_ch = 3 if a.exp == "E0" else 4
    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch})).to(dev)
    if dev == "cuda":
        model = model.to(memory_format=FMT)
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    pnet = None
    if in_ch == 4 and not a.prior_dir:
        pnet = build_priornet({"arch": "litehr"}).to(dev)
        pck = torch.load(a.prior_ckpt, map_location="cpu")
        pnet.load_state_dict(pck.get("model", pck), strict=False)
        pnet.eval()

    import csv as _csv
    rows = [r for r in _csv.DictReader((proc / "split.csv").open(encoding="utf-8"))
            if r["split"] == a.split]
    stems = [r["image_id"] for r in rows]
    if a.limit:
        stems = stems[:a.limit]
    want = set(stems)

    gt = load_gt_polygons(Path(a.xml).resolve(), want)
    print("GT 图像: %d（含实例 %d）" % (len(gt), sum(1 for v in gt.values() if v[2])))

    idx = {}
    for p in raw.rglob("*"):
        if p.suffix in IMG_EXT:
            idx.setdefault(p.stem, p)

    groups = {"all": None, "S": None, "M": None, "L": None}
    recs = {t: {g: [] for g in groups} for t in IOU_THRS}
    ngt = {g: 0 for g in groups}
    t_fps = []
    npred = 0

    for n, stem in enumerate(stems):
        p = idx.get(stem)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        prior = None
        if in_ch == 4:
            if a.prior_dir:
                f = Path(a.prior_dir) / (stem + ".png")
                if f.exists():
                    prior = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if prior is None and pnet is not None:
                prior = tiled_prior(pnet, img, dev)
            if prior is None:
                prior = np.zeros((H, W), dtype=np.float32)

        t0 = time.time()
        mask, prob = tiled_final(model, img, prior, dev, in_ch, thr=a.thr)
        if n < 30:
            t_fps.append(time.time() - t0)

        h, w, polys = gt.get(stem, (H, W, []))
        g_lab, g_areas = rasterize_instances(h, w, polys) if polys else (np.zeros((h, w), np.int32), np.zeros(1))
        if (h, w) != (H, W):
            g_lab = cv2.resize(g_lab, (W, H), interpolation=cv2.INTER_NEAREST)

        p_lab, p_areas, p_scores, p_keep = pred_instances(prob, mask)
        npred += max(0, len(p_areas) - 1)

        gidx = {g: [] for g in ("S", "M", "L")}
        for i in range(1, len(g_areas)):
            ar = g_areas[i]
            gidx["S" if ar < AREA_S else "M" if ar < AREA_M else "L"].append(i)

        ngt["all"] += max(0, len(g_areas) - 1)
        for gname in ("S", "M", "L"):
            ngt[gname] += len(gidx[gname])

        for t in IOU_THRS:
            for gname, gids in (("all", None), ("S", gidx["S"]), ("M", gidx["M"]), ("L", gidx["L"])):
                if gname == "all":
                    tps, fps = match_and_collect(g_lab, g_areas, p_lab, p_areas,
                                                  p_scores, p_keep, t)
                else:
                    if not gids:
                        tps = []
                        fps = [float(s) for k in range(1, len(p_scores)) if p_keep[k]
                               for s in [p_scores[k]]]
                    else:
                        gl = np.where(np.isin(g_lab, gids), g_lab, 0)
                        ga = np.zeros(len(g_areas), dtype=g_areas.dtype)
                        ga[0] = 0
                        remap = {0: 0}
                        for newi, oldi in enumerate(gids, start=1):
                            remap[oldi] = newi
                            ga[newi] = g_areas[oldi]
                        lut = np.zeros(int(g_lab.max()) + 2, dtype=np.int32)
                        for oldi, newi in remap.items():
                            if oldi < len(lut):
                                lut[oldi] = newi
                        gl2 = lut[gl]
                        tps, fps = match_and_collect(gl2, ga, p_lab, p_areas,
                                                      p_scores, p_keep, t)
                for s in tps:
                    recs[t][gname].append((s, 1.0))
                for s in fps:
                    recs[t][gname].append((s, 0.0))

        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(stems)), flush=True)

    def safe(x):
        return float(x) if x == x else float("nan")

    ap_all = [ap_from_records(recs[t]["all"], max(ngt["all"], 1)) for t in IOU_THRS]
    ap_s = [ap_from_records(recs[t]["S"], max(ngt["S"], 1)) for t in IOU_THRS]
    ap_m = [ap_from_records(recs[t]["M"], max(ngt["M"], 1)) for t in IOU_THRS]
    ap_l = [ap_from_records(recs[t]["L"], max(ngt["L"], 1)) for t in IOU_THRS]

    row = {
        "experiment": tag, "split": a.split,
        "AP": safe(np.nanmean(ap_all)),
        "AP50": safe(ap_all[0]), "AP75": safe(ap_all[5]),
        "APS": safe(np.nanmean(ap_s)),
        "APM": safe(np.nanmean(ap_m)),
        "APL": safe(np.nanmean(ap_l)),
        "n_gt": ngt["all"],
        "n_gtS": ngt["S"], "n_gtM": ngt["M"], "n_gtL": ngt["L"],
        "fps_fullimg": round(len(t_fps) / max(sum(t_fps), 1e-9), 2) if t_fps else 0,
        "n_pred": npred,
        "n_images": len(stems),
    }
    print()
    print("=" * 64)
    print("实例分割指标  %s" % tag)
    print("=" * 64)
    for k in ("AP", "AP50", "AP75", "APS", "APM", "APL", "fps_fullimg", "n_gt"):
        print("  %-12s %s" % (k, row[k]))
    print("  GT 实例分布  S=%d M=%d L=%d" % (row["n_gtS"], row["n_gtM"], row["n_gtL"]))

    rep = Path("reports")
    rep.mkdir(exist_ok=True)
    out = rep / "instance_ap.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("写出 ->", out)

if __name__ == "__main__":
    main()
