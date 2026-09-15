#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt


def make_soft_prior(binary: np.ndarray, sigma: float) -> np.ndarray:
    fg = binary.astype(bool)
    prior = np.zeros(fg.shape, dtype=np.float32)
    prior[fg] = 1.0
    if not fg.any():
        return prior
    dist = distance_transform_edt(~fg).astype(np.float32)
    outside = ~fg
    s2 = 2.0 * float(sigma) ** 2
    prior[outside] = np.exp(-(dist[outside] ** 2) / s2)
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def stats_of(p: np.ndarray, fg: np.ndarray):
    d = {
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "std": float(p.std()),
    }
    nb = (p > 0.05) & (~fg)
    d["neighborhood_ratio"] = float(nb.mean())
    d["fg_ratio"] = float(fg.mean())
    return d

_G = {}


def _init_worker(sigma, out, fmt, dry):
    _G.update({"sigma": sigma, "out": Path(out), "fmt": fmt, "dry": dry})


def _work_mp(mpath):
    import cv2 as _cv2
    mp = Path(mpath)
    b = _cv2.imread(str(mp), _cv2.IMREAD_GRAYSCALE)
    if b is None:
        return None
    fg = b > 0
    pp = make_soft_prior(fg, _G["sigma"])
    st = stats_of(pp, fg)
    st["image_id"] = mp.stem
    if not _G["dry"]:
        save_prior(_G["out"] / mp.stem, pp, _G["fmt"])
    return st


def save_prior(path_base: Path, p: np.ndarray, fmt: str):
    import cv2
    if fmt == "npy":
        np.save(str(path_base) + ".npy", p.astype(np.float32))
    else:
        cv2.imwrite(str(path_base) + ".png",
                    (np.clip(p, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def load_prior(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m.astype(np.float32) / 255.0
    d = {
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "std": float(p.std()),
    }
    nb = (p > 0.05) & (~fg)
    d["neighborhood_ratio"] = float(nb.mean())
    d["fg_ratio"] = float(fg.mean())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/soft_prior.yaml")
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--sigmas", default=None, help="逗号分隔，用于候选对比")
    ap.add_argument("--n-vis", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8, help="并行进程数")
    ap.add_argument("--format", default="png", choices=["png", "npy"],
                    help="落盘格式；png=uint8 量化（默认，省 10x 空间）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只出可视化与统计，不写全量 npy")
    args = ap.parse_args()

    cfg = {}
    cpath = Path(args.config)
    if cpath.exists():
        cfg = yaml.safe_load(cpath.read_text(encoding="utf-8")) or {}
    mask_dir = Path(args.mask_dir or cfg.get("mask_dir",
                    "datasets/EVD4UAV_processed/binary_masks")).resolve()
    out = Path(args.out or cfg.get("out_dir", "datasets/EVD4UAV_processed/soft_priors")).resolve()
    sigma = args.sigma or cfg.get("sigma", 24.0)
    n_vis = args.n_vis or cfg.get("n_vis", 20)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out.parent / "visual_examples" / "soft_prior"
    vis_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        raise SystemExit("ERROR: %s 下没有 mask，请先运行 build_binary_masks.py" % mask_dir)
    print("=" * 72)
    print("build_soft_prior_targets")
    print("mask_dir:", mask_dir, " n=", len(masks))
    print("=" * 72)

    import cv2
    random.seed(args.seed)
    picks = random.sample(masks, min(n_vis, len(masks)))

    if args.sigmas:
        sigmas = [float(s) for s in args.sigmas.split(",")]
        print("sigma 候选对比（仅 %d 张样本）: %s" % (len(picks), sigmas))
        agg = {s: [] for s in sigmas}
        for mp in picks:
            b = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if b is None:
                continue
            fg = b > 0
            tiles = [cv2.cvtColor((b > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)]
            for s in sigmas:
                p = make_soft_prior(fg, s)
                agg[s].append(stats_of(p, fg))
                heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)
                tiles.append(heat)
            canvas = np.concatenate(tiles, axis=1)
            cv2.imwrite(str(vis_dir / ("cmp_" + mp.stem + ".png")), canvas)
        print("\n%-8s %-10s %-10s %-10s %-10s %s" %
              ("sigma", "mean", "std", "nb_ratio", "min", "max"))
        for s in sigmas:
            if not agg[s]:
                continue
            m = {k: float(np.mean([a[k] for a in agg[s]])) for k in agg[s][0]}
            print("%-8.1f %-10.5f %-10.5f %-10.5f %-10.3f %.3f"
                  % (s, m["mean"], m["std"], m["neighborhood_ratio"], m["min"], m["max"]))
        print("\n对比图已写出:", vis_dir)
        return

    print("sigma =", sigma, " dry_run =", args.dry_run, " workers =", args.workers)
    todo = [m for m in masks
            if args.dry_run or not (out / (m.stem + "." + ("png" if args.format == "png" else "npy"))).exists()]
    print("待处理 %d / 总 %d（已存在的会跳过）" % (len(todo), len(masks)))

    allstats = []
    todo_s = [str(m) for m in todo]
    if args.workers > 1 and len(todo_s) > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(sigma, str(out), args.format, args.dry_run)) as pool:
            for i, st in enumerate(pool.imap_unordered(_work_mp, todo_s, chunksize=16)):
                if st is not None:
                    allstats.append(st)
                if (i + 1) % 1000 == 0:
                    print("  processed %d/%d" % (i + 1, len(todo_s)))
    else:
        _init_worker(sigma, str(out), args.format, args.dry_run)
        for i, m in enumerate(todo_s):
            st = _work_mp(m)
            if st is not None:
                allstats.append(st)
            if (i + 1) % 500 == 0:
                print("  processed %d/%d" % (i + 1, len(todo_s)))
    rows = allstats

    for mp in picks:
        b = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if b is None:
            continue
        fg = b > 0
        p = make_soft_prior(fg, sigma)
        heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)
        bm = cv2.cvtColor((fg.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(vis_dir / (mp.stem + "_softprior.png")),
                    np.concatenate([bm, heat], axis=1))

    if allstats:
        keys = ["min", "max", "mean", "std", "neighborhood_ratio", "fg_ratio"]
        agg = {k: float(np.mean([a[k] for a in allstats])) for k in keys}
        print("\n全局统计(均值 over %d 张):" % len(allstats))
        for k in keys:
            print("   %-20s %.6f" % (k, agg[k]))
        if not args.dry_run:
            (out / "stats.json").write_text(
                json.dumps({"sigma": sigma, "n": len(allstats), "aggregate": agg},
                           indent=2), encoding="utf-8")

    if not args.dry_run:
        mf = out / "soft_prior_manifest.csv"
        with open(mf, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("manifest:", mf)
    print("npy 目录:", out)
    print("可视化  :", vis_dir)
    print("=" * 72)

if __name__ == "__main__":
    main()
