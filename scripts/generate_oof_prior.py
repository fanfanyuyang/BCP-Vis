#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def kfold_groups(groups, k, seed=42):
    groups = sorted(groups)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(groups))
    folds = {i: [] for i in range(k)}
    for j, gi in enumerate(idx):
        folds[j % k].append(groups[gi])
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw")
    ap.add_argument("--base-config", default="configs/prior_v3_litehr.yaml")
    ap.add_argument("--prior-cfg-name", default="litehr")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="必须与主实验训练 PriorNet 时的 batch size 一致，"
                         "否则 OOF 先验与主先验不可比")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="只跑第 n 个 fold")
    ap.add_argument("--prior-mode", default="mul",
                    choices=["mul", "gated", "uap"],
                    help="OOF 先验组合方式（mul/uap），默认 mul")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="uap/gated 融合系数")
    ap.add_argument("--out-dir", default="datasets/EVD4UAV_processed/oof_priors",
                    help="OOF 先验输出目录（默认 oof_priors，勿覆盖已有）")
    ap.add_argument("--model-config", default=None,
                    help="非默认规格 PriorNet(cascade 等)必须显式传 config")
    args = ap.parse_args()

    proc = Path(args.proc).resolve()
    raw = Path(args.raw).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    splits = list(csv.DictReader((proc / "split.csv").open(encoding="utf-8")))
    train_rows = [r for r in splits if r["split"] == "train"]
    groups = sorted({r["group"] for r in train_rows})
    g2imgs = {}
    for r in train_rows:
        g2imgs.setdefault(r["group"], []).append(r["image_id"])
    print("=" * 72)
    print("5-Fold OOF Prior")
    print("train 图像 %d，分组 %d（按序列去重，防止同序列跨 fold）"
          % (len(train_rows), len(groups)))
    print("=" * 72)

    folds = kfold_groups(groups, args.folds, args.seed)
    for k, gs in folds.items():
        print("  fold %d: %d groups / %d images"
              % (k, len(gs), sum(len(g2imgs[g]) for g in gs)))

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8")) or {}
    epochs = args.epochs or base_cfg.get("epochs", 30)

    fold_dir = proc / "oof_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    todo = [int(args.only)] if args.only is not None else list(range(args.folds))
    manifest = []

    for k in todo:
        held = set(folds[k])
        tr_groups = [g for g in groups if g not in held]
        fsplit = proc / ("split_fold%d.csv" % k)
        with open(fsplit, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "group", "split", "altitude", "snow"])
            for r in splits:
                if r["split"] != "train":
                    continue
                w.writerow([r["image_id"], r["group"],
                            "train" if r["group"] in tr_groups else "val",
                            r.get("altitude", ""), r.get("snow", "")])
        print("\n--- fold %d : 训练 PriorNet（%d train / %d held-out）---"
              % (k, len(tr_groups), len(held)))
        sub_name = "patch_manifest_fold%d.csv" % k
        sub = proc / sub_name
        cmd = [sys.executable, "scripts/make_patches.py",
               "--proc", str(proc), "--root", str(raw),
               "--out-csv", sub_name]
        import shutil
        bak = proc / "split.csv.bak"
        mbak = proc / "patch_manifest.csv.bak"
        shutil.copy(proc / "split.csv", bak)
        if (proc / "patch_manifest.csv").exists():
            shutil.copy(proc / "patch_manifest.csv", mbak)
        shutil.copy(fsplit, proc / "split.csv")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:]); print(r.stderr[-1500:])
                raise SystemExit("make_patches 失败")
            if not sub.exists():
                raise SystemExit("make_patches 未产出 %s" % sub)
        finally:
            shutil.copy(bak, proc / "split.csv")
            if mbak.exists():
                shutil.copy(mbak, proc / "patch_manifest.csv")

        cfg = dict(base_cfg)
        cfg.update({
            "name": "oof_fold%d" % k,
            "epochs": epochs,
            "out_dir": str(fold_dir),
            "patch_manifest": str(sub.name),
        })
        cpath = Path("configs") / ("oof_fold%d.yaml" % k)
        cpath.parent.mkdir(exist_ok=True)
        cpath.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        cmd = [sys.executable, "train/train_prior.py", "--config", str(cpath),
               "--device", args.device,
               "--batch-size", str(args.batch_size),
               "--num-workers", str(args.num_workers)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit("fold %d 训练失败" % k)

        cmd = [sys.executable, "scripts/infer_prior.py",
               "--ckpt", str(fold_dir / ("oof_fold%d" % k) / "best.pth"),
               "--arch", args.prior_cfg_name,
               "--fold-images", ",".join(sorted(
                   [i for g in held for i in g2imgs[g]])),
               "--out-dir", str(out), "--proc", str(proc), "--raw", str(raw),
               "--device", args.device, "--prior-mode", args.prior_mode,
               "--alpha", str(args.alpha)]
        if args.model_config:
            cmd += ["--model-config", args.model_config]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit("fold %d 推理失败（需要 scripts/infer_prior.py）" % k)

        for g in sorted(held):
            for iid in g2imgs[g]:
                manifest.append({"image_id": iid, "group": g,
                                 "generated_by_fold": k,
                                 "seen_by_generator": 0})
        print("fold %d 完成" % k)

    mf = out / "oof_manifest.csv"
    if manifest:
        with open(mf, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)
        print("\noof_manifest:", mf, "(%d 条)" % len(manifest))
    print("说明：seen_by_generator=0 表示该样本的 prior "
          "来自从未在它上面训练过的模型。")
    print("=" * 72)

if __name__ == "__main__":
    main()
