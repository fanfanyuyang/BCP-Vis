#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPS = {
    "E0": ("configs/rgb_baseline.yaml", "none", "runs/rgb_baseline", None),
    "E1": ("configs/bcp_vis.yaml", "zero", "runs", None),
    "E2": ("configs/bcp_vis.yaml", "random", "runs", None),
    "E3": ("configs/bcp_vis.yaml", "predicted", "runs", "v1"),
    "E4": ("configs/bcp_vis.yaml", "predicted", "runs", "v2"),
    "E5": ("configs/bcp_vis.yaml", "predicted", "runs", "v3"),
    "E6": ("configs/bcp_vis.yaml", "predicted", "runs", "v4"),
    "E7": ("configs/bcp_vis.yaml", "oracle", "runs", None),
}


def _merge():
    fm = ROOT / "reports" / "final_metrics.csv"
    if not fm.exists():
        print("没有指标文件，跳过汇总:", fm)
        return
    rows = list(csv.DictReader(fm.open(encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[r["experiment"]] = r
    write_ablation(rows, seen)


def write_ablation(rows, seen):
    abl = ROOT / "reports" / "ablation.csv"
    with open(abl, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for k in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7",
                  "E5_OOF", "E5_OOF_TAG"]:
            if k in seen:
                w.writerow(seen[k])
        for k, v in seen.items():
            if k not in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7",
                         "E5_OOF", "E5_OOF_TAG"]:
                w.writerow(v)
    print("\nablation 表 ->", abl)
    for r in seen.values():
        print("  %-9s dice=%.5f iou=%.5f iou_macro=%s rec=%.5f prec=%.5f"
              % (r["experiment"], float(r["dice"]), float(r["iou"]),
                 r.get("iou_macro", "-"), float(r["recall"]),
                 float(r["precision"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="E0,E1,E2,E3,E4,E5,E6,E7")
    ap.add_argument("--prior-root",
                    default="datasets/EVD4UAV_processed/priors",
                    help="下面按 v1/v2/v3/v4 存放各自 PriorNet 的预测")
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", action="store_true", help="先训练再评估")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true",
                    help="不做训练/评估，仅把 reports/final_metrics.csv "
                         "按实验去重合并成 ablation.csv。"
                         "（执行链已逐个调用 eval_final，这里只负责汇总；"
                         "避免本脚本按 name 拼 ckpt 路径与实际不符导致整表为空）")
    args = ap.parse_args()

    if args.merge_only:
        _merge()
        return

    rows = []
    for exp in [e.strip() for e in args.exps.split(",") if e.strip()]:
        cfg, mode, out, prior_ver = EXPS[exp]
        name = exp + "_" + ("final" if exp == "E0" else "bcp")
        ckpt = Path(args.ckpt_root) / name / "best.pth"

        if args.train and not args.eval_only:
            cmd = [sys.executable, "train/train_final.py", "--config", cfg,
                   "--exp", exp, "--name", name]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.batch_size:
                cmd += ["--batch-size", str(args.batch_size)]
            if prior_ver:
                cmd += ["--prior-dir", str(Path(args.prior_root) / prior_ver)]
            cmd += ["--device", args.device]
            print("\n$ " + " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                print("!!! %s 训练失败，跳过" % exp)
                continue

        if not ckpt.exists():
            print("! 缺少 ckpt %s，跳过评估" % ckpt)
            continue

        cmd = [sys.executable, "eval/eval_final.py", "--ckpt", str(ckpt),
               "--config", cfg, "--exp", exp, "--tag", exp,
               "--device", args.device]
        if prior_ver:
            cmd += ["--prior-dir", str(Path(args.prior_root) / prior_ver)]
        print("\n$ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print("!!! %s 评估失败" % exp)

    fm = ROOT / "reports" / "final_metrics.csv"
    if not fm.exists():
        print("没有指标文件，跳过汇总")
        return
    rows = list(csv.DictReader(fm.open(encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[r["experiment"]] = r
    write_ablation(rows, seen)

if __name__ == "__main__":
    main()
