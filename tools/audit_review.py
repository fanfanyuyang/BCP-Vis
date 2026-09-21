# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sec(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def check_patch_dist(proc: Path, patch: int):
    sec("1. PATCH 前景分布（判定高 IoU 是否合理）")
    mf = proc / "patch_manifest.csv"
    if not mf.exists():
        print("  ! 缺少 patch_manifest.csv")
        return
    df = pd.read_csv(mf)
    col = "foreground_pixels" if "foreground_pixels" in df.columns else df.columns[-1]
    df["fg"] = df[col].astype(float)
    area = float(patch * patch)
    df["ratio"] = df["fg"] / area
    print("  总 patch: %d   列: %s" % (len(df), list(df.columns)))
    for sp, g in df.groupby("split"):
        print("  %-5s n=%-6d fg_ratio mean=%.4f  median=%.4f  p90=%.4f  空patch=%d (%.1f%%)"
              % (sp, len(g), g.ratio.mean(), g.ratio.median(), g.ratio.quantile(0.9),
                 (g.fg == 0).sum(), 100.0 * (g.fg == 0).mean()))
    print("  全局像素级前景占比: %.4f" % (df.fg.sum() / (len(df) * area)))
    print("  -> 若 patch 级 fg_ratio 远高于原始 2.34%，则高 IoU 属正常（负样本被下采样）")


def check_logs(root: Path):
    sec("2. 日志错误扫描")
    logs = sorted((root / "logs").glob("*.log"))
    pat = re.compile(r"Traceback|Error|error|CUDA out of memory|nan|NaN|inf|Killed|"
                     r"RuntimeError|FAIL", re.IGNORECASE)
    bad = re.compile(r"ZeroDivision|^(?!.*(0 error|errors=0))", re.IGNORECASE)
    total = 0
    for f in logs:
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = []
        for i, line in enumerate(txt.splitlines(), 1):
            if pat.search(line):
                if "nan" in line.lower() and "NaN" not in line and "nan" not in line.lower():
                    pass
                hits.append("%d: %s" % (i, line.strip()[:150]))
        if hits:
            total += len(hits)
            print("  [%s] %d 处" % (f.name, len(hits)))
            for h in hits[:6]:
                print("      " + h)
            if len(hits) > 6:
                print("      ... 还有 %d 处" % (len(hits) - 6))
    if total == 0:
        print("  无 Traceback / Error / OOM / Killed / FAIL")
    print("  扫描日志数: %d" % len(logs))


def check_metrics(root: Path):
    sec("3. 训练曲线（过拟合 / 发散检查）")
    for p in sorted(root.rglob("metrics.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "dice" not in df.columns or len(df) < 2:
            continue
        last = df.iloc[-1]
        best_i = df["dice"].idxmax()
        print("  %s" % p.relative_to(root))
        print("     epochs=%d  best_dice=%.4f@ep%d  last_dice=%.4f  last_iou=%.4f"
              % (len(df), df.dice.max(), int(df.loc[best_i, "epoch"]),
                 last["dice"], last["iou"]))
        if "train_loss" in df and "val_loss" in df:
            tl, vl = df.iloc[-1]["train_loss"], df.iloc[-1]["val_loss"]
            ratio = vl / tl if tl else float("nan")
            flag = "  <-- val>>train，可能过拟合" if vl > tl * 1.6 else ""
            print("     train=%.4f val=%.4f (val/train=%.2f)%s"
                  % (tl, vl, ratio, flag))
        if np.isnan(df["dice"]).any():
            print("     !! dice 含 nan")


def check_artifacts(proc: Path):
    sec("4. 产物完整性")
    items = [("binary_masks", ".png"), ("soft_priors", ".png"),
             ("oof_priors", ".png")]
    for name, ext in items:
        d = proc / name
        if not d.exists():
            print("  %-14s 不存在" % name)
            continue
        fs = list(d.glob("*" + ext))
        if fs:
            import os
            sz = sum(os.path.getsize(f) for f in fs[:200]) / max(1, len(fs[:200]))
            print("  %-14s %6d 个 %s   平均 %.0f KB   预计总计 %.2f GB"
                  % (name, len(fs), ext, sz / 1024, sz * len(fs) / 1e9))
        else:
            print("  %-14s %6d 个 %s" % (name, len(fs), ext))
    for f in ("split.csv", "patch_manifest.csv"):
        p = proc / f
        print("  %-14s %s" % (f, "OK" if p.exists() else "缺失"))


def check_split_leak(proc: Path):
    sec("5. 划分泄漏自检")
    sp = proc / "split.csv"
    mf = proc / "patch_manifest.csv"
    if not (sp.exists() and mf.exists()):
        return
    s = pd.read_csv(sp)
    m = pd.read_csv(mf)
    key = "image_id" if "image_id" in s.columns else s.columns[0]
    src = "source_image" if "source_image" in m.columns else None
    tr_img = set(s[s["split"] == "train"][key])
    va_img = set(s[s["split"] == "val"][key])
    print("  图像级 train/val 交集: %d" % len(tr_img & va_img))
    if src:
        tr_p = set(m[m["split"] == "train"][src])
        va_p = set(m[m["split"] == "val"][src])
        print("  patch 级同源图跨 split: %d" % len(tr_p & va_p))
    def seq(x):
        mm = re.match(r"^(DJI_\d{4})", str(x))
        return mm.group(1) if mm else str(x)
    if "split" in s.columns:
        sq = s.assign(g=s[key].map(seq))
        tr_s = set(sq[sq["split"] == "train"]["g"])
        va_s = set(sq[sq["split"] == "val"]["g"])
        print("  序列级跨 split: %d / %d 组（官方划分为帧级，此为已知风险）"
              % (len(tr_s & va_s), len(tr_s | va_s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--patch", type=int, default=768)
    a = ap.parse_args()
    root, proc = Path(a.root).resolve(), Path(a.proc).resolve()
    check_patch_dist(proc, a.patch)
    check_logs(root)
    check_metrics(root)
    check_artifacts(proc)
    check_split_leak(proc)
    print()

if __name__ == "__main__":
    main()
