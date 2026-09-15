# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TINY = 32 * 32
SMALL = 96 * 96


def size_group(fg):
    if fg < TINY:
        return "tiny"
    if fg < SMALL:
        return "small"
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    proc = Path(a.proc).resolve()

    df = pd.read_csv(proc / "patch_manifest.csv")
    df = df[df["split"] == a.split].copy()
    df["fg"] = df["foreground_pixels"].astype(float)
    df["size"] = df["fg"].map(size_group)
    df["alt"] = df["altitude"].fillna("unknown")
    df["snow"] = df["snow"].fillna("0").astype(str)

    print("=" * 70)
    print("1. 前景像素份额：聚合指标能否反映小目标？  (split=%s)" % a.split)
    print("=" * 70)
    tot_fg = df["fg"].sum()
    g = df.groupby("size").agg(n_patch=("fg", "size"),
                               fg_sum=("fg", "sum"),
                               fg_median=("fg", "median"))
    g["patch_%"] = (100 * g.n_patch / len(df)).round(2)
    g["fgpixel_%"] = (100 * g.fg_sum / tot_fg).round(3)
    print(g.to_string())
    print()
    print("结论：tiny 占 patch 的 %.1f%%，但只占前景像素的 %.2f%%。"
          % (100 * g.loc["tiny", "n_patch"] / len(df), g.loc["tiny", "fgpixel_%"]))
    print("      => 像素级微平均 IoU 几乎完全由 medium 决定，tiny 的成败在聚合指标里"
          "是不可见的。")

    print()
    print("=" * 70)
    print("2. 尺寸 x 高度 联合分布（patch 数）")
    print("=" * 70)
    ct = pd.crosstab(df["alt"], df["size"], margins=True)
    print(ct.to_string())

    print()
    print("  同一表按行归一（各高度层的尺寸构成 %）：")
    print((100 * pd.crosstab(df["alt"], df["size"], normalize="index")).round(1).to_string())

    print()
    print("=" * 70)
    print("3. 各高度层：前景像素 / 雪覆盖 / 平均目标尺寸")
    print("=" * 70)
    s = df.groupby("alt").agg(n=("fg", "size"),
                              fg_median=("fg", "median"),
                              fg_mean=("fg", "mean"),
                              empty_patch=("fg", lambda x: int((x == 0).sum())))
    s["snow_%"] = (100 * df.groupby("alt")["snow"]
                   .apply(lambda x: (x.astype(str).isin(["1", "1.0"])).mean())).round(1)
    s["empty_%"] = (100 * s.empty_patch / s.n).round(1)
    print(s.to_string())

    print()
    print("=" * 70)
    print("4. tiny 子集在各高度层的分布")
    print("=" * 70)
    t = df[df["size"] == "tiny"]
    if len(t):
        print(pd.crosstab(t["alt"], t["snow"]).to_string())
        print("\n每高度层 tiny 占该层 patch 比例：")
        print((100 * df.groupby("alt")["size"]
               .apply(lambda x: (x == "tiny").mean())).round(2).to_string())

if __name__ == "__main__":
    main()
