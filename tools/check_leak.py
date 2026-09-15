# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def seq_of(stem: str) -> str:
    m = re.match(r"^(DJI_\d{4})", str(stem))
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z]+_\d+)", str(stem))
    return m.group(1) if m else str(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    a = ap.parse_args()
    proc = Path(a.proc).resolve()

    sp = pd.read_csv(proc / "split.csv")
    key = "image_id" if "image_id" in sp.columns else sp.columns[0]
    sp["seq"] = sp[key].map(seq_of)

    mf = pd.read_csv(proc / "patch_manifest.csv")
    src = "source_image" if "source_image" in mf.columns else mf.columns[1]
    mf["seq"] = mf[src].map(seq_of)

    tr_s = set(sp[sp["split"] == "train"]["seq"])
    va_s = set(sp[sp["split"] == "val"]["seq"])
    leak_seq = tr_s & va_s

    print("=" * 66)
    print("序列级泄露量化")
    print("=" * 66)
    print("序列总数            : %d" % len(tr_s | va_s))
    print("跨 split 的序列     : %d (%.1f%%)" % (len(leak_seq), 100 * len(leak_seq) / len(tr_s | va_s)))

    va = sp[sp["split"] == "val"]
    leaky_img = va["seq"].isin(leak_seq)
    print("val 图像            : %d" % len(va))
    print("val 中落在泄露序列  : %d (%.1f%%)" % (leaky_img.sum(), 100 * leaky_img.mean()))

    vam = mf[mf["split"] == "val"]
    leaky_p = vam["seq"].isin(leak_seq)
    print("val patch           : %d" % len(vam))
    print("val 中泄露 patch    : %d (%.1f%%)" % (leaky_p.sum(), 100 * leaky_p.mean()))

    print()
    print("结论：若 val 中泄露比例很高，则当前 val 指标（E0 IoU≈0.83）被高估，")
    print("      论文正式结果应改用**序列级纯净划分**重跑，或至少报告 clean-val 子集指标。")

    if leaky_img.any():
        out = proc / "val_clean_images.csv"
        clean = va.loc[~leaky_img, [key]].copy()
        clean.to_csv(out, index=False)
        print("\n已写出干净 val 图像清单: %s  (%d 张)" % (out, len(clean)))

if __name__ == "__main__":
    main()
