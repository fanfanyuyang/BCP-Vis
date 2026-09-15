# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    proc = Path(a.proc).resolve()
    k = a.fold

    fm = proc / ("patch_manifest_fold%d.csv" % k)
    fs = proc / ("split_fold%d.csv" % k)
    mm = proc / "patch_manifest.csv"
    sp = proc / "split.csv"

    print("=" * 66)
    print("Fold %d OOF 校验" % k)
    print("=" * 66)

    if not fm.exists():
        print("  !! 折 manifest 不存在:", fm)
        return
    if not fs.exists():
        print("  !! 折 split 不存在:", fs)
        return

    f = pd.read_csv(fm)
    s = pd.read_csv(fs)
    m = pd.read_csv(mm)
    o = pd.read_csv(sp)

    print("折 manifest      : %d 行, %d 张图, split=%s"
          % (len(f), f["source_image"].nunique(),
             dict(f["split"].value_counts())))
    print("主 manifest      : %d 行, %d 张图, split=%s"
          % (len(m), m["source_image"].nunique(),
             dict(m["split"].value_counts())))

    tr_img = set(f[f["split"] == "train"]["source_image"])
    va_img = set(f[f["split"] == "val"]["source_image"])
    s_tr = set(s[s["split"] == "train"]["image_id"])
    s_va = set(s[s["split"] == "val"]["image_id"])
    all_tr = set(o[o["split"] == "train"]["image_id"])

    print()
    ok = True

    inter = tr_img & va_img
    print("[%s] 折内 train/val 图像交集 = %d" % ("OK" if not inter else "FAIL", len(inter)))
    ok &= not inter

    match = (tr_img == s_tr)
    print("[%s] manifest train 图像 == split_fold train 图像" % ("OK" if match else "FAIL"))
    ok &= match

    shrunk = len(tr_img) < len(all_tr)
    print("[%s] 折 train 图像 %d < 全量 train %d（确实做了 held-out）"
          % ("OK" if shrunk else "FAIL", len(tr_img), len(all_tr)))
    ok &= shrunk

    no_leak = not (tr_img & s_va)
    print("[%s] 折 train 图像与 held-out 集合交集 = %d"
          % ("OK" if no_leak else "FAIL", len(tr_img & s_va)))
    ok &= no_leak

    m_tr = set(m[m["split"] == "train"]["source_image"])
    m_va = set(m[m["split"] == "val"]["source_image"])
    clean_split = not (m_tr & m_va)
    print("[%s] 主 manifest train/val 交集 = %d（应为 0）"
          % ("OK" if clean_split else "FAIL", len(m_tr & m_va)))
    ok &= clean_split

    o_tr = set(o[o["split"] == "train"]["image_id"])
    o_va = set(o[o["split"] == "val"]["image_id"])
    clean_o = not (o_tr & o_va)
    print("[%s] 主 split.csv train/val 交集 = %d（应为 0）"
          % ("OK" if clean_o else "FAIL", len(o_tr & o_va)))
    ok &= clean_o

    print()
    print("结论:", "OOF 折划分正确" if ok else "存在问题，需排查")
    print("注意：本脚本只校验「划分」，PriorNet 是否真用了折内 manifest")
    print("      需看 oof_folds/oof_foldK/train.log 里的 'patch manifest = ...'")

if __name__ == "__main__":
    main()
