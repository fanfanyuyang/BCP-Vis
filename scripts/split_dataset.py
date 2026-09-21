#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def group_key(stem: str, mode: str) -> str:
    if mode == "sequence":
        m = re.match(r"^(DJI_\d{4})", stem)
        if m:
            return m.group(1)
        m = re.match(r"^([A-Za-z]+_\d+)", stem)
        if m:
            return m.group(1)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="datasets/EVD4UAV_processed/binary_masks/manifest.csv")
    ap.add_argument("--out", default="datasets/EVD4UAV_processed")
    ap.add_argument("--group-by", default="sequence", choices=["sequence", "stem"])
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official-train", default=None,
                    help="官方 train.txt（每行一个 image stem）")
    ap.add_argument("--official-val", default=None, help="官方 val.txt")
    ap.add_argument("--enforce-group", action="store_true",
                    help="强制同一序列只属于一个 split（会缩小 val，慎用）")
    args = ap.parse_args()

    mf = Path(args.manifest).resolve()
    if not mf.exists():
        raise SystemExit("ERROR: 找不到 manifest: %s（先跑 build_binary_masks.py）" % mf)

    rows = list(csv.DictReader(mf.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("status") == "OK"]
    print("=" * 72)
    print("split_dataset")
    print("可用样本:", len(rows), " 分组方式:", args.group_by)
    print("=" * 72)

    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r["image_id"], args.group_by)].append(r)
    print("分组数:", len(groups))

    gmeta = {}
    for g, rs in groups.items():
        alts = [r.get("altitude", "") for r in rs if r.get("altitude")]
        alt = max(set(alts), key=alts.count) if alts else "unknown"
        snow = "snow" if any(r.get("snow") == "1" for r in rs) else "normal"
        gmeta[g] = (alt, snow)

    strata = defaultdict(list)
    for g, (alt, snow) in gmeta.items():
        strata["%s|%s" % (alt, snow)].append(g)

    assign = {}
    img_assign = {}
    if args.official_train and args.official_val:
        def _load(p):
            return {Path(l.strip()).stem for l in
                    Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}

        tr_s, va_s = _load(args.official_train), _load(args.official_val)
        print("官方 train=%d  val=%d  重叠=%d" % (len(tr_s), len(va_s),
                                                 len(tr_s & va_s)))
        unassigned = 0
        for r in rows:
            iid = r["image_id"]
            if iid in va_s:
                img_assign[iid] = "val"
            elif iid in tr_s:
                img_assign[iid] = "train"
            else:
                img_assign[iid] = "train"
                unassigned += 1
        print("官方未覆盖、归入 train 的图像: %d" % unassigned)

        gmap = {g: set() for g in groups}
        for r in rows:
            gmap[group_key(r["image_id"], args.group_by)].add(img_assign[r["image_id"]])
        shared = [g for g, s in gmap.items() if len(s) > 1]
        print("!! 同序列跨 split 的组数: %d / %d  (官方 split 为帧级，属已知风险)"
              % (len(shared), len(groups)))

        if args.enforce_group:
            print("   --enforce-group 已开启：强行让每个序列只属于一个 split")
            for g, s in gmap.items():
                cnt = {"train": 0, "val": 0}
                for r in groups[g]:
                    cnt[img_assign[r["image_id"]]] += 1
                pick = "val" if cnt["val"] >= cnt["train"] else "train"
                assign[g] = pick
            for r in rows:
                img_assign[r["image_id"]] = assign[
                    group_key(r["image_id"], args.group_by)]
        else:
            assign = {g: ("val" if "val" in s else "train") for g, s in gmap.items()}
    else:
        rng = random.Random(args.seed)
        print("\n%-16s %-8s %-8s %-8s" % ("stratum", "groups", "train", "val"))
        for s, gs in sorted(strata.items()):
            gs = sorted(gs)
            rng.shuffle(gs)
            n = len(gs)
            n_tr = max(1, int(round(n * args.train)))
            n_va = max(1, int(round(n * args.val))) if n >= 3 else (1 if n >= 2 else 0)
            if n_tr + n_va > n:
                n_va = max(0, n - n_tr)
                if n_va == 0 and n >= 2:
                    n_tr = n - 1
                    n_va = 1
            for i, g in enumerate(gs):
                assign[g] = "train" if i < n_tr else (
                    "val" if i < n_tr + n_va else "test")
            print("%-16s %-8d %-8d %-8d"
                  % (s, len(gs), sum(1 for g in gs if assign[g] == "train"),
                     sum(1 for g in gs if assign[g] == "val")))
        for r in rows:
            img_assign[r["image_id"]] = assign[
                group_key(r["image_id"], args.group_by)]

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "split.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "split", "altitude", "snow"])
        for r in rows:
            g = group_key(r["image_id"], args.group_by)
            w.writerow([r["image_id"], g, img_assign.get(r["image_id"], "train"),
                        r.get("altitude", ""), r.get("snow", "")])

    cnt = defaultdict(int)
    for r in rows:
        cnt[img_assign.get(r["image_id"], "train")] += 1
    summary = {
        "group_by": args.group_by,
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val},
        "n_groups": len(groups),
        "n_images": len(rows),
        "images_per_split": dict(cnt),
        "groups_per_split": {s: sum(1 for g, v in assign.items() if v == s)
                             for s in ("train", "val", "test")},
        "official_split_used": bool(args.official_train),
    }
    (out / "split.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    print("\n图像数/划分:", dict(cnt))
    print("组数/划分  :", summary["groups_per_split"])
    print("输出:", cpath)
    print("     ", out / "split.json")
    print("=" * 72)

if __name__ == "__main__":
    main()
