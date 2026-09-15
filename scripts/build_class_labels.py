#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_binary_masks import CvatPolygonAdapter

CLASS2IDX = {"car": 0, "bus": 1, "truck": 2}
IDX2NAME = {0: "car", 1: "bus", 2: "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",
                    default="datasets/EVD4UAV/raw/EVD4UAV/mask_attribute.xml")
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--out", default=None,
                    help="image_classes.csv 输出目录（默认 proc）")
    args = ap.parse_args()
    xml = Path(args.xml).resolve()
    proc = Path(args.proc).resolve()
    out = Path(args.out).resolve() if args.out else proc
    out.mkdir(parents=True, exist_ok=True)

    if not xml.exists():
        sys.exit("ERROR: 找不到标注 xml: %s" % xml)
    adapter = CvatPolygonAdapter(xml)
    print("adapter:", adapter.name, "images:", len(adapter.images))

    rows = []
    for stem, (h, w, polys, im) in adapter.items():
        counts = {0: 0, 1: 0, 2: 0}
        for poly in polys:
            attrs = (poly.get("attrs") or {}) if isinstance(poly, dict) else {}
            t = attrs.get("type", "").strip().lower()
            if t in CLASS2IDX:
                counts[CLASS2IDX[t]] += 1
        total = sum(counts.values())
        if total == 0:
            cidx, cname = -1, "none"
        else:
            cidx = max(counts, key=counts.get)
            cname = IDX2NAME[cidx]
        rows.append({"image_id": stem, "class_idx": cidx, "class_name": cname,
                     "n_car": counts[0], "n_bus": counts[1], "n_truck": counts[2]})

    fields = ["image_id", "class_idx", "class_name", "n_car", "n_bus", "n_truck"]
    fpath = out / "image_classes.csv"
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    n_with = sum(1 for r in rows if r["class_idx"] >= 0)
    dist = Counter(r["class_name"] for r in rows)
    print("图像数:", len(rows), " 有车辆:", n_with, " 无车辆:", len(rows) - n_with)
    print("类别分布:", dict(dist))
    print("image_classes.csv:", fpath)

if __name__ == "__main__":
    main()
