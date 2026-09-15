# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def pick(rows, tag, group=None, key="group"):
    for r in rows:
        if r["experiment"] != tag:
            continue
        if group is None or r.get(key) == group:
            return r
    return None


def f(r, col):
    return float(r[col]) if r and r.get(col) not in (None, "") else None


def show(title, rows, groups, base, others, cols=("iou", "recall", "precision")):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    head = "%-10s" % "子集"
    for c in cols:
        head += "%-8s" % c.upper()
    print("%-10s %-8s %-8s %-8s" % ("子集", *[c.upper() for c in cols]))
    print("-" * 78)
    for g in groups:
        b = pick(rows, base, g)
        if not b:
            continue
        line = "%-10s" % g
        for c in cols:
            line += "%-8.4f" % (f(b, c) or 0)
        print(line)
        for o in others:
            r = pick(rows, o, g)
            if not r:
                continue
            line = "%-10s" % ("  Δ" + o)
            for c in cols:
                bv, ov = f(b, c), f(r, c)
                if bv is None or ov is None:
                    line += "%-8s" % "-"
                else:
                    d = ov - bv
                    line += "%+8.4f" % d
            print(line)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--baseline", default="E0")
    ap.add_argument("--others", default="E1,E2,E5,E5_OOF,E6,E7")
    a = ap.parse_args()
    rep = Path(a.reports)

    others = [x.strip() for x in a.others.split(",") if x.strip()]
    size = load(rep / "by_size.csv")
    snow = load(rep / "by_snow.csv")
    alt = load(rep / "by_altitude.csv")
    fm = load(rep / "final_metrics.csv")

    if fm:
        print("=" * 78)
        print("总体指标")
        print("=" * 78)
        print("%-12s %-9s %-9s %-9s %-9s %-9s" %
              ("实验", "IoU", "IoU_macro", "Dice", "Recall", "Prec"))
        print("-" * 78)
        for tag in [a.baseline] + others:
            r = pick(fm, tag)
            if not r:
                continue
            print("%-12s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f" %
                  (tag, f(r, "iou") or 0, f(r, "iou_macro") or 0,
                   f(r, "dice") or 0, f(r, "recall") or 0, f(r, "precision") or 0))

    show("按目标尺寸", size, ["tiny", "small", "medium"], a.baseline, others)
    show("按雪况", snow, ["snow", "clean"], a.baseline, others)
    show("按飞行高度", alt, ["50m", "70m", "90m"], a.baseline, others)

    print("=" * 78)
    print("可证伪预测检验（§20.3）：ΔRecall_tiny > 0 且 >> ΔRecall_medium")
    print("=" * 78)
    for o in others:
        bt, bm = pick(size, a.baseline, "tiny"), pick(size, a.baseline, "medium")
        ot, om = pick(size, o, "tiny"), pick(size, o, "medium")
        if not (bt and bm and ot and om):
            continue
        dr_t = (f(ot, "recall") or 0) - (f(bt, "recall") or 0)
        dr_m = (f(om, "recall") or 0) - (f(bm, "recall") or 0)
        dp_t = (f(ot, "precision") or 0) - (f(bt, "precision") or 0)
        ok1 = dr_t > 0
        ok2 = dr_t > dr_m
        ok3 = dr_t > dp_t
        print("%-8s  ΔRecall_tiny=%+.4f  ΔRecall_medium=%+.4f  ΔPrec_tiny=%+.4f"
              % (o, dr_t, dr_m, dp_t))
        print("         [ %s ] tiny 召回提升" % ("OK" if ok1 else "FAIL"))
        print("         [ %s ] tiny 增益 > medium 增益" % ("OK" if ok2 else "FAIL"))
        print("         [ %s ] 涨的是 Recall 而非 Precision（注意力机制自洽）"
              % ("OK" if ok3 else "FAIL"))
        print()

if __name__ == "__main__":
    main()
