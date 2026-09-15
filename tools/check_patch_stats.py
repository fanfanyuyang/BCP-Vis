#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "datasets/EVD4UAV_processed/patch_manifest.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
fp = [int(r["foreground_pixels"]) for r in rows]


def g(x):
    return "tiny" if x < 32 * 32 else ("small" if x < 96 * 96 else "medium")

print("n=%d  fg min=%d  median=%d  max=%d  mean=%.1f"
      % (len(rows), min(fp), sorted(fp)[len(fp) // 2], max(fp), sum(fp) / len(fp)))
print("size groups:", dict(collections.Counter(g(x) for x in fp)))
by_split = collections.defaultdict(list)
for r, f in zip(rows, fp):
    by_split[r["split"]].append(f)
for s, v in by_split.items():
    print("  %-6s n=%-5d min=%-7d max=%-7d" % (s, len(v), min(v), max(v)))
assert max(fp) > 0, "所有 patch 前景像素均为 0，检查 mask 生成"
print("CHECK OK")
