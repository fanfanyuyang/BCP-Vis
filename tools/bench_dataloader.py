# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets_py.patch_dataset import PatchDataset


class GroupedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, sources, batch_size, seed=42, shuffle=True):
        self.bs = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        buckets = {}
        for i, s in enumerate(sources):
            buckets.setdefault(s, []).append(i)
        self.batches = []
        pool = []
        for _, idxs in buckets.items():
            for j in range(0, len(idxs), batch_size):
                chunk = idxs[j:j + batch_size]
                if len(chunk) == batch_size:
                    self.batches.append(chunk)
                else:
                    pool.extend(chunk)
        for j in range(0, len(pool), batch_size):
            self.batches.append(pool[j:j + batch_size])

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[i] for i in perm]
        self.epoch += 1
        for bi in order:
            yield self.batches[bi]

    def __len__(self):
        return len(self.batches)


def bench(ds, bs, nw, iters, sampler=None, tag=""):
    if sampler is None:
        dl = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True,
                        shuffle=True, persistent_workers=nw > 0)
    else:
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=nw,
                        pin_memory=True, persistent_workers=nw > 0)
    it = iter(dl)
    for _ in range(3):
        next(it)
    t0 = time.time()
    n = 0
    for i, b in enumerate(it):
        if i >= iters:
            break
        n += b[0].shape[0]
    dt = time.time() - t0
    print("  %-22s workers=%-3d %5.1f it/s  %6.1f patch/s  (%.1fs / %d it)"
          % (tag or ("shuffle" if sampler is None else "grouped"), nw,
             iters / dt, n / dt, dt, iters))
    return iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--raw", default="datasets/EVD4UAV/raw/EVD4UAV")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()

    proc = Path(a.proc).resolve()
    ds = PatchDataset(proc=proc, raw=a.raw, split="train", task="final",
                      prior_mode="none", augment=True, seed=42)
    print("patches:", len(ds))

    rows = list(csv.DictReader((proc / "patch_manifest.csv").open(encoding="utf-8")))
    rows = [r for r in rows if r["split"] == "train"]
    print("train patches:", len(rows))
    per_img = {}
    for r in rows:
        per_img[r["source_image"]] = per_img.get(r["source_image"], 0) + 1
    import statistics
    print("每张原图平均产生 %.2f 个 patch（= JPEG 被重复解码的次数）"
          % statistics.mean(per_img.values()))

    print("\n--- 数据加载吞吐 ---")
    bench(ds, a.batch_size, 8, a.iters, tag="shuffle-plain")
    bench(ds, a.batch_size, 16, a.iters, tag="shuffle-plain")
    bench(ds, a.batch_size, 0, a.iters, tag="shuffle-plain")

    sources = [r["source_image"] for r in rows]
    gs = GroupedBatchSampler(sources, a.batch_size, seed=42)
    print("\n--- 按原图分组组批 ---")
    bench(ds, a.batch_size, 8, a.iters, sampler=gs, tag="grouped")
    bench(ds, a.batch_size, 16, a.iters, sampler=gs, tag="grouped")

if __name__ == "__main__":
    main()
