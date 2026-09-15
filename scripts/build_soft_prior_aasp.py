#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

_G = {}


def _init(mask_dir, out, fmt, sigma0, alt_ref, alt_map):
    _G.update(mask_dir=Path(mask_dir), out=Path(out), fmt=fmt,
              sigma0=float(sigma0), alt_ref=float(alt_ref), alt=alt_map)


def make_soft(binary, sigma):
    fg = binary.astype(bool)
    p = np.zeros(fg.shape, dtype=np.float32)
    p[fg] = 1.0
    if not fg.any():
        return p
    d = distance_transform_edt(~fg).astype(np.float32)
    out = ~fg
    p[out] = np.exp(-(d[out] ** 2) / (2.0 * float(sigma) ** 2))
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def _work(mpath):
    mpath = Path(mpath)
    stem = mpath.stem
    m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    h = _G["alt"].get(stem, _G["alt_ref"]) or _G["alt_ref"]
    sigma = _G["sigma0"] * (_G["alt_ref"] / h)
    p = make_soft((m > 0), sigma)
    if _G["fmt"] == "npy":
        np.save(str(_G["out"] / (stem + ".npy")), p)
    else:
        cv2.imwrite(str(_G["out"] / (stem + ".png")), (p * 255).astype(np.uint8))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="datasets/EVD4UAV_processed")
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--out", default="datasets/EVD4UAV_processed/soft_priors_aasp")
    ap.add_argument("--sigma0", type=float, default=32.0)
    ap.add_argument("--alt-ref", type=float, default=70.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--format", default="png", choices=["png", "npy"])
    args = ap.parse_args()

    proc = Path(args.proc)
    mask_dir = Path(args.mask_dir) if args.mask_dir else (proc / "binary_masks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    alt = {}
    sp = proc / "split.csv"
    if sp.exists():
        for r in csv.DictReader(sp.open(encoding="utf-8")):
            a = str(r.get("altitude", "")).strip().lower().replace("m", "")
            try:
                alt[r["image_id"]] = float(a)
            except ValueError:
                pass
    masks = sorted(mask_dir.glob("*.png"))
    print("masks:", len(masks), "| altitudes:", len(alt))

    from collections import Counter
    c = Counter()
    for m in masks[:5000]:
        h = alt.get(m.stem, args.alt_ref) or args.alt_ref
        c["%gm(sigma=%.1f)" % (h, args.sigma0 * args.alt_ref / h)] += 1
    print("sigma 分布抽样:", dict(c))

    todo = [str(m) for m in masks]
    ok = 0
    with Pool(args.workers, initializer=_init,
              initargs=(str(mask_dir), str(out), args.format,
                        args.sigma0, args.alt_ref, alt)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, todo, chunksize=32)):
            ok += int(r)
            if (i + 1) % 1000 == 0:
                print("  %d/%d" % (i + 1, len(todo)), flush=True)
    print("done:", ok, "/", len(todo), "->", out)

if __name__ == "__main__":
    main()
