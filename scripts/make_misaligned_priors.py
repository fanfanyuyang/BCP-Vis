#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", type=int, default=0, help="0=半幅滚动")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    if not files:
        raise SystemExit("no png in %s" % src)
    for i, p in enumerate(files):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        sy = args.shift or H // 2
        sx = args.shift or W // 2
        ms = np.roll(m, (sy, sx), axis=(0, 1))
        cv2.imwrite(str(out / p.name), ms)
        if (i + 1) % 1000 == 0:
            print("  %d/%d" % (i + 1, len(files)))
    print("done:", out, len(files))

if __name__ == "__main__":
    main()
