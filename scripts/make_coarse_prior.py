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
    ap.add_argument("--sigma", type=float, default=12.0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    print("src images:", len(files))
    for i, p in enumerate(files):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        c = cv2.GaussianBlur(m, (0, 0), sigmaX=args.sigma, sigmaY=args.sigma)
        cv2.imwrite(str(out / p.name), c)
        if (i + 1) % 1500 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)
    print("done ->", out)

if __name__ == "__main__":
    main()
