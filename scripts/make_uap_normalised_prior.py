# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

PROC = "<PROJECT_ROOT>/datasets/EVD4UAV_processed"
SRC_A = os.path.join(PROC, "priors_v4")
SRC_B = os.path.join(PROC, "priors_v8")
OUT = os.path.join(PROC, "priors_v33_du")

os.makedirs(OUT, exist_ok=True)


def read_gray(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def norm255(a):
    s = np.percentile(a, 99.0)
    if s < 1.0:
        return a
    return np.clip(a.astype(np.float32) / s * 255.0, 0, 255)

names = sorted(f for f in os.listdir(SRC_A) if f.endswith(".png"))
n = skipped = 0
for i, fn in enumerate(names):
    pb = os.path.join(SRC_B, fn)
    if not os.path.exists(pb):
        skipped += 1
        continue
    a = read_gray(os.path.join(SRC_A, fn))
    b = read_gray(pb)
    if a is None or b is None:
        skipped += 1
        continue
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    an = norm255(a) / 255.0
    bn = norm255(b) / 255.0
    u = np.abs(an - bn) / (an + bn + 1e-6)
    p = (1.0 - u) * an
    cv2.imwrite(os.path.join(OUT, fn), np.clip(p * 255.0, 0, 255).astype(np.uint8))
    n += 1
    if (i + 1) % 2000 == 0:
        print("done", i + 1, flush=True)

print("OK priors_v33_du =", n, " skipped =", skipped)
