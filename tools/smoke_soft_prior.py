#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SMOKE = Path("/tmp/smoke")
(SMOKE / "masks").mkdir(parents=True, exist_ok=True)

m = np.zeros((1080, 1920), np.uint8)
cv2.rectangle(m, (300, 400), (360, 450), 255, -1)
cv2.rectangle(m, (1200, 700), (1250, 740), 255, -1)
cv2.imwrite(str(SMOKE / "masks" / "DJI_0001.png"), m)
print("synthetic mask:", m.shape, "fg_ratio=%.6f" % (m.mean() / 255.0))

cmd = [sys.executable, "scripts/build_soft_prior_targets.py",
       "--mask-dir", str(SMOKE / "masks"),
       "--out", str(SMOKE / "soft"),
       "--sigmas", "8,16,24,32,48",
       "--n-vis", "1", "--dry-run"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)

sys.path.insert(0, "scripts")
from build_soft_prior_targets import make_soft_prior

fg = m > 0
for sigma in (8, 24, 48):
    p = make_soft_prior(fg, sigma)
    assert p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0, "值域越界"
    assert np.allclose(p[fg], 1.0), "前景内部必须为 1"
    nb = ((p > 0.3) & (~fg)).mean()
    far = p[~fg][p[~fg] > 0].min()
    print("sigma=%-3d  prior_min=%.5f  max=%.5f  mean=%.5f  nb(>0.3)=%.5f"
          % (sigma, p.min(), p.max(), p.mean(), nb))

print("SMOKE TEST PASSED")
