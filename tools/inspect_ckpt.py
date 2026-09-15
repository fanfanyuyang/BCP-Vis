# -*- coding: utf-8 -*-
import sys

import torch

for p in sys.argv[1:]:
    try:
        ck = torch.load(p, map_location="cpu")
    except Exception as exc:
        print(p, "加载失败:", exc)
        continue
    print("==", p)
    print("   keys      :", list(ck.keys()))
    cfg = ck.get("cfg") or {}
    print("   cfg.model :", cfg.get("model"))
    sd = ck.get("model", ck)
    if isinstance(sd, dict):
        ks = [k for k in sd.keys()][:6]
        print("   state_dict 前几个 key:", ks)
        for k in ("stem.c.weight", "bg_tower.0.pw.weight", "head.1.weight"):
            if k in sd:
                print("   ", k, tuple(sd[k].shape))
