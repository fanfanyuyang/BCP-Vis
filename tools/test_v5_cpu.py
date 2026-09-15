# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.priornet import (DualHeadPriorNet, LiteHRPriorNet,
                             MoEBackgroundPriorNet)

x = torch.randn(2, 3, 128, 128)

m1 = DualHeadPriorNet()
fg, bg = m1.forward_dual(x)
print("DualHead   out:", tuple(fg.shape), tuple(bg.shape))
(fg.sum() + bg.sum()).backward()
print("DualHead   backward OK")

m2 = MoEBackgroundPriorNet(n_experts=4)
fg, bg = m2.forward_dual(x)
print("MoE        out:", tuple(fg.shape), tuple(bg.shape),
      "route", tuple(m2.last_route_w.shape))
lb = m2.load_balance_loss()
print("MoE        load_balance =", round(float(lb), 4))
(fg.sum() + bg.sum() + lb).backward()
print("MoE        backward OK")


def np_(m):
    return sum(q.numel() for q in m.parameters())

print("params: V3=%d  DualHead=%d  MoE=%d"
      % (np_(LiteHRPriorNet()), np_(m1), np_(m2)))

from models import build_priornet
for a in ("dual", "moe"):
    mm = build_priornet({"arch": a, "n_experts": 4})
    print("build arch=%-5s -> %s  params=%d"
          % (a, type(mm).__name__, np_(mm)))
