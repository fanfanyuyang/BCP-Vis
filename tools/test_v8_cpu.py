# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.priornet import CascadePriorNet

x = torch.randn(4, 3, 128, 128)
alt = ["50m", "70m", "90m", "50m"]

for mode in ("moe", "film", "none"):
    for nbg in (1, 4):
        m = CascadePriorNet(height_mode=mode, n_bg_experts=nbg)
        fg, bg = m.forward_dual(x, alt)
        lb = m.load_balance_loss()
        lb = lb if lb is not None else torch.tensor(0.0)
        (fg.sum() + bg.sum() + lb).backward()
        g = [bool(torch.isfinite(p.grad).all())
             for p in m.parameters() if p.requires_grad and p.grad is not None]
        print("mode=%-4s n_bg=%d  fg=%s bg=%s  梯度有限=%s  参数=%d"
              % (mode, nbg, tuple(fg.shape), tuple(bg.shape),
                 all(g) if g else "无参与", sum(p.numel() for p in m.parameters())))

m = CascadePriorNet(height_mode="moe")
o1, o2 = m.forward_dual(torch.randn(1, 3, 128, 128), "90m")
print("单个字符串 '90m' ->", tuple(o1.shape), tuple(o2.shape), "(应为 1,1,128,128)")

o1, o2 = m.forward_dual(torch.randn(2, 3, 128, 128), None)
print("无高度输入 ->", tuple(o1.shape), tuple(o2.shape), "(不应报错)")
print("V8 冒烟测试通过")
