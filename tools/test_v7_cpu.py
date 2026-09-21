# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.priornet import AltitudeMoEPriorNet

m = AltitudeMoEPriorNet(n_experts=3)

o = m(torch.randn(4, 3, 128, 128))
print("无高度      out:", tuple(o.shape))

x = torch.randn(4, 3, 128, 128)
o = m(x, ["50m", "70m", "90m", "50m"])
print("高度列表    out:", tuple(o.shape), "route:", tuple(m.last_route_w.shape))
lb = m.load_balance_loss()
print("负载均衡    :", round(float(lb), 4))

o1 = m(torch.randn(1, 3, 128, 128), "90m")
print("单个字符串  out:", tuple(o1.shape), "(若为 (1,1,128,128) 则正确)")

(o.sum() + lb).backward()
print("反向传播    : OK")

g = [bool(torch.isfinite(p.grad).all())
     for p in m.parameters() if p.requires_grad and p.grad is not None]
none_g = [n for n, p in m.named_parameters() if p.grad is None]
print("有梯度且有限:", all(g) if g else "无参与参数")
print("未参与前向  :", none_g if none_g else "无")
print("参数量      :", sum(p.numel() for p in m.parameters()))
