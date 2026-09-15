# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):

    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        p = (k - 1) // 2
        self.c = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.b = nn.BatchNorm2d(cout)
        self.a = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.a(self.b(self.c(x)))


class DSConv(nn.Module):

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        p = (k - 1) // 2
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.dbn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pbn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dbn(self.dw(x)))
        return self.act(self.pbn(self.pw(x)))


class UpBlock(nn.Module):

    def __init__(self, cin, skip_c, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1 = ConvBNAct(cin + skip_c, cout)
        self.c2 = ConvBNAct(cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.c2(self.c1(torch.cat([x, skip], dim=1)))


class FiLM(nn.Module):

    def __init__(self, n_feat, emb_dim):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, n_feat)
        self.beta = nn.Linear(emb_dim, n_feat)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat, emb):
        g = self.gamma(emb).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(emb).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + g) + b


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_flops(model, input_size=(1, 3, 768, 768), device="cuda"):
    try:
        from fvcore.nn import FlopCountAnalysis
        m = model.eval().to(device)
        x = torch.zeros(input_size, device=device)
        with torch.no_grad():
            flops = FlopCountAnalysis(m, x).total()
        return flops
    except Exception:
        return None
