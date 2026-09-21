# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import UpBlock, ConvBNAct


class ResNet18UNet(nn.Module):

    def __init__(self, in_channels=3, pretrained=True, init_prior="mean",
                 out_ch=1, decoder_base=256, alt_cond=False, alt_emb_dim=32,
                 prior_index=3):
        super().__init__()
        from torchvision import models
        try:
            from torchvision.models import ResNet18_Weights
            w = ResNet18_Weights.DEFAULT if pretrained else None
            enc = models.resnet18(weights=w)
        except Exception:
            enc = models.resnet18(pretrained=pretrained)

        old = enc.conv1
        if in_channels == 3:
            conv1 = old
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                w_old = old.weight
                if init_prior == "mean":
                    extra = w_old.mean(dim=1, keepdim=True)
                else:
                    extra = torch.zeros_like(w_old[:, :1])
                pad = extra.repeat(1, in_channels - 3, 1, 1)
                conv1.weight.copy_(torch.cat([w_old, pad], dim=1))
        self.enc = enc
        self.enc.conv1 = conv1

        self.in_channels = in_channels
        self.up3 = UpBlock(512, 256, decoder_base)
        self.up2 = UpBlock(decoder_base, 128, decoder_base // 2)
        self.up1 = UpBlock(decoder_base // 2, 64, decoder_base // 4)
        self.up0 = UpBlock(decoder_base // 4, 64, decoder_base // 8)
        self.head = nn.Conv2d(decoder_base // 8, out_ch, 1)

        self.alt_cond = bool(alt_cond)
        self.prior_index = int(prior_index)
        if self.alt_cond:
            self.alt_mlp = nn.Sequential(
                nn.Linear(1, alt_emb_dim), nn.ReLU(inplace=True),
                nn.Linear(alt_emb_dim, 2))
            nn.init.zeros_(self.alt_mlp[-1].weight)
            nn.init.zeros_(self.alt_mlp[-1].bias)

    @staticmethod

    def _alt_to_tensor(altitude, B, dev):
        if altitude is None:
            return None
        if isinstance(altitude, str):
            altitude = [altitude]
        if torch.is_tensor(altitude):
            t = altitude.float()
            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.view(-1, 1)
        else:
            vals = []
            for a in altitude:
                if a is None or a == "":
                    vals.append(0.0)
                elif isinstance(a, (int, float)):
                    vals.append(float(a))
                else:
                    s = str(a).strip().lower().replace("m", "")
                    try:
                        vals.append(float(s))
                    except ValueError:
                        vals.append(0.0)
            t = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
        t = t.to(dev)
        if t.shape[0] != B:
            t = t.expand(B, -1)
        return (t - 70.0) / 40.0

    def forward(self, x, altitude=None):
        if getattr(self, "alt_cond", False) and altitude is not None \
                and x.shape[1] > self.prior_index:
            t = self._alt_to_tensor(altitude, x.shape[0], x.device)
            if t is not None:
                gb = self.alt_mlp(t)
                gamma = gb[:, 0].view(-1, 1, 1, 1)
                beta = gb[:, 1].view(-1, 1, 1, 1)
                pi = self.prior_index
                prior = x[:, pi:pi + 1]
                prior = (1.0 + gamma) * prior + beta
                x = torch.cat([x[:, :pi], prior, x[:, pi + 1:]], dim=1)
        _, _, H, W = x.shape
        e0 = self.enc.relu(self.enc.bn1(self.enc.conv1(x)))
        p = self.enc.maxpool(e0)
        e1 = self.enc.layer1(p)
        e2 = self.enc.layer2(e1)
        e3 = self.enc.layer3(e2)
        e4 = self.enc.layer4(e3)
        d = self.up3(e4, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        d = self.up0(d, e0)
        out = self.head(d)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


class MiniUNet(nn.Module):

    def __init__(self, in_channels=3, base=32, out_ch=1):
        super().__init__()
        self.e1 = ConvBNAct(in_channels, base)
        self.e2 = ConvBNAct(base, base * 2)
        self.e3 = ConvBNAct(base * 2, base * 4)
        self.e4 = ConvBNAct(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = UpBlock(base * 8, base * 4, base * 4)
        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.in_channels = in_channels

    def forward(self, x, altitude=None):
        _, _, H, W = x.shape
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        d = self.u3(self.e4(self.pool(s3)), s3)
        d = self.u2(d, s2)
        d = self.u1(d, s1)
        out = self.head(d)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


def build_final_cnn(cfg=None, **kw):
    cfg = dict(cfg or {})
    cfg.update(kw)
    if cfg.get("arch", "resnet18_unet") == "mini_unet":
        return MiniUNet(in_channels=cfg.get("in_channels", 3),
                        base=cfg.get("base", 32))
    return ResNet18UNet(
        in_channels=cfg.get("in_channels", 3),
        pretrained=cfg.get("pretrained", True),
        init_prior=cfg.get("init_prior", "mean"),
        decoder_base=cfg.get("decoder_base", 256),
        alt_cond=cfg.get("alt_cond", False),
        alt_emb_dim=cfg.get("alt_emb_dim", 32),
    )
