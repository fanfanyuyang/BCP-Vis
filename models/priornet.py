# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DSConv, FiLM, UpBlock


class MiniUNetPriorNet(nn.Module):

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = ConvBNAct(in_ch, base)
        self.e2 = ConvBNAct(base, base * 2)
        self.e3 = ConvBNAct(base * 2, base * 4)
        self.e4 = ConvBNAct(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = UpBlock(base * 8, base * 4, base * 4)
        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, altitude=None):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.e4(self.pool(s3))
        d = self.u3(b, s3)
        d = self.u2(d, s2)
        d = self.u1(d, s1)
        return self.head(d)


class LiteHRPriorNet(nn.Module):

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64):
        super().__init__()
        self.stem = ConvBNAct(in_ch, c2, 3, s=2)
        self.a1 = DSConv(c2, c2)
        self.b_down = DSConv(c2, c4, 3, s=2)
        self.b1 = DSConv(c4, c4)
        self.c_down = DSConv(c4, c8, 3, s=2)
        self.c1 = DSConv(c8, c8)

        self.c_up = nn.Conv2d(c8, c4, 1)
        self.b_up = nn.Conv2d(c4, c2, 1)
        self.fuse = ConvBNAct(c2, fuse)
        self.head = nn.Sequential(
            ConvBNAct(fuse, fuse // 2),
            nn.Conv2d(fuse // 2, 1, 1),
        )

    def _feat(self, x):
        a = self.a1(self.stem(x))
        b = self.b1(self.b_down(a))
        c = self.c1(self.c_down(b))

        c = F.interpolate(self.c_up(c), size=b.shape[-2:], mode="bilinear",
                          align_corners=False)
        b = b + c
        b = F.interpolate(self.b_up(b), size=a.shape[-2:], mode="bilinear",
                          align_corners=False)
        a = a + b
        return self.fuse(a)

    def forward(self, x, altitude=None):
        _, _, H, W = x.shape
        out = self.head(self._feat(x))
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


class AltitudeAwarePriorNet(LiteHRPriorNet):

    ALT_BUCKETS = [50.0, 70.0, 90.0]

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64, emb_dim=64):
        super().__init__(in_ch, c2, c4, c8, fuse)
        self.emb_dim = emb_dim
        self.accepts_altitude = True
        self.alt_mlp = nn.Sequential(
            nn.Linear(1, emb_dim), nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.film_a = FiLM(c2, emb_dim)
        self.film_b = FiLM(c4, emb_dim)

    @staticmethod

    def _to_scalar(altitude):
        if isinstance(altitude, str):
            altitude = [altitude]
        if isinstance(altitude, torch.Tensor):
            t = altitude
            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.view(-1, 1)
            return t.float()
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
        return torch.tensor(vals, dtype=torch.float32).view(-1, 1)

    def forward(self, x, altitude=None):
        _, _, H, W = x.shape
        a = self.a1(self.stem(x))
        b = self.b1(self.b_down(a))
        c = self.c1(self.c_down(b))

        if altitude is not None:
            dev = x.device
            t = self._to_scalar(altitude).to(dev)
            if t.shape[0] != x.shape[0]:
                t = t.expand(x.shape[0], -1)
            t = (t - 70.0) / 40.0
            emb = self.alt_mlp(t)
            b = self.film_b(b, emb)
            a = self.film_a(a, emb)

        c = F.interpolate(self.c_up(c), size=b.shape[-2:],
                          mode="bilinear", align_corners=False)
        b = b + c
        b = F.interpolate(self.b_up(b), size=a.shape[-2:],
                          mode="bilinear", align_corners=False)
        a = a + b
        f = self.fuse(a)
        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


class DualHeadPriorNet(LiteHRPriorNet):

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64, bg_dim=32):
        super().__init__(in_ch, c2, c4, c8, fuse)
        self.bg_tower = nn.Sequential(
            DSConv(fuse, bg_dim),
            nn.Conv2d(bg_dim, 1, 1),
        )
        self.dual = True
        self.last_bg = None

    def forward_dual(self, x, altitude=None):
        _, _, H, W = x.shape
        f = self._feat(x)
        fg = self.head(f)
        bg = self.bg_tower(f)
        if fg.shape[-2:] != (H, W):
            fg = F.interpolate(fg, size=(H, W), mode="bilinear", align_corners=False)
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        self.last_bg = bg.detach()
        return fg, bg

    def forward(self, x, altitude=None):
        fg, _ = self.forward_dual(x, altitude)
        return fg


class MoEBackgroundPriorNet(DualHeadPriorNet):

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64,
                 bg_dim=32, n_experts=4):
        super().__init__(in_ch, c2, c4, c8, fuse, bg_dim)
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(DSConv(fuse, bg_dim), nn.Conv2d(bg_dim, 1, 1))
            for _ in range(n_experts)
        ])
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fuse, n_experts),
        )
        self.last_route_w = None

    def forward_dual(self, x, altitude=None):
        _, _, H, W = x.shape
        f = self._feat(x)
        fg = self.head(f)
        w = torch.softmax(self.router(f), dim=1)
        bg = sum(w[:, k:k + 1, None, None] * self.experts[k](f)
                 for k in range(self.n_experts))
        if fg.shape[-2:] != (H, W):
            fg = F.interpolate(fg, size=(H, W), mode="bilinear", align_corners=False)
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        self.last_route_w = w.detach()
        self.last_bg = bg.detach()
        return fg, bg

    def load_balance_loss(self):
        w = self.last_route_w
        if w is None:
            return None
        mean_w = w.mean(dim=0)
        K = w.shape[1]
        return K * (mean_w * mean_w).sum()


class AltitudeMoEPriorNet(LiteHRPriorNet):

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64,
                 emb_dim=64, n_experts=3, expert_dim=32):
        super().__init__(in_ch, c2, c4, c8, fuse)
        self.n_experts = n_experts
        self.alt_emb = nn.Sequential(
            nn.Linear(1, emb_dim), nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(DSConv(fuse, expert_dim),
                          nn.Conv2d(expert_dim, 1, 1))
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(emb_dim, n_experts)
        self.dual = False
        self.accepts_altitude = True
        self.last_route_w = None

    @staticmethod

    def _to_scalar(altitude):
        if isinstance(altitude, str):
            altitude = [altitude]
        if isinstance(altitude, torch.Tensor):
            t = altitude
            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.view(-1, 1)
            return t.float()
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
        return torch.tensor(vals, dtype=torch.float32).view(-1, 1)

    def forward(self, x, altitude=None):
        _, _, H, W = x.shape
        f = self._feat(x)
        B = x.shape[0]

        if altitude is None:
            w = torch.full((B, self.n_experts), 1.0 / self.n_experts,
                           device=x.device, dtype=x.dtype)
        else:
            t = self._to_scalar(altitude).to(x.device)
            if t.shape[0] != B:
                t = t.expand(B, -1)
            t = (t - 70.0) / 40.0
            e = self.alt_emb(t)
            w = torch.softmax(self.router(e), dim=1)

        out = sum(w[:, k:k + 1, None, None] * self.experts[k](f)
                  for k in range(self.n_experts))
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear",
                                align_corners=False)
        self.last_route_w = w.detach()
        return out

    def load_balance_loss(self):
        w = self.last_route_w
        if w is None:
            return None
        mean_w = w.mean(dim=0)
        K = w.shape[1]
        return K * (mean_w * mean_w).sum()


class CascadePriorNet(LiteHRPriorNet):

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64,
                 height_mode="moe", n_alt_experts=3, emb_dim=64,
                 alt_expert_dim=32, bg_dim=32, n_bg_experts=1):
        super().__init__(in_ch, c2, c4, c8, fuse)
        self.height_mode = height_mode
        self.n_bg_experts = n_bg_experts
        self.dual = True
        self.accepts_altitude = True
        self.last_route_w = None
        self.last_bg_route_w = None

        if height_mode == "moe":
            self.alt_emb = nn.Sequential(
                nn.Linear(1, emb_dim), nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim))
            self.alt_experts = nn.ModuleList([
                nn.Sequential(DSConv(fuse, alt_expert_dim),
                              nn.Conv2d(alt_expert_dim, 1, 1))
                for _ in range(n_alt_experts)])
            self.alt_router = nn.Linear(emb_dim, n_alt_experts)
        elif height_mode == "film":
            self.alt_mlp = nn.Sequential(
                nn.Linear(1, emb_dim), nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim))
            self.film_a = FiLM(c2, emb_dim)
            self.film_b = FiLM(c4, emb_dim)

        if n_bg_experts > 1:
            self.bg_experts = nn.ModuleList([
                nn.Sequential(DSConv(fuse, bg_dim), nn.Conv2d(bg_dim, 1, 1))
                for _ in range(n_bg_experts)])
            self.bg_router = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(fuse, n_bg_experts))
        else:
            self.bg_tower = nn.Sequential(DSConv(fuse, bg_dim),
                                          nn.Conv2d(bg_dim, 1, 1))

    @staticmethod

    def _to_scalar(altitude):
        if isinstance(altitude, str):
            altitude = [altitude]
        if isinstance(altitude, torch.Tensor):
            t = altitude
            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.view(-1, 1)
            return t.float()
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
        return torch.tensor(vals, dtype=torch.float32).view(-1, 1)

    def _alt_tensor(self, altitude, B, dev):
        if altitude is None:
            return None
        t = self._to_scalar(altitude).to(dev)
        if t.shape[0] != B:
            t = t.expand(B, -1)
        return (t - 70.0) / 40.0

    def _feat_film(self, x, altitude):
        a = self.a1(self.stem(x))
        b = self.b1(self.b_down(a))
        c = self.c1(self.c_down(b))
        t = self._alt_tensor(altitude, x.shape[0], x.device)
        if t is not None:
            emb = self.alt_mlp(t)
            b = self.film_b(b, emb)
            a = self.film_a(a, emb)
        c = F.interpolate(self.c_up(c), size=b.shape[-2:], mode="bilinear",
                          align_corners=False)
        b = b + c
        b = F.interpolate(self.b_up(b), size=a.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(a + b)

    def forward_dual(self, x, altitude=None):
        _, _, H, W = x.shape
        B = x.shape[0]
        dev = x.device

        if self.height_mode == "film":
            f = self._feat_film(x, altitude)
            fg = self.head(f)
        else:
            f = self._feat(x)
            if self.height_mode == "moe" and altitude is not None:
                t = self._alt_tensor(altitude, B, dev)
                w = torch.softmax(self.alt_router(self.alt_emb(t)), dim=1)
                fg = sum(w[:, k:k + 1, None, None] * self.alt_experts[k](f)
                         for k in range(len(self.alt_experts)))
                self.last_route_w = w.detach()
            else:
                fg = self.head(f)

        if self.n_bg_experts > 1:
            wb = torch.softmax(self.bg_router(f), dim=1)
            bg = sum(wb[:, k:k + 1, None, None] * self.bg_experts[k](f)
                     for k in range(self.n_bg_experts))
            self.last_bg_route_w = wb.detach()
        else:
            bg = self.bg_tower(f)

        if fg.shape[-2:] != (H, W):
            fg = F.interpolate(fg, size=(H, W), mode="bilinear", align_corners=False)
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        self.last_bg = bg.detach()
        return fg, bg

    def forward(self, x, altitude=None):
        fg, _ = self.forward_dual(x, altitude)
        return fg

    def load_balance_loss(self):
        loss = None
        for w in (self.last_route_w, self.last_bg_route_w):
            if w is None:
                continue
            K = w.shape[1]
            v = K * (w.mean(dim=0) ** 2).sum()
            loss = v if loss is None else loss + v
        return loss


class CategoryBgPriorNet(DualHeadPriorNet):

    N_CLASS = 3

    def __init__(self, in_ch=3, c2=32, c4=64, c8=128, fuse=64, bg_dim=32, emb_dim=32):
        super().__init__(in_ch, c2, c4, c8, fuse, bg_dim)
        self.class_aware = True
        self.emb_dim = emb_dim
        self.cls_emb = nn.Embedding(self.N_CLASS, emb_dim)
        self.bg_film = FiLM(fuse, emb_dim)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(fuse, self.N_CLASS))

    def forward_dual(self, x, altitude=None):
        _, _, H, W = x.shape
        f = self._feat(x)
        fg = self.head(f)
        cls = self.class_head(f)
        c = cls.argmax(dim=1)
        e = self.cls_emb(c)
        f_bg = self.bg_film(f, e)
        bg = self.bg_tower(f_bg)
        if fg.shape[-2:] != (H, W):
            fg = F.interpolate(fg, size=(H, W), mode="bilinear", align_corners=False)
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        self.last_bg = bg.detach()
        self.last_cls = cls.detach()
        return fg, bg, cls

    def forward(self, x, altitude=None):
        return self.forward_dual(x, altitude)[0]


def build_priornet(cfg=None, **kw):
    cfg = dict(cfg or {})
    cfg.update(kw)
    arch = cfg.get("arch", "litehr")
    if arch in ("v1", "mini", "miniunet"):
        return MiniUNetPriorNet(in_ch=cfg.get("in_ch", 3), base=cfg.get("base", 32))
    if arch in ("v4", "altitude"):
        return AltitudeAwarePriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64), emb_dim=cfg.get("emb_dim", 64))
    if arch in ("v5", "dual", "dual_head"):
        return DualHeadPriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64), bg_dim=cfg.get("bg_dim", 32))
    if arch in ("v8", "cascade", "cascade_prior"):
        return CascadePriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64),
            height_mode=cfg.get("height_mode", "moe"),
            n_alt_experts=cfg.get("n_alt_experts", 3),
            emb_dim=cfg.get("emb_dim", 64),
            alt_expert_dim=cfg.get("alt_expert_dim", 32),
            bg_dim=cfg.get("bg_dim", 32),
            n_bg_experts=cfg.get("n_bg_experts", 1))
    if arch in ("v7", "alt_moe", "altitude_moe"):
        return AltitudeMoEPriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64), emb_dim=cfg.get("emb_dim", 64),
            n_experts=cfg.get("n_experts", 3),
            expert_dim=cfg.get("expert_dim", 32))
    if arch in ("v5-moe", "moe", "moe_bg"):
        return MoEBackgroundPriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64), bg_dim=cfg.get("bg_dim", 32),
            n_experts=cfg.get("n_experts", 4))
    if arch in ("v5cat", "cat_bg", "category_bg"):
        return CategoryBgPriorNet(
            in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
            c4=cfg.get("c4", 64), c8=cfg.get("c8", 128),
            fuse=cfg.get("fuse", 64), bg_dim=cfg.get("bg_dim", 32),
            emb_dim=cfg.get("emb_dim", 32))
    return LiteHRPriorNet(
        in_ch=cfg.get("in_ch", 3), c2=cfg.get("c2", 32),
        c4=cfg.get("c4", 64), c8=cfg.get("c8", 128), fuse=cfg.get("fuse", 64))
