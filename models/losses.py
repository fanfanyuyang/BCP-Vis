# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(target >= 0.5, p, 1 - p)
        a = torch.where(target >= 0.5,
                        torch.full_like(pt, self.alpha),
                        torch.full_like(pt, 1 - self.alpha))
        loss = a * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = (0, 2, 3) if p.dim() == 4 else (0,)
        inter = (p * target).sum(dim=dims)
        union = p.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return (1 - dice).mean()


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target)


class CombinedLoss(nn.Module):

    def __init__(self, main="focal", lambda_dice=1.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.main = FocalLoss(focal_alpha, focal_gamma) if main == "focal" else BCELoss()
        self.dice = DiceLoss()
        self.lambda_dice = lambda_dice
        self.last_parts = {}

    def forward(self, logits, target):
        lm = self.main(logits, target)
        ld = self.dice(logits, target)
        self.last_parts = {"main": float(lm.detach()), "dice": float(ld.detach())}
        return lm + self.lambda_dice * ld


class DistillFeatureLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, fs, ft):
        fs = F.normalize(fs.flatten(1), dim=1)
        ft = F.normalize(ft.flatten(1), dim=1)
        return (1.0 - (fs * ft).sum(dim=1)).mean()


def build_loss(cfg):
    return CombinedLoss(
        main=cfg.get("main", "focal"),
        lambda_dice=cfg.get("lambda_dice", 1.0),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )
