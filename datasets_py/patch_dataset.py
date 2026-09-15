# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def _imread(p, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(str(p), flag)
    return img


class PatchDataset(Dataset):

    def __init__(self,
                 proc="datasets/EVD4UAV_processed",
                 raw="datasets/EVD4UAV/raw",
                 split="train",
                 task="prior",
                 prior_mode=None,
                 prior_dir=None,
                 prior_dir2=None,
                 soft_dir=None,
                 augment=False,
                 class_aware=False,
                 class_file=None,
                 seed=42,
                 image_norm=True,
                 manifest="patch_manifest.csv"):
        self.proc = Path(proc).resolve()
        self.raw = Path(raw).resolve()
        self.task = task
        self.augment = augment
        self.manifest_name = manifest
        self.rng = random.Random(seed)
        self.image_norm = image_norm
        self.class_aware = class_aware
        self.cls_map = {}
        if class_aware:
            cf = Path(class_file).resolve() if class_file else (self.proc / "image_classes.csv")
            if not cf.exists():
                raise SystemExit(
                    "ERROR: 缺少 %s，请先运行 scripts/build_class_labels.py" % cf)
            for r in csv.DictReader(cf.open(encoding="utf-8")):
                try:
                    self.cls_map[r["image_id"]] = int(r["class_idx"])
                except (ValueError, KeyError):
                    self.cls_map[r["image_id"]] = -1
        self.prior_mode = prior_mode or ("predicted" if task == "final" else "none")

        mf = self.proc / self.manifest_name
        if not mf.exists():
            raise SystemExit("ERROR: 缺少 %s，请先运行 make_patches.py" % mf)
        self.rows = [r for r in csv.DictReader(mf.open(encoding="utf-8"))
                     if r["split"] == split]
        if not self.rows:
            raise SystemExit("ERROR: split=%s 没有 patch" % split)

        self.img_index = {}
        for p in self.raw.rglob("*"):
            if p.suffix in IMG_EXT:
                self.img_index.setdefault(p.stem, p)
        self.mask_dir = self.proc / "binary_masks"
        self.soft_dir = Path(soft_dir).resolve() if soft_dir else (self.proc / "soft_priors")
        self.prior_dir = Path(prior_dir).resolve() if prior_dir else (self.proc / "oof_priors")
        self.prior_dir2 = Path(prior_dir2).resolve() if prior_dir2 else None

    def __len__(self):
        return len(self.rows)

    def _load_rgb(self, stem):
        p = self.img_index.get(stem)
        if p is None:
            hits = list(self.raw.rglob(stem + ".*"))
            hits = [h for h in hits if h.suffix in IMG_EXT]
            if not hits:
                return None
            p = hits[0]
            self.img_index[stem] = p
        img = _imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, stem):
        p = self.mask_dir / (stem + ".png")
        if not p.exists():
            return None
        return _imread(p, cv2.IMREAD_GRAYSCALE)

    def _load_soft(self, stem):
        p = self.soft_dir / (stem + ".png")
        if p.exists():
            import cv2
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            return m.astype(np.float32) / 255.0
        p = self.soft_dir / (stem + ".npy")
        if p.exists():
            return np.load(str(p)).astype(np.float32)
        return None

    def _load_pred_prior(self, stem):
        for cand in (self.prior_dir / (stem + ".png"),
                     self.prior_dir / (stem + ".npy")):
            if cand.exists():
                if cand.suffix == ".npy":
                    return np.load(str(cand)).astype(np.float32)
                import cv2
                m = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    return None
                return m.astype(np.float32) / 255.0
        return None

    def _load_pred_prior2(self, stem):
        if self.prior_dir2 is None:
            return None
        for cand in (self.prior_dir2 / (stem + ".png"),
                     self.prior_dir2 / (stem + ".npy")):
            if cand.exists():
                if cand.suffix == ".npy":
                    return np.load(str(cand)).astype(np.float32)
                import cv2
                m = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    return None
                return m.astype(np.float32) / 255.0
        return None

    def __getitem__(self, idx):
        r = self.rows[idx]
        stem = r["source_image"]
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])

        rgb = self._load_rgb(stem)
        mask = self._load_mask(stem)
        if rgb is None or mask is None:
            return self[(idx + 1) % len(self)]
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        flip_h = self.augment and self.rng.random() < 0.5
        flip_v = self.augment and self.rng.random() < 0.5

        def _geo(a):
            if flip_h:
                a = np.ascontiguousarray(a[:, ::-1])
            if flip_v:
                a = np.ascontiguousarray(a[::-1, :])
            return a

        rgb = _geo(rgb[y1:y2, x1:x2])
        mask = _geo(mask[y1:y2, x1:x2])

        extra = None
        if self.task == "prior":
            extra = self._load_soft(stem)
            if extra is None:
                extra = (mask > 0).astype(np.float32)
            extra = _geo(extra[y1:y2, x1:x2])
        elif self.prior_mode == "predicted":
            extra = self._load_pred_prior(stem)
            if extra is not None:
                extra = _geo(extra[y1:y2, x1:x2])
            else:
                extra = np.zeros(mask.shape, dtype=np.float32)
        elif self.prior_mode == "predicted2":
            extra = self._load_pred_prior(stem)
            extra = _geo(extra[y1:y2, x1:x2]) if extra is not None \
                else np.zeros(mask.shape, dtype=np.float32)
            extra2 = self._load_pred_prior2(stem)
            extra2 = _geo(extra2[y1:y2, x1:x2]) if extra2 is not None \
                else np.zeros(mask.shape, dtype=np.float32)

        if self.augment and self.rng.random() < 0.3:
            a = self.rng.uniform(0.85, 1.15)
            b = self.rng.uniform(-12, 12)
            rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        t_rgb = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        if self.image_norm:
            t_rgb = t_rgb / 255.0
        t_mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        if self.task == "prior":
            t_tgt = torch.from_numpy(np.ascontiguousarray(extra)).float().unsqueeze(0)
            if self.class_aware:
                cidx = self.cls_map.get(stem, -1)
                return t_rgb, t_tgt, r.get("altitude", ""), \
                    torch.tensor(cidx, dtype=torch.long)
            return t_rgb, t_tgt, r.get("altitude", "")

        if self.prior_mode == "none":
            meta = {"patch_id": r["patch_id"], "altitude": r.get("altitude", ""),
                    "snow": r.get("snow", ""), "source_image": stem,
                    "fg_pixels": int(r.get("foreground_pixels", 0))}
            return t_rgb, t_mask, meta

        if self.prior_mode == "zero":
            p = np.zeros(mask.shape, dtype=np.float32)
        elif self.prior_mode == "random":
            p = np.random.RandomState(
                (hash(stem) + y1 * 7919 + x1) % (2 ** 31)
            ).rand(*mask.shape).astype(np.float32)
        elif self.prior_mode == "oracle":
            p = (mask > 0).astype(np.float32)
        elif self.prior_mode == "predicted2":
            p = extra if extra is not None else np.zeros(mask.shape, dtype=np.float32)
            p2 = extra2 if extra2 is not None else np.zeros(mask.shape, dtype=np.float32)
        else:
            p = extra if extra is not None else np.zeros(mask.shape, dtype=np.float32)
            if p.shape != mask.shape:
                p = cv2.resize(p, (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        t_p = torch.from_numpy(np.ascontiguousarray(p)).float().unsqueeze(0)
        if self.prior_mode == "predicted2":
            if p2.shape != mask.shape:
                p2 = cv2.resize(p2, (mask.shape[1], mask.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            t_p2 = torch.from_numpy(np.ascontiguousarray(p2)).float().unsqueeze(0)
            rgp = torch.cat([t_rgb, t_p, t_p2], dim=0)
        else:
            rgp = torch.cat([t_rgb, t_p], dim=0)
        meta = {"patch_id": r["patch_id"], "altitude": r.get("altitude", ""),
                "snow": r.get("snow", ""), "source_image": stem,
                "fg_pixels": int(r.get("foreground_pixels", 0))}
        return rgp, t_mask, meta
