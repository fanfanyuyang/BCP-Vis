#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from datasets_py.patch_dataset import PatchDataset
from models import build_final_cnn, count_params, maybe_flops
from train.common import (binary_metrics_from_logits, load_cfg,
                          set_seed, setup_perf)

FMT = torch.channels_last

TINY = 32 * 32
SMALL = 96 * 96


def snow_group(v):
    s = str(v).strip()
    return "snow" if s in ("1", "1.0", "True", "true") else "clean"


def size_group(fg_px):
    if fg_px <= 0:
        return "empty"
    if fg_px < TINY:
        return "tiny"
    if fg_px < SMALL:
        return "small"
    return "medium"

@torch.no_grad()


def evaluate(model, loader, dev, thr=0.5, amp=True):
    model.eval()
    agg = defaultdict(list)
    for batch in loader:
        x, y = batch[0].to(dev), batch[1].to(dev)
        if dev == "cuda":
            x = x.contiguous(memory_format=FMT)
        with torch.autocast(device_type="cuda", enabled=amp and dev == "cuda"):
            out = model(x, batch[2].get("altitude") if (len(batch) > 2 and isinstance(batch[2], dict)) else None)
        m = binary_metrics_from_logits(out, y, thr=thr)
        for k, v in m.items():
            agg[k].append(v)
    return {k: float(sum(v) / len(v)) for k, v in agg.items()}


def aggregate_by(matches, keyfn):
    groups = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0})
    for meta, tp, fp, fn in matches:
        g = groups[keyfn(meta)]
        g["tp"] += tp
        g["fp"] += fp
        g["fn"] += fn
        g["n"] += 1
    out = []
    eps = 1e-7
    for k, g in sorted(groups.items()):
        tp, fp, fn = g["tp"], g["fp"], g["fn"]
        out.append({
            "group": k, "n_patches": g["n"],
            "fp_px": int(fp), "fn_px": int(fn), "tp_px": int(tp),
            "fp_per_patch": round(fp / max(1, g["n"]), 1),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "f1": 2 * tp / (2 * tp + fp + fn + eps),
        })
    return out

@torch.no_grad()


def collect_per_patch(model, loader, dev, thr=0.5, amp=True):
    model.eval()
    matches = []
    for batch in loader:
        x, y = batch[0].to(dev), batch[1].to(dev)
        metas = batch[2]
        if dev == "cuda":
            x = x.contiguous(memory_format=FMT)
        with torch.autocast(device_type="cuda", enabled=amp and dev == "cuda"):
            out = model(x, batch[2].get("altitude") if (len(batch) > 2 and isinstance(batch[2], dict)) else None)
        p = (torch.sigmoid(out) > thr)
        t = y > 0.5
        for i in range(p.shape[0]):
            tp = float((p[i] & t[i]).sum())
            fp = float((p[i] & ~t[i]).sum())
            fn = float((~p[i] & t[i]).sum())
            matches.append(({k: _pick(v, i) for k, v in metas.items()},
                            tp, fp, fn))
    return matches


def _pick(v, i):
    if torch.is_tensor(v):
        v = v[i]
        return v.item() if v.dim() == 0 else v
    if isinstance(v, (list, tuple)):
        return v[i]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="E5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--proc", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--prior-dir", default=None)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.proc:
        cfg["proc"] = args.proc
    if args.raw:
        cfg["raw"] = args.raw
    proc, raw = cfg.get("proc"), cfg.get("raw")

    modes = {"E0": ("none", 3), "E1": ("zero", 4), "E2": ("random", 4),
             "E3": ("predicted", 4), "E4": ("predicted", 4),
             "E5": ("predicted", 4), "E6": ("predicted", 4),
             "E7": ("oracle", 4), "E8": ("predicted", 4),
             "E9": ("predicted", 4), "E10": ("predicted", 4),
             "E11": ("predicted", 4), "E12": ("predicted", 4),
             "E13": ("predicted", 4), "E14": ("predicted", 4),
             "E15": ("predicted", 4), "E13b": ("predicted", 4), "E16": ("predicted", 4), "E17": ("predicted", 4), "E18": ("predicted", 4),
             "E19": ("predicted", 4), "E20": ("predicted2", 5), "E21": ("predicted", 4), "E22": ("predicted2", 5),
             "E23": ("predicted", 4), "E24": ("predicted", 4), "E25": ("predicted", 4), "E26": ("predicted", 4), "E28": ("predicted2", 5), "E29": ("predicted", 4), "E30": ("predicted2", 5), "E31": ("predicted", 4), "E32": ("predicted", 4), "E33": ("predicted", 4)}
    prior_mode, in_ch = modes[args.exp]

    set_seed(cfg.get("seed", 42))
    dev = args.device if torch.cuda.is_available() else "cpu"
    setup_perf()
    model = build_final_cnn(cfg.get("model", {"in_channels": in_ch})).to(dev)
    if dev == "cuda":
        model = model.to(memory_format=FMT)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    ds = PatchDataset(proc=proc, raw=raw, split=args.split, task="final",
                      prior_mode=prior_mode,
                      prior_dir=args.prior_dir or cfg.get("prior_dir"),
                      prior_dir2=cfg.get("prior_dir2"),
                      augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    n_warm, n_test = 3, 20
    times = []
    for i, batch in enumerate(dl):
        if i >= n_warm + n_test:
            break
        x = batch[0].to(dev)
        if dev == "cuda":
            x = x.contiguous(memory_format=FMT)
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= n_warm:
            times.append(time.time() - t0)
    fps = (args.batch_size / (sum(times) / len(times))) if times else 0.0

    overall = evaluate(model, dl, dev, thr=args.thr)
    matches = collect_per_patch(model, dl, dev, thr=args.thr)

    _i, _d = [], []
    for _m, _tp, _fp, _fn in matches:
        _i.append(_tp / (_tp + _fp + _fn + 1e-7))
        _d.append(2 * _tp / (2 * _tp + _fp + _fn + 1e-7))
    macro_iou = sum(_i) / len(_i) if _i else 0.0
    macro_dice = sum(_d) / len(_d) if _d else 0.0

    rep = Path("reports")
    rep.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.exp

    row = {
        "experiment": tag, "split": args.split,
        "precision": round(overall["precision"], 6),
        "recall": round(overall["recall"], 6),
        "iou": round(overall["iou"], 6),
        "dice": round(overall["dice"], 6),
        "f1": round(overall["f1"], 6),
        "iou_macro": round(macro_iou, 6),
        "dice_macro": round(macro_dice, 6),
        "n_patches": len(matches),
        "params": count_params(model),
        "fps": round(fps, 2),
    }
    fl = maybe_flops(model, (1, in_ch, 768, 768), dev)
    row["flops"] = int(fl) if fl else ""
    fm = rep / "final_metrics.csv"
    new = not fm.exists()
    with open(fm, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    by_alt = aggregate_by(matches, lambda m: (m.get("altitude") or "unknown"))
    by_size = aggregate_by(matches, lambda m: size_group(int(m.get("fg_pixels", 0) or 0)))
    by_snow = aggregate_by(matches, lambda m: snow_group(m.get("snow", "0")))
    by_size_alt = aggregate_by(
        matches,
        lambda m: "%s@%s" % (size_group(int(m.get("fg_pixels", 0) or 0)),
                             m.get("altitude") or "unknown"))

    by_size_snow = aggregate_by(
        matches,
        lambda m: "%s@%s" % (size_group(int(m.get("fg_pixels", 0) or 0)),
                             snow_group(m.get("snow", "0"))))

    for name, rows_ in (("by_altitude", by_alt), ("by_size", by_size),
                        ("by_snow", by_snow), ("by_size_altitude", by_size_alt),
                        ("by_size_snow", by_size_snow)):
        if not rows_:
            continue
        p = rep / (name + ".csv")
        fields = ["experiment", "split"] + list(rows_[0].keys())
        need_rewrite = True
        if p.exists():
            try:
                old_fields = list(csv.DictReader(p.open(encoding="utf-8")).fieldnames or [])
                if old_fields == fields:
                    need_rewrite = False
            except Exception:
                need_rewrite = True
        if need_rewrite:
            keep = []
            if p.exists():
                for r in csv.DictReader(p.open(encoding="utf-8")):
                    if r.get("experiment") not in (tag,):
                        keep.append(r)
            if keep and list(keep[0].keys()) != fields:
                keep = []
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in keep:
                    w.writerow(r)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for r in rows_:
                w.writerow({"experiment": tag, "split": args.split, **r})

    print(json.dumps(row, indent=2))
    print("by_altitude:", json.dumps(by_alt, indent=2, ensure_ascii=False))
    print("by_size    :", json.dumps(by_size, indent=2, ensure_ascii=False))
    print("写出 ->", fm)

if __name__ == "__main__":
    main()
