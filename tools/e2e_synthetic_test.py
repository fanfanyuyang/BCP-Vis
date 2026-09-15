#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

E2E = Path("/tmp/e2e")
RAW = E2E / "raw"
PROC = E2E / "processed"
N_IMG = 12
H, W = 1080, 1920

rng = np.random.RandomState(0)


def run(cmd, cwd="/root/autodl-tmp/BCP-Vis", check=True):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                       capture_output=True, text=True)
    tail = (r.stdout or "")[-1800:]
    if tail:
        print(tail)
    if r.returncode != 0:
        print("!!! FAILED rc=%d" % r.returncode)
        print((r.stderr or "")[-2500:])
        if check:
            sys.exit(1)
    return r


def build_fake_dataset():
    (RAW / "images").mkdir(parents=True, exist_ok=True)
    (RAW / "annotations").mkdir(parents=True, exist_ok=True)
    imgs, anns = [], []
    aid = 0
    for i in range(N_IMG):
        stem = ("DJI_%04d" % (100 + i)) if i % 3 != 0 else \
               ("DJI_%04d_frame_%02d" % (200 + i, i))
        p = RAW / "images" / (stem + ".jpg")
        img = rng.randint(60, 200, (H, W, 3), dtype=np.uint8)
        boxes = []
        for _ in range(rng.randint(8, 20)):
            bw = int(rng.randint(35, 90))
            bh = int(rng.randint(30, 70))
            x = int(rng.randint(0, W - bw - 1))
            y = int(rng.randint(0, H - bh - 1))
            cv2.rectangle(img, (x, y), (x + bw, y + bh),
                          (int(rng.randint(0, 255)), int(rng.randint(0, 255)),
                           int(rng.randint(0, 255))), -1)
            boxes.append((x, y, bw, bh))
        cv2.imwrite(str(p), img)

        iid = i + 1
        imgs.append({"id": iid, "file_name": stem + ".jpg",
                     "height": H, "width": W})
        for (x, y, bw, bh) in boxes:
            m = np.zeros((H, W), dtype=np.uint8, order="F")
            m[y:y + bh, x:x + bw] = 1
            import pycocotools.mask as maskUtils
            rle = maskUtils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("utf-8")
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": 1,
                         "segmentation": rle, "area": float(bw * bh),
                         "bbox": [x, y, bw, bh], "iscrowd": 0})
    coco = {"images": imgs, "annotations": anns,
            "categories": [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]}
    (RAW / "annotations" / "instances.json").write_text(
        json.dumps(coco), encoding="utf-8")
    print("fake dataset: %d images, %d instances" % (len(imgs), len(anns)))
    return imgs


def main():
    import shutil
    if E2E.exists():
        shutil.rmtree(E2E)
    build_fake_dataset()

    py = sys.executable
    print("=" * 72)
    print("STEP 1  audit_dataset")
    print("=" * 72)
    r = run([py, "scripts/audit_dataset.py", "--root", str(RAW),
             "--meta", "datasets/EVD4UAV/_official_metadata",
             "--out", "/tmp/e2e/reports", "--max-probe", "20"])
    aj = json.loads(Path("/tmp/e2e/reports/dataset_audit.json").read_text())
    print(">> 探测到的 json 类型:",
          [j["kind"] for j in aj.get("json_probes", [])])
    print(">> segmentation 编码:",
          [j.get("segmentation_encoding") for j in aj.get("json_probes", [])])

    print("=" * 72)
    print("STEP 2  build_binary_masks")
    print("=" * 72)
    run([py, "scripts/build_binary_masks.py", "--root", str(RAW),
         "--out", str(PROC), "--meta", "datasets/EVD4UAV/_official_metadata",
         "--n-vis", "4"])

    print("=" * 72)
    print("STEP 3  split_dataset")
    print("=" * 72)
    run([py, "scripts/split_dataset.py",
         "--manifest", str(PROC / "binary_masks" / "manifest.csv"),
         "--out", str(PROC), "--group-by", "sequence"])

    print("=" * 72)
    print("STEP 4  build_soft_prior_targets")
    print("=" * 72)
    run([py, "scripts/build_soft_prior_targets.py",
         "--mask-dir", str(PROC / "binary_masks"),
         "--out", str(PROC / "soft_priors"), "--sigma", "32", "--n-vis", "4"])

    print("=" * 72)
    print("STEP 5  make_patches")
    print("=" * 72)
    run([py, "scripts/make_patches.py", "--root", str(RAW),
         "--proc", str(PROC), "--patch", "768", "--overlap", "192",
         "--empty-keep-ratio", "0.5", "--n-vis", "2"])

    print("=" * 72)
    print("STEP 6  train_prior (1 epoch, limit 16)")
    print("=" * 72)
    run([py, "train/train_prior.py", "--config", "configs/prior_v3_litehr.yaml",
         "--name", "e2e_prior", "--epochs", "1", "--limit", "16",
         "--batch-size", "2", "--num-workers", "2",
         "--proc", str(PROC), "--raw", str(RAW), "--device", "cuda"])

    print("=" * 72)
    print("STEP 7  infer_prior")
    print("=" * 72)
    run([py, "scripts/infer_prior.py",
         "--ckpt", "runs/priornet/e2e_prior/best.pth", "--arch", "litehr",
         "--proc", str(PROC), "--raw", str(RAW),
         "--out-dir", str(PROC / "oof_priors"), "--tile", "1024"])

    print("=" * 72)
    print("STEP 8  train_final E5 (BCP-Vis, 1 epoch)")
    print("=" * 72)
    run([py, "train/train_final.py", "--config", "configs/bcp_vis.yaml",
         "--exp", "E5", "--name", "e2e_E5", "--epochs", "1", "--limit", "16",
         "--batch-size", "2", "--num-workers", "2",
         "--proc", str(PROC), "--raw", str(RAW),
         "--prior-dir", str(PROC / "oof_priors"), "--device", "cuda"])

    print("=" * 72)
    print("STEP 9  train_final E0 (RGB baseline, 1 epoch)")
    print("=" * 72)
    run([py, "train/train_final.py", "--config", "configs/rgb_baseline.yaml",
         "--exp", "E0", "--name", "e2e_E0", "--epochs", "1", "--limit", "16",
         "--batch-size", "2", "--num-workers", "2",
         "--proc", str(PROC), "--raw", str(RAW), "--device", "cuda"])

    print("\n" + "=" * 72)
    print("E2E ALL PASSED")
    print("=" * 72)

if __name__ == "__main__":
    main()
