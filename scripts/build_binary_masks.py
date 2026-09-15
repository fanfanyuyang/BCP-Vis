#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def _rle_fr_uncompressed(counts, h, w):
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        c = int(c)
        if val:
            mask[pos:pos + c] = 1
        pos += c
        val ^= 1
    return mask.reshape((h, w), order="F")


def _rle_fr_string(s, h, w):
    cnts = []
    m = 0
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
        m += 1
    if m % 2 == 1:
        cnts.append(0)
    return _rle_fr_uncompressed(cnts, h, w)


def decode_rle(seg, h, w) -> np.ndarray:
    if isinstance(seg, dict):
        counts = seg.get("counts")
        size = seg.get("size", [h, w])
        hh, ww = int(size[0]), int(size[1])
        if isinstance(counts, str):
            return _rle_fr_string(counts, hh, ww)
        return _rle_fr_uncompressed(counts, hh, ww)
    if isinstance(seg, str):
        return _rle_fr_string(seg, h, w)
    raise ValueError("unsupported RLE type: %s" % type(seg))


def decode_polygon(seg, h, w) -> np.ndarray:
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = seg if seg and isinstance(seg[0], list) else [seg]
    for poly in polys:
        if not poly or len(poly) < 6:
            continue
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [arr.astype(np.int32)], 1)
    return mask


class CocoAdapter:

    name = "coco"

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data
        self.images = {int(im["id"]): im for im in data.get("images", [])}
        self.anns_by_img = {}
        for a in data.get("annotations", []):
            self.anns_by_img.setdefault(int(a["image_id"]), []).append(a)
        kinds = {}
        for a in data.get("annotations", [])[:500]:
            s = a.get("segmentation")
            if isinstance(s, dict):
                kinds["rle_dict"] = kinds.get("rle_dict", 0) + 1
            elif isinstance(s, str):
                kinds["rle_str"] = kinds.get("rle_str", 0) + 1
            elif isinstance(s, list) and s:
                kinds["polygon"] = kinds.get("polygon", 0) + 1
            else:
                kinds["empty"] = kinds.get("empty", 0) + 1
        self.seg_kind = max(kinds, key=kinds.get) if kinds else "empty"
        self.kinds = kinds
        self.name = "coco_rle" if self.seg_kind.startswith("rle") else "coco_polygon"

    def items(self):
        for iid, im in self.images.items():
            stem = Path(im.get("file_name", str(iid))).stem
            h = int(im.get("height", 0))
            w = int(im.get("width", 0))
            yield stem, (h, w, self.anns_by_img.get(iid, []), im)

    def masks_for(self, h, w, anns):
        out = []
        for a in anns:
            seg = a.get("segmentation")
            if seg is None or (isinstance(seg, list) and len(seg) == 0):
                continue
            try:
                if isinstance(seg, dict) or isinstance(seg, str):
                    m = decode_rle(seg, h, w)
                else:
                    m = decode_polygon(seg, h, w)
            except Exception as e:
                print("   ! seg decode failed: %s" % str(e)[:100], file=sys.stderr)
                continue
            if m is not None and m.any():
                out.append(m.astype(bool))
        return out


class CvatPolygonAdapter:

    name = "cvat_polygon"

    def __init__(self, xml_path: Path):
        import xml.etree.ElementTree as ET
        self.xml_path = Path(xml_path)
        tree = ET.parse(str(self.xml_path))
        root = tree.getroot()
        self.images = root.findall("image")
        self.w = int(self.images[0].attrib.get("width", 0))
        self.h = int(self.images[0].attrib.get("height", 0))

    def items(self):
        for im in self.images:
            name = im.attrib.get("name", "")
            stem = Path(name).stem
            w = int(im.attrib.get("width", self.w))
            h = int(im.attrib.get("height", self.h))
            polys = []
            for p in im.findall("polygon"):
                pts = p.attrib.get("points", "")
                if not pts:
                    continue
                attrs = {a.attrib.get("name"): (a.text or "").strip()
                         for a in p.findall("attribute")}
                polys.append({"points": pts, "attrs": attrs})
            yield stem, (h, w, polys, im)

    def masks_for(self, h, w, refs):
        import cv2
        out = []
        for poly in refs:
            pts = []
            for pair in poly["points"].split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                xy = pair.split(",")
                if len(xy) < 2:
                    continue
                try:
                    pts.append([float(xy[0]), float(xy[1])])
                except ValueError:
                    continue
            if len(pts) < 3:
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
            cv2.fillPoly(m, [arr.astype(np.int32)], 1)
            if m.any():
                out.append(m.astype(bool))
        return out


class PngDirAdapter:

    name = "png_dir"

    def __init__(self, mask_dir: Path):
        self.mask_dir = Path(mask_dir)

    def items(self):
        for p in sorted(self.mask_dir.rglob("*")):
            if p.suffix in IMG_EXT:
                yield p.stem, (0, 0, [p], None)

    def masks_for(self, h, w, refs):
        import cv2
        out = []
        for p in refs:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            out.append(m > 0)
        return out


def _is_cvat(xml_path: Path) -> bool:
    try:
        head = xml_path.open("r", encoding="utf-8", errors="replace").read(4096)
    except Exception:
        return False
    return "<annotations" in head and "<image" in head


def build_adapter(root: Path, audit_json: Path | None, force: str | None):
    if force == "cvat":
        xs = sorted(root.rglob("*.xml"))
        xs = [p for p in xs if _is_cvat(p)]
        if not xs:
            sys.exit("ERROR: 找不到 CVAT xml")
        return CvatPolygonAdapter(max(xs, key=lambda p: p.stat().st_size))
    if force:
        if force.startswith("coco"):
            js = sorted(root.rglob("*.json"))
            js = [p for p in js if "category" not in p.name.lower()]
            if not js:
                sys.exit("ERROR: 找不到 COCO json")
            best, best_n = None, -1
            for p in js:
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    n = len(d.get("annotations", []))
                    if n > best_n:
                        best, best_n = p, n
                except Exception:
                    continue
            if best is None:
                sys.exit("ERROR: 无法解析任何 json 为 COCO")
            return CocoAdapter(best)
        if force == "png_dir":
            for d in root.rglob("*"):
                if d.is_dir() and any(k in d.name.lower() for k in ("mask", "seg", "instance")):
                    return PngDirAdapter(d)
            sys.exit("ERROR: 找不到 mask 目录")

    xs = [p for p in sorted(root.rglob("*.xml")) if _is_cvat(p)]
    if xs:
        return CvatPolygonAdapter(max(xs, key=lambda p: p.stat().st_size))

    for p in [q for q in sorted(root.rglob("*.json"))]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "annotations" in d and "images" in d:
            return CocoAdapter(p)
    for d in root.rglob("*"):
        if d.is_dir() and any(k in d.name.lower() for k in ("mask", "seg", "instance")):
            return PngDirAdapter(d)
    sys.exit("ERROR: 自动探测失败，请用 --adapter 指定（coco_rle / coco_polygon / png_dir）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/EVD4UAV/raw")
    ap.add_argument("--out", default="datasets/EVD4UAV_processed")
    ap.add_argument("--meta", default="datasets/EVD4UAV/_official_metadata")
    ap.add_argument("--audit", default="reports/dataset_audit.json")
    ap.add_argument("--adapter", default=None,
                    choices=["coco_rle", "coco_polygon", "png_dir"])
    ap.add_argument("--n-vis", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    mask_dir = out / "binary_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "visual_examples" / "binary_overlay"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("build_binary_masks  (原始数据只读)")
    print("root:", root)
    print("=" * 72)

    adapter = build_adapter(root, Path(args.audit), args.adapter)
    print("adapter:", adapter.name)
    if isinstance(adapter, CocoAdapter):
        print("  json :", adapter.json_path.name)
        print("  seg  :", adapter.seg_kind, adapter.kinds)
        print("  images=%d  annotations=%d"
              % (len(adapter.images), len(adapter.data.get("annotations", []))))

    meta = Path(args.meta).resolve()
    alt_map, snow_set = {}, set()
    if (meta / "altitude_mapping.txt").exists():
        for line in (meta / "altitude_mapping.txt").read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 2:
                alt_map[p[0]] = p[1]
    if (meta / "snow_images.txt").exists():
        snow_set = {l.strip() for l in
                    (meta / "snow_images.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    print("official meta: altitude=%d  snow=%d" % (len(alt_map), len(snow_set)))

    rows = []
    n_no_instance = 0
    n_img = 0
    fg_ratios = []

    for stem, (h, w, anns, imrec) in adapter.items():
        n_img += 1
        if h == 0 or w == 0:
            cand = None
            for ext in IMG_EXT:
                p = root / (stem + ext)
                if p.exists():
                    cand = p
                    break
            if cand is None:
                hits = list(root.rglob(stem + ".*"))
                cand = hits[0] if hits else None
            if cand is None:
                rows.append({"image_id": stem, "status": "NO_IMAGE"})
                n_no_instance += 1
                continue
            from PIL import Image
            with Image.open(cand) as im:
                w, h = im.size

        masks = adapter.masks_for(h, w, anns)
        if not masks:
            n_no_instance += 1
            rows.append({"image_id": stem, "status": "NO_INSTANCE_MASK"})
            continue

        binary = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            if m.shape != (h, w):
                import cv2
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            binary |= m.astype(np.uint8)

        fg = int(binary.sum())
        ratio = fg / float(h * w)
        fg_ratios.append(ratio)

        mp = mask_dir / (stem + ".png")
        import cv2
        cv2.imwrite(str(mp), binary * 255)

        rows.append({
            "image_id": stem,
            "status": "OK",
            "height": h,
            "width": w,
            "n_instances": len(masks),
            "fg_pixels": fg,
            "fg_ratio": round(ratio, 8),
            "altitude": alt_map.get(stem, ""),
            "snow": int((stem + ".jpg") in snow_set),
            "mask_path": str(mp.relative_to(out)),
        })

        if n_img % 500 == 0:
            print("  processed %d ..." % n_img)

    manifest = mask_dir / "manifest.csv"
    fields = ["image_id", "status", "height", "width", "n_instances",
              "fg_pixels", "fg_ratio", "altitude", "snow", "mask_path"]
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})

    ok = [r for r in rows if r.get("status") == "OK"]
    print("=" * 72)
    print("图像条目        :", n_img)
    print("成功生成 mask   :", len(ok))
    print("无实例 mask     :", n_no_instance)
    print("mask 文件数     :", len(list(mask_dir.glob('*.png'))))
    if fg_ratios:
        arr = np.array(fg_ratios)
        print("前景占比  min=%.6f  mean=%.6f  max=%.6f" % (arr.min(), arr.mean(), arr.max()))
    print("manifest        :", manifest)

    ok_vis = [r for r in ok if r.get("mask_path")]
    random.seed(args.seed)
    picks = random.sample(ok_vis, min(args.n_vis, len(ok_vis)))
    import cv2
    for r in picks:
        stem = r["image_id"]
        img_p = None
        for p in root.rglob(stem + ".*"):
            if p.suffix in IMG_EXT:
                img_p = p
                break
        if img_p is None:
            continue
        img = cv2.imread(str(img_p))
        msk = cv2.imread(str(out / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = img.copy()
        ov[msk > 0] = (0.45 * ov[msk > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        canvas = np.concatenate([img, cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR), ov], axis=1)
        cv2.imwrite(str(vis_dir / (stem + "_rgb_mask_overlay.jpg")), canvas)
    print("可视化          :", vis_dir, "(%d 张)" % len(picks))
    print("=" * 72)

if __name__ == "__main__":
    main()
