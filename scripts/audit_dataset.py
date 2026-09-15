#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import binascii
import collections
import json
import os
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
JSON_EXT = {".json"}
TXT_EXT = {".txt"}
XML_EXT = {".xml"}


def hr(title=""):
    return ("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72) if title else ("=" * 72)


def sniff_magic(path: Path, n=12) -> str:
    try:
        with open(path, "rb") as f:
            head = f.read(n)
    except Exception as e:
        return f"UNREADABLE({e})"
    if head[:2] == b"\xff\xd8":
        return "JPEG"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:2] == b"BM":
        return "BMP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if head[:5] == b"<?xml":
        return "XML"
    if head[:1] in (b"{", b"["):
        return "JSON?"
    return "UNKNOWN " + binascii.hexlify(head[:6]).decode()


def fast_image_size(path: Path):
    try:
        from PIL import Image
        from PIL import UnidentifiedImageError
    except Exception:
        return None, "PIL_NOT_AVAILABLE"
    try:
        with Image.open(path) as im:
            return im.size, None
    except Exception as e:
        return None, type(e).__name__ + ":" + str(e)[:80]


def scan_tree(root: Path, max_depth=6):
    ext_counter = collections.Counter()
    dir_counter = collections.Counter()
    samples_by_ext = collections.defaultdict(list)
    total_bytes = 0
    n_files = 0
    empty_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        dir_counter[rel] += len(filenames)
        for fn in filenames:
            n_files += 1
            p = Path(dirpath) / fn
            ext = p.suffix
            ext_counter[ext] += 1
            try:
                sz = p.stat().st_size
                total_bytes += sz
                if sz == 0:
                    empty_files.append(str(p.relative_to(root)))
            except Exception:
                sz = -1
            if len(samples_by_ext[ext]) < 5:
                samples_by_ext[ext].append(str(p.relative_to(root)))

    return {
        "n_files": n_files,
        "total_bytes": total_bytes,
        "ext_counter": dict(ext_counter.most_common()),
        "dir_counter": dict(sorted(dir_counter.items(), key=lambda kv: -kv[1])[:40]),
        "samples_by_ext": {k: v for k, v in samples_by_ext.items()},
        "empty_files": empty_files[:50],
        "n_empty_files": len(empty_files),
    }


def probe_json(path: Path):
    info = {"path": str(path), "size_bytes": path.stat().st_size, "kind": "unknown"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        info["kind"] = "unparsable"
        info["error"] = str(e)[:200]
        return info

    if isinstance(data, dict):
        keys = list(data.keys())
        info["top_level_keys"] = keys[:20]
        info["n_top_level_keys"] = len(keys)
        has = {k: (k in data) for k in ("images", "annotations", "categories")}
        info["coco_markers"] = has
        if all(has.values()):
            info["kind"] = "coco"
            try:
                info["n_images"] = len(data["images"])
                info["n_annotations"] = len(data["annotations"])
                info["n_categories"] = len(data["categories"])
                im0 = data["images"][0]
                info["image_record_keys"] = sorted(im0.keys())
                info["image_record_sample"] = {
                    k: (str(im0[k])[:120]) for k in list(im0.keys())[:12]
                }
                an0 = data["annotations"][0]
                info["ann_record_keys"] = sorted(an0.keys())
                info["ann_record_sample"] = {
                    k: (str(an0[k])[:160]) for k in list(an0.keys())[:12]
                }
                segs = [a.get("segmentation") for a in data["annotations"][:200]
                        if isinstance(a, dict) and "segmentation" in a]
                kinds = collections.Counter()
                for s in segs:
                    if isinstance(s, dict):
                        kinds["RLE(dict, counts+size)" if "counts" in s else "dict?"] += 1
                    elif isinstance(s, str):
                        kinds["RLE(str counts)"] += 1
                    elif isinstance(s, list):
                        if s and isinstance(s[0], list):
                            kinds["polygon(list of lists)"] += 1
                        else:
                            kinds["polygon(flat list)"] += 1
                    elif s is None:
                        kinds["null"] += 1
                    else:
                        kinds[type(s).__name__] += 1
                info["segmentation_encoding"] = dict(kinds)
                info["categories_sample"] = [
                    {k: str(c.get(k))[:60] for k in ("id", "name", "supercategory")}
                    for c in data["categories"][:20]
                ]
            except Exception as e:
                info["coco_parse_error"] = str(e)[:200]
        elif "images" in data or "annotations" in data:
            info["kind"] = "coco_like_partial"
        else:
            info["kind"] = "generic_dict"
    elif isinstance(data, list):
        info["kind"] = "list"
        info["n_items"] = len(data)
        if data and isinstance(data[0], dict):
            info["item_keys"] = sorted(data[0].keys())[:20]
    return info


def probe_txt(path: Path, n_lines=5):
    info = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [next(f, "") for _ in range(n_lines)]
    except Exception as e:
        info["kind"] = "unreadable"
        info["error"] = str(e)[:120]
        return info
    info["sample_lines"] = [l.rstrip("\n")[:120] for l in lines if l]
    first = (lines[0] or "").strip()
    toks = first.split()
    numeric = False
    if toks:
        try:
            [float(t) for t in toks[:5]]
            numeric = len(toks) >= 5
        except Exception:
            numeric = False
    if numeric:
        info["kind"] = "yolo_like_or_numeric"
        info["n_tokens_first_line"] = len(toks)
    elif first.lower().endswith((".jpg", ".png", ".jpeg")):
        info["kind"] = "file_list"
    else:
        info["kind"] = "text_other"
    return info


def scan_images(root: Path, img_paths, max_probe, verify_sample):
    sizes = collections.Counter()
    corrupt = []
    probed = 0
    bad_magic = []
    for p in img_paths:
        if probed >= max_probe:
            break
        probed += 1
        sz, err = fast_image_size(p)
        if err is not None:
            corrupt.append({"path": str(p.relative_to(root)), "error": err})
        else:
            sizes["%dx%d" % sz] += 1
    for p in img_paths[:: max(1, len(img_paths) // 400)]:
        m = sniff_magic(p)
        if m not in ("JPEG", "PNG", "BMP", "TIFF"):
            bad_magic.append({"path": str(p.relative_to(root)), "magic": m})
    return {
        "probed": probed,
        "size_histogram": dict(sizes.most_common(20)),
        "corrupt_sample": corrupt[:50],
        "n_corrupt_in_sample": len(corrupt),
        "bad_magic_sample": bad_magic[:50],
        "n_bad_magic_in_sample": len(bad_magic),
        "verify_sample": verify_sample,
    }


def load_official_meta(meta_dir: Path):
    out = {"dir": str(meta_dir), "files": {}}
    if not meta_dir.exists():
        out["status"] = "MISSING"
        return out
    am = meta_dir / "altitude_mapping.txt"
    if am.exists():
        m = {}
        bad = 0
        for line in am.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
            elif line.strip():
                bad += 1
        out["files"]["altitude_mapping"] = {
            "path": str(am),
            "n_entries": len(m),
            "n_malformed_lines": bad,
            "altitude_distribution": dict(collections.Counter(m.values()).most_common()),
            "key_format": "image stem without extension  ->  altitude",
            "sample": dict(list(m.items())[:5]),
        }
        out["_altitude_map"] = m
    for name, key in (("snow_images.txt", "snow"),
                      ("non_snow_images.txt", "non_snow")):
        f = meta_dir / name
        if f.exists():
            items = [l.strip() for l in
                     f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
            out["files"][name] = {
                "path": str(f),
                "n_entries": len(items),
                "n_unique": len(set(items)),
                "sample": items[:5],
            }
            out["_" + key] = set(items)
    out["status"] = "OK"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/EVD4UAV/raw")
    ap.add_argument("--meta", default="datasets/EVD4UAV/_official_metadata")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--max-probe", type=int, default=1500,
                    help="最多解码多少张图取尺寸")
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        sys.exit("ERROR: root 不存在: %s" % root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "root": str(root),
        "audit_mode": "READ_ONLY",
        "note": "本脚本不修改任何原始数据",
    }

    print(hr("EVD4UAV 数据集审计（只读）"))
    print("root:", root)

    print(hr("1. 文件树扫描"))
    tree = scan_tree(root, max_depth=args.max_depth)
    report["tree"] = tree
    print("文件总数:", tree["n_files"], " 总体积: %.2f GB" % (tree["total_bytes"] / 1e9))
    print("扩展名分布:", json.dumps(tree["ext_counter"], ensure_ascii=False))
    print("文件数最多的目录(top10):")
    for d, c in list(tree["dir_counter"].items())[:10]:
        print("   %-55s %d" % (d, c))
    if tree["n_empty_files"]:
        print("!! 空文件 %d 个，示例: %s" % (tree["n_empty_files"], tree["empty_files"][:5]))

    print(hr("2. 图像定位"))
    img_paths = [p for p in root.rglob("*") if p.suffix in IMG_EXT]
    report["n_images"] = len(img_paths)
    print("图像文件数:", len(img_paths))
    if img_paths:
        exts = collections.Counter(p.suffix.lower() for p in img_paths)
        print("图像扩展名:", dict(exts))
        img_dirs = collections.Counter(str(p.parent.relative_to(root)) for p in img_paths)
        print("图像所在目录(top10):")
        for d, c in img_dirs.most_common(10):
            print("   %-55s %d" % (d, c))
        report["image_dirs"] = dict(img_dirs.most_common(20))
        istat = scan_images(root, img_paths, args.max_probe, True)
        report["image_stats"] = istat
        print("尺寸分布(采样 %d):" % istat["probed"], istat["size_histogram"])
        print("损坏(采样内):", istat["n_corrupt_in_sample"])
        print("魔数异常(采样内):", istat["n_bad_magic_in_sample"])

    print(hr("3. 标注文件探测"))
    jsons = [p for p in root.rglob("*.json")]
    report["n_json"] = len(jsons)
    print("json 文件数:", len(jsons))
    json_probes = []
    for p in jsons[:10]:
        pr = probe_json(p)
        pr["path"] = str(p.relative_to(root))
        json_probes.append(pr)
        print(" -", pr["path"], "->", pr["kind"],
              ("images=%s ann=%s cat=%s" % (pr.get("n_images"), pr.get("n_annotations"),
                                            pr.get("n_categories")))
              if pr["kind"] == "coco" else "")
        if pr["kind"] == "coco":
            print("     image keys :", pr.get("image_record_keys"))
            print("     ann keys   :", pr.get("ann_record_keys"))
            print("     seg 编码   :", pr.get("segmentation_encoding"))
            print("     categories :", pr.get("categories_sample"))
    report["json_probes"] = json_probes

    txts = [p for p in root.rglob("*.txt")]
    txt_probes = []
    for p in txts[:15]:
        pr = probe_txt(p)
        pr["path"] = str(p.relative_to(root))
        txt_probes.append(pr)
    report["n_txt"] = len(txts)
    report["txt_probes"] = txt_probes
    print("txt 文件数:", len(txts))
    kinds = collections.Counter(pr["kind"] for pr in txt_probes)
    print("txt 类型(采样):", dict(kinds))

    xmls = [p for p in root.rglob("*.xml")]
    report["n_xml"] = len(xmls)
    print("xml 文件数:", len(xmls))

    print(hr("4. Mask 线索"))
    mask_hint = {}
    for d, c in tree["dir_counter"].items():
        low = d.lower()
        if any(k in low for k in ("mask", "seg", "instance", "instances", "annotation", "label")):
            mask_hint[d] = c
    report["mask_dir_hints"] = mask_hint
    print("疑似 mask/标注 目录:", json.dumps(mask_hint, ensure_ascii=False, indent=2))

    print(hr("5. 官方 metadata 关联"))
    meta = load_official_meta(Path(args.meta).resolve())
    report["official_metadata"] = {k: v for k, v in meta.items() if not k.startswith("_")}
    print(json.dumps(report["official_metadata"], ensure_ascii=False, indent=2)[:2000])

    if img_paths and "_altitude_map" in meta:
        amap = meta["_altitude_map"]
        matched = sum(1 for p in img_paths if p.stem in amap)
        report["altitude_join"] = {
            "join_rule": "image stem == altitude_mapping key",
            "n_images": len(img_paths),
            "n_matched": matched,
            "n_unmatched": len(img_paths) - matched,
            "unmatched_sample": [p.name for p in img_paths if p.stem not in amap][:10],
        }
        print("高度关联: %d/%d 命中" % (matched, len(img_paths)))
        print(json.dumps(report["altitude_join"], ensure_ascii=False, indent=2))

    jpath = out_dir / "dataset_audit.json"
    tpath = out_dir / "dataset_audit.txt"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(hr("EVD4UAV 数据集审计报告"))
    lines.append("root: %s" % root)
    lines.append("模式: 只读，未修改任何原始文件")
    lines.append(hr("1. 文件树"))
    lines.append("文件总数: %d    总体积: %.2f GB" % (tree["n_files"], tree["total_bytes"] / 1e9))
    lines.append("扩展名分布: %s" % json.dumps(tree["ext_counter"], ensure_ascii=False))
    lines.append("空文件数: %d" % tree["n_empty_files"])
    lines.append(hr("2. 图像"))
    lines.append("图像数: %d" % len(img_paths))
    for d, c in list(report.get("image_dirs", {}).items())[:10]:
        lines.append("  %-55s %d" % (d, c))
    if "image_stats" in report:
        lines.append("尺寸分布(采样%d): %s" % (report["image_stats"]["probed"],
                                              report["image_stats"]["size_histogram"]))
        lines.append("损坏(采样内): %d" % report["image_stats"]["n_corrupt_in_sample"])
    lines.append(hr("3. 标注"))
    lines.append("json: %d   txt: %d   xml: %d" % (len(jsons), len(txts), len(xmls)))
    for pr in json_probes:
        lines.append("  [%s] %s  size=%dB" % (pr["kind"], pr["path"], pr["size_bytes"]))
        if pr["kind"] == "coco":
            lines.append("      images=%d annotations=%d categories=%d"
                         % (pr["n_images"], pr["n_annotations"], pr["n_categories"]))
            lines.append("      image_keys=%s" % pr.get("image_record_keys"))
            lines.append("      ann_keys=%s" % pr.get("ann_record_keys"))
            lines.append("      segmentation_encoding=%s" % pr.get("segmentation_encoding"))
    lines.append(hr("4. Mask 线索"))
    lines.append(json.dumps(mask_hint, ensure_ascii=False, indent=2))
    lines.append(hr("5. 官方 metadata"))
    lines.append(json.dumps(report["official_metadata"], ensure_ascii=False, indent=2))
    if "altitude_join" in report:
        lines.append(hr("6. 高度关联"))
        lines.append(json.dumps(report["altitude_join"], ensure_ascii=False, indent=2))
    lines.append(hr("结论"))
    lines.append("审计完成。下一步：依据本报告中的 annotation schema 编写 build_binary_masks.py，")
    lines.append("在此之前不得写死任何解析逻辑。")
    tpath.write_text("\n".join(lines), encoding="utf-8")

    print(hr())
    print("已写出:")
    print("  ", jpath)
    print("  ", tpath)
    print(hr())

if __name__ == "__main__":
    main()
