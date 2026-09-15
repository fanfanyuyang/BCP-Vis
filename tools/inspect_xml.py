#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else
         "datasets/EVD4UAV/raw/EVD4UAV/mask_attribute.xml")
print("file:", p, "%.1f MB" % (p.stat().st_size / 1e6))

tree = ET.parse(str(p))
root = tree.getroot()
print("root tag:", root.tag)
print("root children:", Counter(c.tag for c in root))

images = root.findall("image")
print("\n<image> 元素数:", len(images))
im0 = images[0]
print("image 属性:", dict(im0.attrib))
print("image 子元素:", Counter(c.tag for c in im0))

polys = im0.findall("polygon")
print("\n首图 polygon 数:", len(polys))
if polys:
    print("polygon 属性:", dict(polys[0].attrib))
    print("polygon 子元素:", Counter(c.tag for c in polys[0]))
    for c in polys[0]:
        print("   attr:", c.attrib.get("name"), "=", (c.text or "").strip())

n_poly = 0
names = []
wh = Counter()
labels = Counter()
for im in images:
    names.append(im.attrib.get("name", ""))
    wh[(im.attrib.get("width"), im.attrib.get("height"))] += 1
    ps = im.findall("polygon")
    n_poly += len(ps)
    for x in ps:
        labels[x.attrib.get("label")] += 1

print("\n总 polygon(实例) 数:", n_poly)
print("图像名唯一数:", len(set(names)), " 总数:", len(names))
print("分辨率分布:", dict(wh.most_common(5)))
print("label 分布:", dict(labels.most_common()))
print("图像名示例:", names[:5])
print("带 _frame_ 的:", sum("_frame_" in n for n in names))
sys.exit(0)
