#!/usr/bin/env bash
# ============================================================
# BCP-Vis 阶段 02~06 一键流水线（数据到达后执行）
#   02 数据集审计        -> reports/dataset_audit.{json,txt}
#   03 Binary GT         -> EVD4UAV_processed/binary_masks/
#   04 原图级 split      -> EVD4UAV_processed/split.csv
#   04b Soft BCP GT      -> EVD4UAV_processed/soft_priors/
#   06 Patch 索引        -> EVD4UAV_processed/patch_manifest.csv
#
# 严格只读原始数据。任何一步失败立即退出。
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/BCP-Vis
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw
META=datasets/EVD4UAV/_official_metadata

echo "============================================================"
echo "STEP 02  数据集审计"
echo "============================================================"
python scripts/audit_dataset.py \
    --root "$RAW" \
    --meta "$META" \
    --out reports \
    --max-probe 3000

echo
echo "============================================================"
echo "STEP 03  Binary Foreground GT"
echo "============================================================"
python scripts/build_binary_masks.py \
    --root "$RAW" \
    --out "$PROC" \
    --meta "$META" \
    --n-vis 20

echo
echo "============================================================"
echo "STEP 05  原图级 Train/Val Split（按序列分组，防泄露）"
echo "============================================================"
python scripts/split_dataset.py \
    --manifest "$PROC/binary_masks/manifest.csv" \
    --out "$PROC" \
    --group-by sequence \
    --train 0.8 --val 0.1

echo
echo "============================================================"
echo "STEP 04  Soft BCP GT（sigma=32，先用 20 张做候选对比）"
echo "============================================================"
python scripts/build_soft_prior_targets.py \
    --config configs/soft_prior.yaml \
    --mask-dir "$PROC/binary_masks" \
    --out "$PROC/soft_priors" \
    --n-vis 20

echo
echo "============================================================"
echo "STEP 06  Patch 索引"
echo "============================================================"
python scripts/make_patches.py \
    --root "$RAW" \
    --proc "$PROC" \
    --patch 768 --overlap 192 \
    --empty-keep-ratio 0.15 \
    --n-vis 12

python tools/check_patch_stats.py "$PROC/patch_manifest.csv"

echo
echo "============================================================"
echo "阶段 02~06 完成。请检查："
echo "  reports/dataset_audit.txt"
echo "  $PROC/binary_masks/manifest.csv"
echo "  $PROC/split.json"
echo "  $PROC/patch_manifest.csv"
echo "下一步：PriorNet V1 smoke test"
echo "============================================================"
