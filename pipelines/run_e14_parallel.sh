#!/bin/bash
# run_e14_parallel.sh —— 利用空闲资源并行跑 E14（类别感知背景专家消融）
#
# 与主线 e11→e12→e13 完全独立（E14 是链上最后一个实验），
# 故可现在并行启动以占满空闲的 GPU 显存 / CPU / 内存。
# 不等待 e13_queue.done；结果写到标准路径，供链上 run_e14_queue.sh 幂等跳过。
cd /root/autodl-tmp/BCP-Vis
trap 'rm -f logs/e14_parallel.active' EXIT
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e14_parallel.log
EXPECT=9371
CFG=configs/prior_v5_cat.yaml
CKPT=runs/priornet/prior_v5_cat/best.pth
OUTDIR="$PROC/priors_cat"
echo "[$(date)] E14 并行启动（利用空闲资源，与 e11/e12/e13 并行）" >> "$LOG"
touch logs/e14_parallel.active

# 步骤1：类别标签（已预建则跳过）
if [ ! -f "$PROC/image_classes.csv" ]; then
  python scripts/build_class_labels.py --proc "$PROC" --xml "$RAW/mask_attribute.xml" >> "$LOG" 2>&1 || echo "[FAIL] build_class_labels" >> "$LOG"
fi

# 步骤2：训练类别感知背景专家 PriorNet
if [ ! -f "$CKPT" ]; then
  python train/train_prior.py --config "$CFG" --device cuda --num-workers 8 --batch-size 16 >> "$LOG" 2>&1 || echo "[FAIL] E14 prior train" >> "$LOG"
fi
if [ ! -f "$CKPT" ]; then echo "[ABORT] E14 ckpt 缺失" >> "$LOG"; exit 1; fi

mkdir -p "$OUTDIR"
# 步骤3：推理 E14 先验（mul，与 E8 同口径）
if [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -lt "$EXPECT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cat_bg --model-config "$CFG" \
    --prior-mode mul --proc "$PROC" --raw "$RAW" --out-dir "$OUTDIR" --tile 1024 --overlap 192 \
    >> "$LOG" 2>&1 || echo "[FAIL] infer cat" >> "$LOG"
fi
N=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "cat prior files: $N / $EXPECT" >> "$LOG"
if [ "$N" -lt "$EXPECT" ]; then echo "[ABORT] cat prior $N < $EXPECT" >> "$LOG"; exit 1; fi

# 步骤4：训练 E14 Final CNN
if [ ! -f runs/bcp_vis/E14_cat/best.pth ]; then
  python train/train_final.py --config configs/bcp_vis.yaml --exp E14 --name E14_cat \
    --batch-size 16 --num-workers 8 --proc "$PROC" --raw "$RAW" --prior-dir "$OUTDIR" \
    >> "$LOG" 2>&1 || echo "[FAIL] E14 train" >> "$LOG"
fi
if [ -f runs/bcp_vis/E14_cat/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E14_cat/best.pth --config configs/bcp_vis.yaml \
    --exp E14 --tag E14 --proc "$PROC" --raw "$RAW" --prior-dir "$OUTDIR" --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E14 eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo "[$(date)] E14 并行完成" >> "$LOG"
fi
echo DONE > logs/e14_parallel.done
