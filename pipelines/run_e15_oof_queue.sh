#!/bin/bash
# E15 队列：V8 级联 + UAP 的 OOF 严谨性重跑
# 等 E13(e13_queue.done) 释放 GPU 后，零空闲接手：
#   Step A: 4 折 cascade PriorNet OOF 先验（uap 组合，仅 train）
#   Step B: val/test 用 E12 全量 cascade ckpt 推理（uap，本就 OOF）
#   Step C: 用 E13 ckpt 在 OOF uap 先验上评测 -> E15
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e15.lock
flock -n 200 || { echo "E15 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e15_queue.log
echo "[$(date)] E15 队列启动，等 logs/e13_queue.done" >> "$LOG"
while [ ! -f logs/e13_queue.done ]; do sleep 120; done
echo "[$(date)] E13 完成，开始 E15 (V8级联 + UAP 的 OOF 严谨性重跑)" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
OUTDIR=datasets/EVD4UAV_processed/oof_priors_cascade_uap
E12CKPT=runs/priornet/prior_v8_cascade/best.pth

# Step A: 4 折 cascade PriorNet OOF 先验（uap 组合）
python scripts/generate_oof_prior.py \
  --base-config configs/prior_v8_cascade.yaml \
  --prior-cfg-name cascade \
  --prior-mode uap --alpha 0.3 \
  --model-config configs/prior_v8_cascade.yaml \
  --out-dir "$OUTDIR" \
  --proc "$PROC" --raw "$RAW" \
  --batch-size 16 --num-workers 16 --epochs 40 --folds 5 \
  >> "$LOG" 2>&1 || echo "[FAIL] E15 OOF prior gen" >> "$LOG"

# Step B: val/test 用 E12 全量 cascade ckpt 推理（uap，本就 OOF）
for sp in val test; do
  python scripts/infer_prior.py \
    --ckpt "$E12CKPT" --arch cascade --model-config configs/prior_v8_cascade.yaml \
    --split "$sp" --prior-mode uap --alpha 0.3 \
    --out-dir "$OUTDIR" --proc "$PROC" --raw "$RAW" \
    >> "$LOG" 2>&1 || echo "[FAIL] E15 $sp prior" >> "$LOG"
done

# Step C: 用 E13 ckpt 在 OOF uap 先验上评测 -> E15
if [ -f runs/bcp_vis/E13_uap/best.pth ]; then
  python eval/eval_final.py \
    --ckpt runs/bcp_vis/E13_uap/best.pth --config configs/bcp_vis.yaml \
    --exp E15 --tag E15 --prior-dir "$OUTDIR" \
    --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E15 eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo "[$(date)] E15 done" >> "$LOG"
fi

echo DONE > logs/e15_queue.done
echo "[$(date)] E15 队列 DONE" >> "$LOG"
