#!/bin/bash
# V5 自动流水线：双分支专家头（前景专家 + 背景专家）验证
#   1. 等 E6/V4 流水线结束（不抢 GPU）
#   2. 训 V5-A DualHead PriorNet（前景头监督 S，背景头监督 1-S）
#   3. 推理先验：P = P_f ⊙ (1 - P_b)（"是目标 且 不是背景"）
#   4. 训 E8（RGB + 双分支先验，4 通道，与 E5/E6 同架构只换先验）
#   5. 评估 E8 + 生成报告
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_v5.lock
flock -n 200 || { echo "v5 pipeline already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
V5DIR="$PROC/priors_v5d"
LOG=logs/v5_pipeline.log
EXPECT=9371

echo "[$(date)] V5 pipeline: waiting for E6/V4 pipeline ..." >> $LOG
while [ ! -f logs/v4_pipeline.done ]; do sleep 120; done
echo "[$(date)] E6 done, start V5-A training" >> $LOG

# ---------- 1) 训练 V5-A 双分支 PriorNet ----------
python train/train_prior.py --config configs/prior_v5_dual.yaml \
    --device cuda --num-workers 8 --batch-size 16 \
    >> $LOG 2>&1 || echo "[FAIL] V5-A train" >> $LOG

CKPT=runs/priornet/prior_v5_dual/best.pth
if [ ! -f "$CKPT" ]; then
    echo "[ABORT] V5-A ckpt missing" >> $LOG
    exit 1
fi

# ---------- 2) 推理双分支先验 ----------
mkdir -p "$V5DIR"
echo "[$(date)] STEP infer prior (V5 dual)" >> $LOG
python scripts/infer_prior.py \
    --ckpt "$CKPT" --arch dual \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$V5DIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer v5" >> $LOG

N=$(ls "$V5DIR"/*.png 2>/dev/null | wc -l)
echo "v5 prior files: $N / $EXPECT" >> $LOG
if [ "$N" -lt "$EXPECT" ]; then
    echo "[ABORT] prior count $N < $EXPECT" >> $LOG
    exit 1
fi

# ---------- 3) 训练 E8 ----------
echo "[$(date)] STEP train E8 (RGB + dual prior)" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E8 --name E8_dual --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$V5DIR" \
    >> $LOG 2>&1 || echo "[FAIL] E8 train" >> $LOG

if [ ! -f runs/bcp_vis/E8_dual/best.pth ]; then
    echo "[ABORT] E8 ckpt missing" >> $LOG
    exit 1
fi

# ---------- 4) 评估 E8 ----------
echo "[$(date)] STEP eval E8" >> $LOG
python eval/eval_final.py \
    --ckpt runs/bcp_vis/E8_dual/best.pth --config configs/bcp_vis.yaml \
    --exp E8 --tag E8 --proc "$PROC" --raw "$RAW" \
    --prior-dir "$V5DIR" --batch-size 8 --num-workers 8 \
    >> $LOG 2>&1 || echo "[FAIL] E8 eval" >> $LOG

# ---------- 5) 报告 ----------
echo "[$(date)] STEP report" >> $LOG
python eval/make_report.py >> $LOG 2>&1 || echo "[FAIL] report" >> $LOG

echo DONE > logs/v5_pipeline.done
echo "[$(date)] V5 pipeline DONE" >> $LOG
