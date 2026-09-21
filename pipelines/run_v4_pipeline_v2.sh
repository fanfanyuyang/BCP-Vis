#!/bin/bash
# V4 → E6 自动流水线（v2：带产物校验断言）
#
# v1 的教训：infer_prior 静默失败生成 0 个先验，E6 仍照常训练 5 小时，
#            第4通道退化成全零（≡ E1），白白浪费一整轮。
# v2 改进：每个关键步骤后校验产物数量，不达标立即中止并写 [ABORT]。
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_v4pipe.lock
flock -n 200 || { echo "pipeline already running, exit"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
V4DIR="$PROC/priors_v4"
LOG=logs/v4_pipeline.log
CKPT=runs/priornet/prior_v4_altitude/best.pth
EXPECT=9371          # 必须生成的先验文件数（全量图像数）

echo "[$(date)] pipeline v2 start: waiting for V4 ..." >> $LOG
while [ ! -f logs/train_v4.done ]; do sleep 120; done
echo "[$(date)] V4 finished, ckpt=$CKPT" >> $LOG

# ---------- 1) 推理 V4 高度自适应先验 ----------
mkdir -p "$V4DIR"
echo "[$(date)] STEP infer prior (V4 altitude-aware)" >> $LOG
python scripts/infer_prior.py \
    --ckpt "$CKPT" --arch altitude \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$V4DIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer v4" >> $LOG

N=$(ls "$V4DIR"/*.png 2>/dev/null | wc -l)
echo "v4 prior files: $N / expect $EXPECT" >> $LOG
# 断言：先验数量不足则中止，绝不带着空先验训练
if [ "$N" -lt "$EXPECT" ]; then
    echo "[ABORT] prior count $N < $EXPECT, stop before E6 (避免空先验导致第4通道退化为全零)" >> $LOG
    exit 1
fi

# ---------- 2) 训练 E6 ----------
echo "[$(date)] STEP train E6" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E6 --name E6_altitude --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$V4DIR" \
    >> $LOG 2>&1 || echo "[FAIL] E6 train" >> $LOG

if [ ! -f runs/bcp_vis/E6_altitude/best.pth ]; then
    echo "[ABORT] E6 ckpt missing, stop" >> $LOG
    exit 1
fi

# ---------- 3) 评估 E6 ----------
echo "[$(date)] STEP eval E6" >> $LOG
python eval/eval_final.py \
    --ckpt runs/bcp_vis/E6_altitude/best.pth --config configs/bcp_vis.yaml \
    --exp E6 --tag E6 --proc "$PROC" --raw "$RAW" \
    --prior-dir "$V4DIR" --batch-size 8 --num-workers 8 \
    >> $LOG 2>&1 || echo "[FAIL] E6 eval" >> $LOG

# ---------- 4) 报告 ----------
echo "[$(date)] STEP report" >> $LOG
python eval/make_report.py >> $LOG 2>&1 || echo "[FAIL] report" >> $LOG

echo DONE > logs/v4_pipeline.done
echo "[$(date)] pipeline DONE" >> $LOG
