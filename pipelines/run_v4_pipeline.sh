#!/bin/bash
# V4 → E6 自动流水线
# 触发：等待 V4 PriorNet 训练完成（logs/train_v4.done）后自动接上
#   1. 用 V4(Altitude-Aware) 推理高度自适应先验 → priors_v4/
#   2. 训练 E6（RGB + V4 高度先验）
#   3. 评估 E6（含尺寸×高度交叉）
#   4. 生成统一报告（含 V4 落地验证：tiny@90m 是否转正）
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_v4pipe.lock
flock -n 200 || { echo "pipeline already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
V4DIR="$PROC/priors_v4"
LOG=logs/v4_pipeline.log
CKPT=runs/priornet/prior_v4_altitude/best.pth

echo "[$(date)] pipeline start: waiting for V4 ..." >> $LOG
while [ ! -f logs/train_v4.done ]; do sleep 120; done
echo "[$(date)] V4 finished, ckpt=$CKPT" >> $LOG

mkdir -p "$V4DIR"
echo "[$(date)] STEP infer prior (V4 altitude-aware)" >> $LOG
python scripts/infer_prior.py \
    --ckpt "$CKPT" --arch altitude \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$V4DIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer v4" >> $LOG
echo "v4 prior files: $(ls $V4DIR/*.png 2>/dev/null | wc -l)" >> $LOG

echo "[$(date)] STEP train E6" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E6 --name E6_altitude --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$V4DIR" \
    >> $LOG 2>&1 || echo "[FAIL] E6 train" >> $LOG

echo "[$(date)] STEP eval E6" >> $LOG
python eval/eval_final.py \
    --ckpt runs/bcp_vis/E6_altitude/best.pth --config configs/bcp_vis.yaml \
    --exp E6 --tag E6 --proc "$PROC" --raw "$RAW" \
    --prior-dir "$V4DIR" --batch-size 8 --num-workers 8 \
    >> $LOG 2>&1 || echo "[FAIL] E6 eval" >> $LOG

echo "[$(date)] STEP report" >> $LOG
python eval/make_report.py >> $LOG 2>&1 || echo "[FAIL] report" >> $LOG

echo DONE > logs/v4_pipeline.done
echo "[$(date)] pipeline DONE" >> $LOG
