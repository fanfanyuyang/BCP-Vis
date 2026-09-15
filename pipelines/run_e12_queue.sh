#!/bin/bash
# ============================================================
# E12：V8 级联融合（第 1 级高度 + 第 2 级目标/背景）
#
# 【自动决策】第 1 级用哪种高度机制，由 E11 与 E6 的实测结果决定：
#     E11（高度专家）IoU > E6（FiLM）IoU  ⇒ height_mode = moe
#     否则                                ⇒ height_mode = film
#   脚本自动比较并改写配置，无需人工介入。
# ============================================================
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e12.lock
flock -n 200 || { echo "e12 already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e12_queue.log
EXPECT=9371
CFG=configs/prior_v8_cascade.yaml
CKPT=runs/priornet/prior_v8_cascade/best.pth
OUTDIR="$PROC/priors_v8"

echo "[$(date)] E12 队列：等待 E11 完成 ..." >> $LOG
while [ ! -f logs/e11_queue.done ]; do sleep 180; done

# ---------- 自动决策：高度机制 ----------
E11=$(grep "^E11," reports/final_metrics.csv 2>/dev/null | tail -1 | cut -d, -f3)
E6=$(grep "^E6," reports/final_metrics.csv 2>/dev/null | tail -1 | cut -d, -f3)
MODE=$(awk -v a="${E11:-0}" -v b="${E6:-0}" 'BEGIN{print (a+0 > b+0) ? "moe" : "film"}')
echo "[$(date)] E11 IoU=${E11:-NA}  E6 IoU=${E6:-NA}  => height_mode=$MODE" >> $LOG
sed -i "s/^  height_mode:.*/  height_mode: $MODE/" $CFG
grep "^  height_mode:" $CFG >> $LOG

echo "[$(date)] 开始训练 V8 级联先验网络" >> $LOG
python train/train_prior.py --config $CFG \
    --device cuda --num-workers 8 --batch-size 16 \
    >> $LOG 2>&1 || echo "[FAIL] V8 train" >> $LOG

if [ ! -f "$CKPT" ]; then
    echo "[ABORT] V8 ckpt 缺失" >> $LOG
    exit 1
fi

mkdir -p "$OUTDIR"
echo "[$(date)] 推理 V8 先验" >> $LOG
python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config $CFG \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$OUTDIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer v8" >> $LOG

N=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "v8 prior files: $N / $EXPECT" >> $LOG
if [ "$N" -lt "$EXPECT" ]; then
    echo "[ABORT] v8 prior $N < $EXPECT" >> $LOG
    exit 1
fi

echo "[$(date)] 训练 E12" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E12 --name E12_cascade --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$OUTDIR" \
    >> $LOG 2>&1 || echo "[FAIL] E12 train" >> $LOG

if [ -f runs/bcp_vis/E12_cascade/best.pth ]; then
    python eval/eval_final.py \
        --ckpt runs/bcp_vis/E12_cascade/best.pth --config configs/bcp_vis.yaml \
        --exp E12 --tag E12 --proc "$PROC" --raw "$RAW" \
        --prior-dir "$OUTDIR" --batch-size 8 --num-workers 8 \
        >> $LOG 2>&1 || echo "[FAIL] E12 eval" >> $LOG
    python eval/make_report.py >> $LOG 2>&1
    echo "[$(date)] E12 done" >> $LOG
fi

echo DONE > logs/e12_queue.done
echo "[$(date)] E12 队列 DONE" >> $LOG
