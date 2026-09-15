#!/bin/bash
# ============================================================
# E11：V7 高度专家 MoE（软路由按高度分工）验证
#
# 与 E6（V4-FiLM 高度旋钮）严格对照：骨干相同、单目标头、只改高度机制
#   E6  : F' = (1+γ(h))·F + β(h)         共享卷积核，只缩放/平移
#   E11 : P = Σ_k w_k(h)·Expert_k(F)     每高度档独立分支
#
# 排队：等 E9/E10 补跑（fix_e9_e10）结束后自动开始
# ============================================================
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e11.lock
flock -n 200 || { echo "e11 queue already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e11_queue.log
EXPECT=9371
CFG=configs/prior_v7_altmoe.yaml
CKPT=runs/priornet/prior_v7_altmoe/best.pth
OUTDIR="$PROC/priors_v7"

echo "[$(date)] E11 队列：等待 E9/E10 补跑结束 ..." >> $LOG
while [ ! -f logs/fix_e9_e10.done ]; do sleep 120; done
echo "[$(date)] 开始训练 V7 高度专家 MoE" >> $LOG

python train/train_prior.py --config $CFG \
    --device cuda --num-workers 8 --batch-size 16 \
    >> $LOG 2>&1 || echo "[FAIL] V7 train" >> $LOG

if [ ! -f "$CKPT" ]; then
    echo "[ABORT] V7 ckpt 缺失" >> $LOG
    exit 1
fi

mkdir -p "$OUTDIR"
echo "[$(date)] 推理 V7 先验" >> $LOG
python scripts/infer_prior.py --ckpt "$CKPT" --arch alt_moe \
    --model-config $CFG \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$OUTDIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer v7" >> $LOG

N=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "v7 prior files: $N / $EXPECT" >> $LOG
if [ "$N" -lt "$EXPECT" ]; then
    echo "[ABORT] v7 prior $N < $EXPECT" >> $LOG
    exit 1
fi

echo "[$(date)] 训练 E11" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E11 --name E11_altmoe --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$OUTDIR" \
    >> $LOG 2>&1 || echo "[FAIL] E11 train" >> $LOG

if [ -f runs/bcp_vis/E11_altmoe/best.pth ]; then
    python eval/eval_final.py \
        --ckpt runs/bcp_vis/E11_altmoe/best.pth --config configs/bcp_vis.yaml \
        --exp E11 --tag E11 --proc "$PROC" --raw "$RAW" \
        --prior-dir "$OUTDIR" --batch-size 8 --num-workers 8 \
        >> $LOG 2>&1 || echo "[FAIL] E11 eval" >> $LOG
    python eval/make_report.py >> $LOG 2>&1
    echo "[$(date)] E11 done" >> $LOG
fi

echo DONE > logs/e11_queue.done
echo "[$(date)] E11 队列 DONE" >> $LOG
