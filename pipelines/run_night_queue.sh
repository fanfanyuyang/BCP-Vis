#!/bin/bash
# ============================================================
# 夜间实验队列（AutoDL 整夜不停）
#
# 顺序（能跑多少算多少，每步都有产物断言）：
#   0. 等 V5-A -> E8 完成
#   1. V5-B（MoE 多背景专家，8 专家 / 宽度 64）-> 推理 -> 训 E9 -> 评估
#   2. 容量对照 Large-Plain（骨干翻倍、无专家）-> 推理 -> 训 E10 -> 评估
#
# E10 的意义：它的参数量【大于】MoE 版，却没有专家结构。
#   若 E9 胜出 ⇒ 增益来自专家结构而非堆参数 ⇒ 论文结论才立得住
# ============================================================
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_night.lock
flock -n 200 || { echo "night queue already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/night_queue.log
EXPECT=9371

echo "[$(date)] night queue: waiting for V5 pipeline (E8) ..." >> $LOG
while [ ! -f logs/v5_pipeline.done ]; do sleep 120; done
echo "[$(date)] E8 done. start V5-B (MoE)" >> $LOG

# ---------------- 1) V5-B MoE 多背景专家 ----------------
python train/train_prior.py --config configs/prior_v5_moe.yaml \
    --device cuda --num-workers 8 --batch-size 16 \
    >> $LOG 2>&1 || echo "[FAIL] V5-B train" >> $LOG

V5B=runs/priornet/prior_v5_moe/best.pth
MOEDIR="$PROC/priors_v5moe"
mkdir -p "$MOEDIR"
if [ -f "$V5B" ]; then
    python scripts/infer_prior.py --ckpt "$V5B" --arch moe --model-config configs/prior_v5_moe.yaml \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$MOEDIR" --tile 1024 --overlap 192 \
        >> $LOG 2>&1 || echo "[FAIL] infer moe" >> $LOG
    N=$(ls "$MOEDIR"/*.png 2>/dev/null | wc -l)
    echo "moe prior files: $N / $EXPECT" >> $LOG
    if [ "$N" -ge "$EXPECT" ]; then
        python train/train_final.py --config configs/bcp_vis.yaml \
            --exp E9 --name E9_moe --batch-size 16 --num-workers 8 \
            --proc "$PROC" --raw "$RAW" --prior-dir "$MOEDIR" \
            >> $LOG 2>&1 || echo "[FAIL] E9 train" >> $LOG
        if [ -f runs/bcp_vis/E9_moe/best.pth ]; then
            python eval/eval_final.py \
                --ckpt runs/bcp_vis/E9_moe/best.pth --config configs/bcp_vis.yaml \
                --exp E9 --tag E9 --proc "$PROC" --raw "$RAW" \
                --prior-dir "$MOEDIR" --batch-size 8 --num-workers 8 \
                >> $LOG 2>&1 || echo "[FAIL] E9 eval" >> $LOG
            python eval/make_report.py >> $LOG 2>&1
            echo "[$(date)] E9 done" >> $LOG
        fi
    else
        echo "[ABORT] moe prior count $N < $EXPECT" >> $LOG
    fi
fi

# ---------------- 2) 容量对照：大号普通 PriorNet ----------------
echo "[$(date)] start Large-Plain (capacity control)" >> $LOG
python train/train_prior.py --config configs/prior_large_plain.yaml \
    --device cuda --num-workers 8 --batch-size 16 \
    >> $LOG 2>&1 || echo "[FAIL] Large train" >> $LOG

LP=runs/priornet/prior_large_plain/best.pth
LPDIR="$PROC/priors_large"
mkdir -p "$LPDIR"
if [ -f "$LP" ]; then
    python scripts/infer_prior.py --ckpt "$LP" --arch litehr --model-config configs/prior_large_plain.yaml \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$LPDIR" --tile 1024 --overlap 192 \
        >> $LOG 2>&1 || echo "[FAIL] infer large" >> $LOG
    N=$(ls "$LPDIR"/*.png 2>/dev/null | wc -l)
    echo "large prior files: $N / $EXPECT" >> $LOG
    if [ "$N" -ge "$EXPECT" ]; then
        python train/train_final.py --config configs/bcp_vis.yaml \
            --exp E10 --name E10_large --batch-size 16 --num-workers 8 \
            --proc "$PROC" --raw "$RAW" --prior-dir "$LPDIR" \
            >> $LOG 2>&1 || echo "[FAIL] E10 train" >> $LOG
        if [ -f runs/bcp_vis/E10_large/best.pth ]; then
            python eval/eval_final.py \
                --ckpt runs/bcp_vis/E10_large/best.pth --config configs/bcp_vis.yaml \
                --exp E10 --tag E10 --proc "$PROC" --raw "$RAW" \
                --prior-dir "$LPDIR" --batch-size 8 --num-workers 8 \
                >> $LOG 2>&1 || echo "[FAIL] E10 eval" >> $LOG
            python eval/make_report.py >> $LOG 2>&1
            echo "[$(date)] E10 done" >> $LOG
        fi
    else
        echo "[ABORT] large prior count $N < $EXPECT" >> $LOG
    fi
fi

echo DONE > logs/night_queue.done
echo "[$(date)] night queue DONE" >> $LOG
