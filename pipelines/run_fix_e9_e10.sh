#!/bin/bash
# 修复补跑：V5-B(MoE) 与 Large-Plain 的训练其实已完成，
# 失败的是【推理先验】这一步（infer_prior 用默认维度建模型导致 shape 不匹配，已修）。
# 本脚本跳过训练，直接从已有 ckpt 重做推理 -> 训 E9/E10 -> 评估。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_fix.lock
flock -n 200 || { echo "fix queue already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/fix_e9_e10.log
EXPECT=9371
echo "[$(date)] 修复补跑开始" >> $LOG

# ---------------- 1) MoE -> E9 ----------------
MOEDIR="$PROC/priors_v5moe"; mkdir -p "$MOEDIR"
V5B=runs/priornet/prior_v5_moe/best.pth
if [ -f "$V5B" ] && [ ! -f runs/bcp_vis/E9_moe/best.pth ]; then
    echo "[$(date)] infer MoE prior" >> $LOG
    python scripts/infer_prior.py --ckpt "$V5B" --arch moe \
        --model-config configs/prior_v5_moe.yaml \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$MOEDIR" --tile 1024 --overlap 192 \
        >> $LOG 2>&1 || echo "[FAIL] infer moe" >> $LOG
    N=$(ls "$MOEDIR"/*.png 2>/dev/null | wc -l)
    echo "moe prior files: $N / $EXPECT" >> $LOG
    if [ "$N" -ge "$EXPECT" ]; then
        echo "[$(date)] train E9" >> $LOG
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
        echo "[ABORT] moe prior $N < $EXPECT" >> $LOG
    fi
fi

# ---------------- 2) 容量对照 Large-Plain -> E10 ----------------
LPDIR="$PROC/priors_large"; mkdir -p "$LPDIR"
LP=runs/priornet/prior_large_plain/best.pth
if [ -f "$LP" ] && [ ! -f runs/bcp_vis/E10_large/best.pth ]; then
    echo "[$(date)] infer Large prior" >> $LOG
    python scripts/infer_prior.py --ckpt "$LP" --arch litehr \
        --model-config configs/prior_large_plain.yaml \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$LPDIR" --tile 1024 --overlap 192 \
        >> $LOG 2>&1 || echo "[FAIL] infer large" >> $LOG
    N=$(ls "$LPDIR"/*.png 2>/dev/null | wc -l)
    echo "large prior files: $N / $EXPECT" >> $LOG
    if [ "$N" -ge "$EXPECT" ]; then
        echo "[$(date)] train E10" >> $LOG
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
        echo "[ABORT] large prior $N < $EXPECT" >> $LOG
    fi
fi

echo DONE > logs/fix_e9_e10.done
echo "[$(date)] 修复补跑 DONE" >> $LOG
