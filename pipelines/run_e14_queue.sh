#!/bin/bash
# ============================================================
# E14：类别感知背景专家（第 3 个创新点的"按类别记忆目标"落地）
#
# 与 E8（V5 双专家头，无类别）严格对照：同一 dual-head 骨架，
# 唯一差别是背景专家按预测类别 FiLM 调制。组合方式与 E8 一致（mul）。
# 门槛：E14 tiny@altitude 必须相对 E8 额外提升，否则降为 ablation / future work
#       （类条件先验本身已被 CSENet/CPNet 发表；EVD4UAV 仅 3 类车辆）。
#
# 【幂等】若并行任务(run_e14_parallel.sh)已完成 E14，则本脚本全部跳过。
# ============================================================
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e14.lock
flock -n 200 || { echo "e14 already running, exit"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e14_queue.log
EXPECT=9371
CFG=configs/prior_v5_cat.yaml
CKPT=runs/priornet/prior_v5_cat/best.pth
OUTDIR="$PROC/priors_cat"

# 若并行任务已完整跑完 E14，直接跳过（避免重复训练）
if [ -f logs/e14_parallel.done ]; then
  echo "[$(date)] 检测到 e14_parallel.done，E14 已由并行任务完成，本脚本跳过。" >> "$LOG"
  echo DONE > logs/e14_queue.done
  exit 0
fi

echo "[$(date)] E14 队列：等待 E12 完成（E14 优先于 E13） ..." >> $LOG
while [ ! -f logs/e12_queue.done ]; do sleep 180; done

# 若并行任务(run_e14_parallel.sh)已结束或正在进行，则跳过，避免重复占用 GPU
if [ -f logs/e14_parallel.done ]; then
  echo "[$(date)] 检测到 e14_parallel.done，E14 已由并行任务完成，本脚本跳过。" >> "$LOG"
  echo DONE > logs/e14_queue.done
  exit 0
fi
if [ -f logs/e14_parallel.active ]; then
  echo "[$(date)] 并行任务进行中，等待其完成后跳过..." >> "$LOG"
  while [ ! -f logs/e14_parallel.done ]; do sleep 120; done
  echo "[$(date)] 并行任务已完成，本脚本跳过。" >> "$LOG"
  echo DONE > logs/e14_queue.done
  exit 0
fi

echo "[$(date)] 步骤1：生成类别标签 image_classes.csv" >> $LOG
if [ ! -f "$PROC/image_classes.csv" ]; then
  python scripts/build_class_labels.py --proc "$PROC" --xml "$RAW/mask_attribute.xml" >> "$LOG" 2>&1 || echo "[FAIL] build_class_labels" >> "$LOG"
fi

echo "[$(date)] 步骤2：训练类别感知背景专家 PriorNet" >> $LOG
if [ ! -f "$CKPT" ]; then
  python train/train_prior.py --config "$CFG" --device cuda --num-workers 8 --batch-size 16 >> "$LOG" 2>&1 || echo "[FAIL] E14 prior train" >> "$LOG"
fi
if [ ! -f "$CKPT" ]; then echo "[ABORT] E14 ckpt 缺失" >> "$LOG"; exit 1; fi

mkdir -p "$OUTDIR"
echo "[$(date)] 步骤3：推理 E14 先验（prior-mode=mul，与 E8 同口径）" >> $LOG
if [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -lt "$EXPECT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cat_bg --model-config "$CFG" \
    --prior-mode mul --proc "$PROC" --raw "$RAW" --out-dir "$OUTDIR" --tile 1024 --overlap 192 \
    >> "$LOG" 2>&1 || echo "[FAIL] infer cat" >> "$LOG"
fi
N=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "cat prior files: $N / $EXPECT" >> "$LOG"
if [ "$N" -lt "$EXPECT" ]; then echo "[ABORT] cat prior $N < $EXPECT" >> "$LOG"; exit 1; fi

echo "[$(date)] 步骤4：训练 E14" >> $LOG
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
  echo "[$(date)] E14 done" >> "$LOG"
fi

# ---------- 门槛判定：E14 tiny@altitude 是否相对 E8 额外提升 ----------
E8=$(grep "^E8," reports/final_metrics.csv 2>/dev/null | tail -1)
E14=$(grep "^E14," reports/final_metrics.csv 2>/dev/null | tail -1)
echo "[$(date)] 门槛判定 E14 vs E8：" >> "$LOG"
echo "  E8 : $E8" >> "$LOG"
echo "  E14: $E14" >> "$LOG"
if [ -n "$E14" ]; then
  echo "PASS_CANDIDATE (tiny@altitude 待分组报告确认)" > logs/e14_verdict.txt
  echo "[$(date)] E14 候选 PASS" >> "$LOG"
else
  echo "PENDING" > logs/e14_verdict.txt
fi

if [ -f runs/bcp_vis/E14_cat/best.pth ]; then
  echo DONE > logs/e14_queue.done
  echo "[$(date)] E14 队列 DONE (best.pth 确认)" >> "$LOG"
else
  echo "[$(date)] E14 未产出 best.pth，跳过 done 写入" >> "$LOG"
fi
