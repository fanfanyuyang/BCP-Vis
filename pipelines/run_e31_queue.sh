#!/bin/bash
# E31：E12（V8 级联/moe）的 seed43 复制，用于论文方差。等 E30 完成后接手。
# 完整流程：PriorNet(seed43) -> 推理(默认 mul，λ_bg=1 背景头已训) -> 下游(seed43) -> 评测。
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e31.lock
flock -n 200 || { echo "E31 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e31_queue.log
echo "[$(date)] E31(E12 seed43 方差) 队列启动，等 logs/e30_queue.done" >> "$LOG"
while [ ! -f logs/e30_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E31" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFGP=configs/prior_v8_cascade_s43.yaml
CKPT=runs/priornet/prior_v8_cascade_s43/best.pth
PD="$PROC/priors_v8_s43"

if [ ! -f "$CKPT" ]; then
  python train/train_prior.py --config "$CFGP" --device cuda \
    --num-workers 16 --batch-size 16 >> "$LOG" 2>&1 \
    || echo "[FAIL] E31 prior train" >> "$LOG"
else
  echo "[$(date)] E31 复用已有 ckpt，跳过重训" >> "$LOG"
fi

if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E31 infer" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_s43 = $NP" >> "$LOG"

if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E31 --name E31_cascade_s43 --seed 43 --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E31 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E31_cascade_s43/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E31_cascade_s43/best.pth \
      --config configs/bcp_vis.yaml --exp E31 --tag E31 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E31 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e31_queue.done
    echo "[$(date)] E31 done" >> "$LOG"
  else
    echo "[$(date)] E31 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E31 先验不足: $NP" >> "$LOG"
fi
