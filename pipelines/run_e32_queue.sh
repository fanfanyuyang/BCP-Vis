#!/bin/bash
# E32：E24（V8 级联 + FiLM）的 seed43 复制，用于论文方差。等 E31 完成后接手。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e32.lock
flock -n 200 || { echo "E32 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e32_queue.log
echo "[$(date)] E32(E24 seed43 方差) 队列启动，等 logs/e31_queue.done" >> "$LOG"
while [ ! -f logs/e31_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E32" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFGP=configs/prior_v8_cascade_film_s43.yaml
CKPT=runs/priornet/prior_v8_cascade_film_s43/best.pth
PD="$PROC/priors_v8_film_s43"

if [ ! -f "$CKPT" ]; then
  python train/train_prior.py --config "$CFGP" --device cuda \
    --num-workers 16 --batch-size 16 >> "$LOG" 2>&1 \
    || echo "[FAIL] E32 prior train" >> "$LOG"
else
  echo "[$(date)] E32 复用已有 ckpt，跳过重训" >> "$LOG"
fi

if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E32 infer" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_film_s43 = $NP" >> "$LOG"

if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E32 --name E32_film_s43 --seed 43 --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E32 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E32_film_s43/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E32_film_s43/best.pth \
      --config configs/bcp_vis.yaml --exp E32 --tag E32 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E32 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e32_queue.done
    echo "[$(date)] E32 done" >> "$LOG"
  else
    echo "[$(date)] E32 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E32 先验不足: $NP" >> "$LOG"
fi
