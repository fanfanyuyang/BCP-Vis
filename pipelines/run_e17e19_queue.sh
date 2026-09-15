#!/bin/bash
# E17/E18/E19 队列：等 e16_queue.done 后零空闲接手
#   E18 错位先验因果对照（eval-only，最快，先跑）
#   E17 logit 证据融合先验（infer + train + eval）
#   E19 先验一致性正则（train + eval）
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e17e19.lock
flock -n 200 || { echo "e17e19 队列已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e17e19_queue.log
echo "[$(date)] E17/E18/E19 队列启动，等 logs/e16_queue.done" >> "$LOG"
while [ ! -f logs/e16_queue.done ]; do sleep 120; done
echo "[$(date)] E16 完成，开始 E18/E17/E19" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD8="$PROC/priors_v8"
E12CKPT=runs/bcp_vis/E12_cascade/best.pth

# ---------- E18: 错位先验因果对照（无训练） ----------
python scripts/make_misaligned_priors.py --src "$PD8" --out "$PROC/priors_v8_misaligned" \
  >> "$LOG" 2>&1 || echo "[FAIL] E18 misalign gen" >> "$LOG"
N=$(ls "$PROC/priors_v8_misaligned"/*.png 2>/dev/null | wc -l)
echo "[$(date)] misaligned priors: $N" >> "$LOG"
if [ "$N" -ge 9000 ] && [ -f "$E12CKPT" ]; then
  python eval/eval_final.py --ckpt "$E12CKPT" --config configs/bcp_vis.yaml \
    --exp E18 --tag E18 --proc "$PROC" --raw "$RAW" \
    --prior-dir "$PROC/priors_v8_misaligned" --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E18 eval" >> "$LOG"
  echo "[$(date)] E18 eval done" >> "$LOG"
fi

# ---------- E17: logit 证据融合先验 ----------
python scripts/infer_prior.py --ckpt runs/priornet/prior_v8_cascade/best.pth \
  --arch cascade --model-config configs/prior_v8_cascade.yaml \
  --prior-mode logit --alpha 1.0 \
  --proc "$PROC" --raw "$RAW" --out-dir "$PROC/priors_v8_logit" \
  --tile 1024 --overlap 192 \
  >> "$LOG" 2>&1 || echo "[FAIL] E17 infer" >> "$LOG"
NL=$(ls "$PROC/priors_v8_logit"/*.png 2>/dev/null | wc -l)
echo "[$(date)] logit priors: $NL" >> "$LOG"
if [ "$NL" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E17 --name E17_logit --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8_logit" \
    >> "$LOG" 2>&1 || echo "[FAIL] E17 train" >> "$LOG"
  if [ -f runs/bcp_vis/E17_logit/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E17_logit/best.pth \
      --config configs/bcp_vis.yaml --exp E17 --tag E17 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8_logit" \
      --batch-size 8 --num-workers 8 \
      >> "$LOG" 2>&1 || echo "[FAIL] E17 eval" >> "$LOG"
    echo "[$(date)] E17 done" >> "$LOG"
  fi
fi

# ---------- E19: 先验一致性正则 ----------
python train/train_final.py --config configs/bcp_vis_pc.yaml \
  --exp E19 --name E19_pc --batch-size 16 --num-workers 8 \
  --proc "$PROC" --raw "$RAW" --prior-dir "$PD8" \
  >> "$LOG" 2>&1 || echo "[FAIL] E19 train" >> "$LOG"
if [ -f runs/bcp_vis/E19_pc/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E19_pc/best.pth \
    --config configs/bcp_vis_pc.yaml --exp E19 --tag E19 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD8" \
    --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E19 eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo "[$(date)] E19 done" >> "$LOG"
fi

echo "[$(date)] E17/E18/E19 队列收尾" >> "$LOG"
if [ -f runs/bcp_vis/E17_logit/best.pth ] && [ -f runs/bcp_vis/E19_pc/best.pth ]; then
  echo DONE > logs/e17e19_queue.done
  echo "[$(date)] E17/E18/E19 队列 DONE" >> "$LOG"
else
  echo "[$(date)] 关键产物缺失，不写 done（防假完成）" >> "$LOG"
fi
