#!/bin/bash
# E29：专家分歧不确定性（P* = (1-|P_film-P_moe|) * P_film）。等 E28 完成后接手。
# 零额外 PriorNet 训练：合成分歧先验 ⇒ 只训下游。这是"UAP 的正确做法"。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e29.lock
flock -n 200 || { echo "E29 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e29_queue.log
echo "[$(date)] E29(专家分歧不确定性) 队列启动，等 logs/e28_queue.done" >> "$LOG"
while [ ! -f logs/e28_queue.done ]; do sleep 120; done
echo "[$(date)] 上游 E28 完成，开始 E29" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD="$PROC/priors_v8_du"

# 1) 合成分歧先验（复用两个已训专家，无需 GPU 训练）
python _synth_du.py >> "$LOG" 2>&1 || echo "[FAIL] E29 synth" >> "$LOG"
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_du = $NP" >> "$LOG"

# 2) 下游 4ch 训练 + 评测
if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E29 --name E29_du --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E29 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E29_du/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E29_du/best.pth \
      --config configs/bcp_vis.yaml --exp E29 --tag E29 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E29 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e29_queue.done
    echo "[$(date)] E29 done" >> "$LOG"
  else
    echo "[$(date)] E29 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E29 先验不足: $NP" >> "$LOG"
fi
