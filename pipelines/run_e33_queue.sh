#!/bin/bash
# E33：③（不确定性融合）的公平重测 —— 健康基准(priors_v4) + 尺度归一化分歧。
# 等 E32 完成后接手，串行不抢卡。合成用 CPU，随后只训下游。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e33.lock
flock -n 200 || { echo "E33 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e33_queue.log
echo "[$(date)] E33(③公平重测) 队列启动，等 logs/e32_queue.done" >> "$LOG"
while [ ! -f logs/e32_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E33" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD="$PROC/priors_v33_du"

python _synth_du2.py >> "$LOG" 2>&1 || echo "[FAIL] E33 synth" >> "$LOG"
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v33_du = $NP" >> "$LOG"

if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E33 --name E33_du2 --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E33 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E33_du2/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E33_du2/best.pth \
      --config configs/bcp_vis.yaml --exp E33 --tag E33 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E33 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e33_queue.done
    echo "[$(date)] E33 done" >> "$LOG"
  else
    echo "[$(date)] E33 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E33 先验不足: $NP" >> "$LOG"
fi
