#!/bin/bash
# E30：双专家先验融合（5ch，P_v4(E6) + P_v8(E12)）。等雪控制重评完成后接手。
# priors_v4 与 priors_v8 均已就绪 ⇒ 只训下游。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e30.lock
flock -n 200 || { echo "E30 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e30_queue.log
echo "[$(date)] E30(双专家融合 v4+moe) 队列启动，等 logs/snow_reeval.done" >> "$LOG"
while [ ! -f logs/snow_reeval.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E30" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
N1=$(ls "$PROC/priors_v4"/*.png 2>/dev/null | wc -l)
N2=$(ls "$PROC/priors_v8"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v4=$N1 priors_v8=$N2" >> "$LOG"

if [ "$N1" -ge 9000 ] && [ "$N2" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis_dual_v4moe.yaml \
    --exp E30 --name E30_dual_v4moe --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" >> "$LOG" 2>&1 \
    || echo "[FAIL] E30 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E30_dual_v4moe/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E30_dual_v4moe/best.pth \
      --config configs/bcp_vis_dual_v4moe.yaml --exp E30 --tag E30 \
      --proc "$PROC" --raw "$RAW" --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E30 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e30_queue.done
    echo "[$(date)] E30 done" >> "$LOG"
  else
    echo "[$(date)] E30 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E30 先验不足 v4=$N1 v8=$N2" >> "$LOG"
fi
