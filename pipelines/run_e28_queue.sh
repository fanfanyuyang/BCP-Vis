#!/bin/bash
# E28：双专家先验融合（5ch，RGB + P_film + P_moe）。等 E26 完成后接手（串行保持 GPU 满载）。
# 两个先验均已训好 ⇒ 只训下游，成本最低、直接回答"最好组合"。
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e28.lock
flock -n 200 || { echo "E28 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e28_queue.log
echo "[$(date)] E28(双专家融合5ch) 队列启动，等 logs/e26_queue.done" >> "$LOG"
while [ ! -f logs/e26_queue.done ]; do sleep 120; done
echo "[$(date)] 上游 E26 完成，开始 E28" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
NPF=$(ls "$PROC/priors_v8_film"/*.png 2>/dev/null | wc -l)
NPM=$(ls "$PROC/priors_v8"/*.png 2>/dev/null | wc -l)
echo "[$(date)] film=$NPF moe=$NPM" >> "$LOG"

if [ "$NPF" -ge 9000 ] && [ "$NPM" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis_dual_filmmoe.yaml \
    --exp E28 --name E28_filmmoe --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" >> "$LOG" 2>&1 \
    || echo "[FAIL] E28 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E28_filmmoe/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E28_filmmoe/best.pth \
      --config configs/bcp_vis_dual_filmmoe.yaml --exp E28 --tag E28 \
      --proc "$PROC" --raw "$RAW" --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E28 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e28_queue.done
    echo "[$(date)] E28 done" >> "$LOG"
  else
    echo "[$(date)] E28 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E28 先验缺失 film=$NPF moe=$NPM" >> "$LOG"
fi
