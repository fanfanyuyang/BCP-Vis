#!/bin/bash
# E22 = 双尺度先验通道；等 E21 完成后接手
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e22.lock
flock -n 200 || { echo "E22 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e22_queue.log
echo "[$(date)] E22 队列启动，等 logs/e21_queue.done" >> "$LOG"
while [ ! -f logs/e21_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E22（双尺度先验）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV

python scripts/make_coarse_prior.py --src "$PROC/priors_v8" \
  --out "$PROC/priors_v8_coarse" --sigma 12 >> "$LOG" 2>&1 \
  || echo "[FAIL] E22 coarse gen" >> "$LOG"
NC=$(ls "$PROC/priors_v8_coarse"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_coarse = $NC" >> "$LOG"

if [ "$NC" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis_multiscale.yaml \
    --exp E22 --name E22_ms --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" >> "$LOG" 2>&1 || echo "[FAIL] E22 train" >> "$LOG"
  if [ -f runs/bcp_vis/E22_ms/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E22_ms/best.pth \
      --config configs/bcp_vis_multiscale.yaml --exp E22 --tag E22 \
      --proc "$PROC" --raw "$RAW" --batch-size 8 --num-workers 8 \
      >> "$LOG" 2>&1 || echo "[FAIL] E22 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e22_queue.done
    echo "[$(date)] E22 done" >> "$LOG"
  else
    echo "[$(date)] E22 未产出 best.pth，不写 done" >> "$LOG"
  fi
fi
