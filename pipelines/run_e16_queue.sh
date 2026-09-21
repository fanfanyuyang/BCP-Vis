#!/bin/bash
# E16：ACPC —— 高度条件化的先验通道；等 E13b 完成后零空闲接手
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e16.lock
flock -n 200 || { echo "E16 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e16_queue.log
echo "[$(date)] E16 队列启动，等 logs/e13b_queue.done" >> "$LOG"
while [ ! -f logs/e13b_queue.done ]; do sleep 120; done
echo "[$(date)] E13b 完成，开始 E16 (ACPC 高度条件先验通道)" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD=datasets/EVD4UAV_processed/priors_v8

python train/train_final.py --config configs/bcp_vis_acpc.yaml \
  --exp E16 --name E16_acpc --batch-size 16 --num-workers 8 \
  --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
  >> "$LOG" 2>&1 || echo "[FAIL] E16 train" >> "$LOG"

# done 哨兵仅当 best.pth 存在才写（杜绝 E14 式假完成）
if [ -f runs/bcp_vis/E16_acpc/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E16_acpc/best.pth \
    --config configs/bcp_vis_acpc.yaml \
    --exp E16 --tag E16 --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
    --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E16 eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo "[$(date)] E16 done" >> "$LOG"
  echo DONE > logs/e16_queue.done
  echo "[$(date)] E16 队列 DONE (best.pth 确认)" >> "$LOG"
else
  echo "[$(date)] E16 未产出 best.pth，跳过 done 写入" >> "$LOG"
fi
