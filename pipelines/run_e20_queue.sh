#!/bin/bash
# E20：双先验通道解耦；等 e17e19_queue.done 后接手
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e20.lock
flock -n 200 || { echo "E20 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e20_queue.log
echo "[$(date)] E20 队列启动，等 logs/e17e19_queue.done" >> "$LOG"
while [ ! -f logs/e17e19_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E20（双先验通道 P_f/P_b 解耦，5 通道）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CKPT=runs/priornet/prior_v8_cascade/best.pth
CFG=configs/prior_v8_cascade.yaml

python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade --model-config "$CFG" \
  --prior-mode raw --proc "$PROC" --raw "$RAW" \
  --out-dir "$PROC/priors_v8_pf" --tile 1024 --overlap 192 \
  >> "$LOG" 2>&1 || echo "[FAIL] E20 infer pf" >> "$LOG"

python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade --model-config "$CFG" \
  --prior-mode raw_bg --proc "$PROC" --raw "$RAW" \
  --out-dir "$PROC/priors_v8_pb" --tile 1024 --overlap 192 \
  >> "$LOG" 2>&1 || echo "[FAIL] E20 infer pb" >> "$LOG"

NPF=$(ls "$PROC/priors_v8_pf"/*.png 2>/dev/null | wc -l)
NPB=$(ls "$PROC/priors_v8_pb"/*.png 2>/dev/null | wc -l)
echo "[$(date)] pf=$NPF pb=$NPB" >> "$LOG"

if [ "$NPF" -ge 9000 ] && [ "$NPB" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis_dual.yaml \
    --exp E20 --name E20_dual --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" \
    >> "$LOG" 2>&1 || echo "[FAIL] E20 train" >> "$LOG"
  if [ -f runs/bcp_vis/E20_dual/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E20_dual/best.pth \
      --config configs/bcp_vis_dual.yaml \
      --exp E20 --tag E20 --proc "$PROC" --raw "$RAW" \
      --batch-size 8 --num-workers 8 \
      >> "$LOG" 2>&1 || echo "[FAIL] E20 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e20_queue.done
    echo "[$(date)] E20 done" >> "$LOG"
  else
    echo "[$(date)] E20 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
fi
