#!/bin/bash
# E21 = AASP（高度自适应软先验）：等「全量重评」完成后接手
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e21.lock
flock -n 200 || { echo "E21 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e21_queue.log
echo "[$(date)] E21(AASP) 队列启动，等 logs/reeval_all_queue.done" >> "$LOG"
while [ ! -f logs/reeval_all_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E21 AASP" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
SOFT="$PROC/soft_priors_aasp"
CFGP=configs/prior_v8_cascade_aasp.yaml
CKPT=runs/priornet/prior_v8_cascade_aasp/best.pth
PD="$PROC/priors_v8_aasp"

python scripts/build_soft_prior_aasp.py --proc "$PROC" --out "$SOFT" \
  --sigma0 32 --alt-ref 70 --workers 12 --format png >> "$LOG" 2>&1 \
  || echo "[FAIL] E21 soft gen" >> "$LOG"
NS=$(ls "$SOFT"/*.png 2>/dev/null | wc -l)
echo "[$(date)] soft_priors_aasp = $NS" >> "$LOG"

if [ "$NS" -ge 9000 ]; then
  python train/train_prior.py --config "$CFGP" --device cuda \
    --num-workers 8 --batch-size 16 >> "$LOG" 2>&1 \
    || echo "[FAIL] E21 prior train" >> "$LOG"
fi

if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E21 infer" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_aasp = $NP" >> "$LOG"

if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E21 --name E21_aasp --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E21 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E21_aasp/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E21_aasp/best.pth \
      --config configs/bcp_vis.yaml --exp E21 --tag E21 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 8 >> "$LOG" 2>&1 \
      || echo "[FAIL] E21 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e21_queue.done
    echo "[$(date)] E21 done" >> "$LOG"
  else
    echo "[$(date)] E21 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
fi
