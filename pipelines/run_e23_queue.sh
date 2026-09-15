#!/bin/bash
# E23 = OOF 严谨性（5 折 cascade PriorNet；E5_OOF 口径：OOF 先验用于评测，单一最终 CNN）
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e23.lock
flock -n 200 || { echo "E23 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e23_queue.log
echo "[$(date)] E23(OOF) 队列启动，等 logs/e22_queue.done" >> "$LOG"
while [ ! -f logs/e22_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E23（5 折 OOF cascade 先验）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD="$PROC/priors_v8_oof"
CFG=configs/prior_v8_cascade.yaml

# 1) 5 折 OOF：对 held-out train 折推理（生成它的模型从未见过该样本）
python scripts/generate_oof_prior.py \
  --base-config "$CFG" --prior-cfg-name cascade --prior-mode mul \
  --model-config "$CFG" --out-dir "$PD" \
  --proc "$PROC" --raw "$RAW" \
  --batch-size 16 --num-workers 8 --epochs 40 --folds 5 \
  >> "$LOG" 2>&1 || echo "[FAIL] E23 oof gen" >> "$LOG"

# 2) val/test 用全量 cascade ckpt 推理（PriorNet 本就未见过 val/test ⇒ 已 OOF）
for sp in val test; do
  python scripts/infer_prior.py --ckpt runs/priornet/prior_v8_cascade/best.pth \
    --arch cascade --model-config "$CFG" --split "$sp" --prior-mode mul \
    --proc "$PROC" --raw "$RAW" --out-dir "$PD" --tile 1024 --overlap 192 \
    >> "$LOG" 2>&1 || echo "[FAIL] E23 $sp prior" >> "$LOG"
done
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_oof = $NP" >> "$LOG"

# 3) E12 ckpt 在 OOF 先验上评测（与 E5↔E5_OOF 同口径）
if [ -f runs/bcp_vis/E12_cascade/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E12_cascade/best.pth \
    --config configs/bcp_vis.yaml --exp E23 --tag E23 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
    --batch-size 8 --num-workers 8 >> "$LOG" 2>&1 \
    || echo "[FAIL] E23 eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo DONE > logs/e23_queue.done
  echo "[$(date)] E23 done" >> "$LOG"
fi
