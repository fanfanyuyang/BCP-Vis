#!/bin/bash
# E13b 队列：修正版 UAP（独立先验=litehr，非失效的 1-P_b）
#   Step1: 独立先验 UAP 推理 -> priors_uap_indep (cascade fg + litehr 独立 fg)
#   Step2: 训练最终 CNN E13b（在独立先验上）
#   Step3: 评测 -> E13b 行
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e13b.lock
flock -n 200 || { echo "E13b 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e13b_queue.log
echo "[$(date)] E13b 开始（修正版 UAP：独立先验=litehr）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFG_E=configs/bcp_vis.yaml
OUTDIR=datasets/EVD4UAV_processed/priors_uap_indep

# Step1: 独立先验 UAP 推理
python scripts/infer_uap_indep.py \
  --ckpt-fg runs/priornet/prior_v8_cascade/best.pth --arch-fg cascade --cfg-fg configs/prior_v8_cascade.yaml \
  --ckpt-ind runs/priornet/v3_litehr/best.pth --arch-ind litehr --cfg-ind configs/prior_v3_litehr.yaml \
  --proc "$PROC" --raw "$RAW" --out-dir "$OUTDIR" \
  --tile 1024 --overlap 192 --alpha 1.0 \
  >> "$LOG" 2>&1 || echo "[FAIL] E13b infer" >> "$LOG"

# Step2: 训练最终 CNN
python train/train_final.py --config "$CFG_E" --exp E13b --name E13b_uap_indep \
  --batch-size 16 --num-workers 8 --prior-dir "$OUTDIR" \
  >> "$LOG" 2>&1 || echo "[FAIL] E13b train" >> "$LOG"

# Step3: 评测
if [ -f runs/bcp_vis/E13b_uap_indep/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E13b_uap_indep/best.pth --config "$CFG_E" \
    --exp E13b --tag E13b --prior-dir "$OUTDIR" \
    --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] E13b eval" >> "$LOG"
  python eval/make_report.py >> "$LOG" 2>&1
  echo "[$(date)] E13b done" >> "$LOG"
fi

echo DONE > logs/e13b_queue.done
echo "[$(date)] E13b 队列 DONE" >> "$LOG"
