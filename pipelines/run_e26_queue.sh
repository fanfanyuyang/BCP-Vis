#!/bin/bash
# E26 = V8 级联 + MoE 高度条件 + 无背景头（lambda_bg=0）。
# 与 E12(moe+有背景头)、E24(film+有背景头)、E25(film+无背景头) 构成 2×2 因子消融。
# 触发：等 E24 或 E25 任一完成（释放 GPU 配额）即接手，保持 GPU 不空转。
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_e26.lock
flock -n 200 || { echo "E26 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/e26_queue.log
echo "[$(date)] E26(MoE 无背景头) 队列启动，等 logs/e25_queue.done（串行以保持 GPU 满载）" >> "$LOG"
while [ ! -f logs/e25_queue.done ]; do sleep 120; done
echo "[$(date)] 上游 E25 完成，开始 E26" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFGP=configs/prior_v8_cascade_moe_nobg.yaml
CKPT=runs/priornet/prior_v8_cascade_moe_nobg/best.pth
PD="$PROC/priors_v8_moe_nobg"

# 1) 训练 PriorNet（moe，无背景头监督）
python train/train_prior.py --config "$CFGP" --device cuda \
  --num-workers 16 --batch-size 16 >> "$LOG" 2>&1 \
  || echo "[FAIL] E26 prior train" >> "$LOG"
echo "[$(date)] E26 prior train 结束" >> "$LOG"

# 2) 推理先验（★必须 --prior-mode raw：本实验 lambda_bg=0，背景头未训练，
#    用默认 mul 会拿未训练的背景头污染先验）
if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --prior-mode raw \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E26 infer" >> "$LOG"
else
  echo "[FAIL] E26 无 ckpt: $CKPT" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_moe_nobg = $NP" >> "$LOG"

# 3) 下游 Final CNN + 评测
if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E26 --name E26_moe_nobg --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E26 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E26_moe_nobg/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E26_moe_nobg/best.pth \
      --config configs/bcp_vis.yaml --exp E26 --tag E26 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E26 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e26_queue.done
    echo "[$(date)] E26 done" >> "$LOG"
  else
    echo "[$(date)] E26 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E26 先验不足: $NP" >> "$LOG"
fi
