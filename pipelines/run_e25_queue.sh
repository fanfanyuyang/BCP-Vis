#!/bin/bash
# E25 = V8 级联 + FiLM 高度条件 + 无背景头（lambda_bg=0）。
# 立即启动：与 E24 下游并行，利用当前 GPU 余量（带宽 74%、功耗 76%、显存剩 15G）。
# 与 E24 唯一差异 = lambda_bg 1.0 -> 0.0 ⇒ 干净隔离"背景头"的独立贡献（创新点②缺口）。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e25.lock
flock -n 200 || { echo "E25 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e25_queue.log
echo "[$(date)] E25(FiLM 无背景头) 队列启动，等 logs/e24_queue.done（串行以保持 GPU 满载）" >> "$LOG"
while [ ! -f logs/e24_queue.done ]; do sleep 120; done
echo "[$(date)] 上游 E24 完成，开始 E25" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFGP=configs/prior_v8_cascade_film_nobg.yaml
CKPT=runs/priornet/prior_v8_cascade_film_nobg/best.pth
PD="$PROC/priors_v8_film_nobg"

# 1) 训练 PriorNet（film，无背景头监督）
python train/train_prior.py --config "$CFGP" --device cuda \
  --num-workers 16 --batch-size 16 >> "$LOG" 2>&1 \
  || echo "[FAIL] E25 prior train" >> "$LOG"
echo "[$(date)] E25 prior train 结束" >> "$LOG"

# 2) 推理先验（★必须 --prior-mode raw：本实验 lambda_bg=0，背景头未训练，
#    若用默认 mul=P_f*(1-P_b) 会拿未训练的背景头污染先验，失去"无背景头"语义）
if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --prior-mode raw \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E25 infer" >> "$LOG"
else
  echo "[FAIL] E25 无 ckpt: $CKPT" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_film_nobg = $NP" >> "$LOG"

# 3) 下游 Final CNN + 评测
if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E25 --name E25_film_nobg --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E25 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E25_film_nobg/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E25_film_nobg/best.pth \
      --config configs/bcp_vis.yaml --exp E25 --tag E25 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 16 >> "$LOG" 2>&1 \
      || echo "[FAIL] E25 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e25_queue.done
    echo "[$(date)] E25 done" >> "$LOG"
  else
    echo "[$(date)] E25 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E25 先验不足: $NP" >> "$LOG"
fi
