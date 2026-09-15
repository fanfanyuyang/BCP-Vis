#!/bin/bash
# E24 = V8 级联 + FiLM 高度条件（height_mode: film + 背景专家头）
#
# 动机（2026-09-13 数据驱动）：
#   - E12(V8, moe) 在 90m 低于无先验 E0（0.7661 < 0.7819）→ 高空净损害
#   - E6(V4, FiLM) 唯一修好 90m（0.7957 > E0），50m / tiny 总体亦最佳
#   - E21(AASP)、E22(双尺度) 硬编码 sigma(h) 均失败 ⇒ 不硬编码 sigma，改用学习式 FiLM
# 本实验 = priornet.py 里从未测过的组合："film" + 背景头（E6=film无背景头，E12=moe有背景头）
#
# 与 E12 唯一差异：PriorNet 的 height_mode moe -> film。
# 训练目标为基线 soft prior（默认），下游 CNN 配置、损失、epochs、seed 全部与 E12 一致。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e24.lock
flock -n 200 || { echo "E24 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e24_queue.log
echo "[$(date)] E24(FiLM+背景头) 队列启动，等 logs/e23_queue.done" >> "$LOG"
while [ ! -f logs/e23_queue.done ]; do sleep 120; done
echo "[$(date)] 上游完成，开始 E24（V8 级联 + FiLM 高度条件）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CFGP=configs/prior_v8_cascade_film.yaml
CKPT=runs/priornet/prior_v8_cascade_film/best.pth
PD="$PROC/priors_v8_film"

# 1) 训练级联 PriorNet（FiLM 高度条件；目标=基线 soft prior，与 E12 同口径）
python train/train_prior.py --config "$CFGP" --device cuda \
  --num-workers 24 --batch-size 16 >> "$LOG" 2>&1 \
  || echo "[FAIL] E24 prior train" >> "$LOG"
echo "[$(date)] E24 prior train 结束" >> "$LOG"

# 2) 推理先验（mul 融合，与 E12 同口径；注意必须传 --model-config，ckpt 不存 cfg）
if [ -f "$CKPT" ]; then
  python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config "$CFGP" --proc "$PROC" --raw "$RAW" \
    --out-dir "$PD" --tile 1024 --overlap 192 >> "$LOG" 2>&1 \
    || echo "[FAIL] E24 infer" >> "$LOG"
else
  echo "[FAIL] E24 无 ckpt: $CKPT" >> "$LOG"
fi
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_film = $NP" >> "$LOG"

# 3) 最终 CNN 训练 + 评测
if [ "$NP" -ge 9000 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E24 --name E24_film --batch-size 16 --num-workers 24 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PD" >> "$LOG" 2>&1 \
    || echo "[FAIL] E24 final train" >> "$LOG"
  if [ -f runs/bcp_vis/E24_film/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E24_film/best.pth \
      --config configs/bcp_vis.yaml --exp E24 --tag E24 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PD" \
      --batch-size 8 --num-workers 24 >> "$LOG" 2>&1 \
      || echo "[FAIL] E24 eval" >> "$LOG"
    python eval/make_report.py >> "$LOG" 2>&1
    echo DONE > logs/e24_queue.done
    echo "[$(date)] E24 done" >> "$LOG"
  else
    echo "[$(date)] E24 未产出 best.pth，不写 done（防假完成）" >> "$LOG"
  fi
else
  echo "[FAIL] E24 先验数量不足: $NP" >> "$LOG"
fi
