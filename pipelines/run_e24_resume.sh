#!/bin/bash
# E24 下游补跑：PriorNet 与先验已就绪（priors_v8_film=9371），仅补 train_final + eval。
#
# 背景：07:38 E24 下游因 train_final.py 的 --exp 白名单未含 E24 而报错退出（防假完成未写 done）。
# 已用 _patch_e24.py 注册 E24；本脚本从"下游训练"这一步续跑，不重训 PriorNet、不重推理先验。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e24r.lock
flock -n 200 || { echo "E24 resume 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/e24_queue.log
echo "[$(date)] E24 下游补跑启动（PriorNet 与 9371 张先验已就绪）" >> "$LOG"

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
PD="$PROC/priors_v8_film"
NP=$(ls "$PD"/*.png 2>/dev/null | wc -l)
echo "[$(date)] priors_v8_film = $NP" >> "$LOG"

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
  echo "[FAIL] E24 先验不足: $NP" >> "$LOG"
fi
