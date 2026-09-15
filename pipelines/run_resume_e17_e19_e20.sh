#!/bin/bash
# 扩容重启后续跑：E17(logit 融合) -> E19(一致性正则) -> E20(双先验通道)
# 断点续跑：E17 若先验不全则补推理；各步产物存在才写 done（防假完成）
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_resume.lock
flock -n 200 || { echo "resume 队列已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/resume_queue.log
echo "[$(date)] ===== resume 队列启动（E17/E19/E20）=====" >> "$LOG"
df -h /root/autodl-tmp >> "$LOG" 2>&1

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV

# ---------- E17: logit 证据融合 ----------
N=$(ls "$PROC/priors_v8_logit"/*.png 2>/dev/null | wc -l)
echo "[$(date)] E17 logit priors = $N" >> "$LOG"
if [ "$N" -lt 9300 ]; then
  echo "[$(date)] 补齐 priors_v8_logit" >> "$LOG"
  python scripts/infer_prior.py --ckpt runs/priornet/prior_v8_cascade/best.pth \
    --arch cascade --model-config configs/prior_v8_cascade.yaml \
    --prior-mode logit --alpha 1.0 \
    --proc "$PROC" --raw "$RAW" --out-dir "$PROC/priors_v8_logit" \
    --tile 1024 --overlap 192 >> "$LOG" 2>&1 || echo "[FAIL] resume E17 infer" >> "$LOG"
fi
N=$(ls "$PROC/priors_v8_logit"/*.png 2>/dev/null | wc -l)
if [ "$N" -ge 9300 ]; then
  python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E17 --name E17_logit --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8_logit" \
    >> "$LOG" 2>&1 || echo "[FAIL] resume E17 train" >> "$LOG"
  if [ -f runs/bcp_vis/E17_logit/best.pth ]; then
    python eval/eval_final.py --ckpt runs/bcp_vis/E17_logit/best.pth \
      --config configs/bcp_vis.yaml --exp E17 --tag E17 \
      --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8_logit" \
      --batch-size 8 --num-workers 8 >> "$LOG" 2>&1 || echo "[FAIL] resume E17 eval" >> "$LOG"
    echo "[$(date)] E17 done" >> "$LOG"
  fi
fi

# ---------- E19: 先验一致性正则 ----------
if [ ! -f runs/bcp_vis/E19_pc/best.pth ]; then
  python train/train_final.py --config configs/bcp_vis_pc.yaml \
    --exp E19 --name E19_pc --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8" \
    >> "$LOG" 2>&1 || echo "[FAIL] resume E19 train" >> "$LOG"
fi
if [ -f runs/bcp_vis/E19_pc/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E19_pc/best.pth \
    --config configs/bcp_vis_pc.yaml --exp E19 --tag E19 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/priors_v8" \
    --batch-size 8 --num-workers 8 >> "$LOG" 2>&1 || echo "[FAIL] resume E19 eval" >> "$LOG"
  echo "[$(date)] E19 done" >> "$LOG"
fi

# ---------- E20: 双先验通道 ----------
NPF=$(ls "$PROC/priors_v8_pf"/*.png 2>/dev/null | wc -l)
NPB=$(ls "$PROC/priors_v8_pb"/*.png 2>/dev/null | wc -l)
echo "[$(date)] E20 priors pf=$NPF pb=$NPB" >> "$LOG"
if [ "$NPF" -ge 9000 ] && [ "$NPB" -ge 9000 ] && [ ! -f runs/bcp_vis/E20_dual/best.pth ]; then
  python train/train_final.py --config configs/bcp_vis_dual.yaml \
    --exp E20 --name E20_dual --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" \
    >> "$LOG" 2>&1 || echo "[FAIL] resume E20 train" >> "$LOG"
fi
if [ -f runs/bcp_vis/E20_dual/best.pth ]; then
  python eval/eval_final.py --ckpt runs/bcp_vis/E20_dual/best.pth \
    --config configs/bcp_vis_dual.yaml --exp E20 --tag E20 \
    --proc "$PROC" --raw "$RAW" --batch-size 8 --num-workers 8 \
    >> "$LOG" 2>&1 || echo "[FAIL] resume E20 eval" >> "$LOG"
  echo "[$(date)] E20 done" >> "$LOG"
fi

python eval/make_report.py >> "$LOG" 2>&1
echo "[$(date)] ===== resume 队列结束 =====" >> "$LOG"
echo DONE > logs/resume_queue.done
