#!/usr/bin/env bash
# ============================================================
# BCP-Vis Phase-3：V4 Altitude-Aware PriorNet + E6
#
# 说明：本链在 phase2 全部结束后才启动，不阻塞主结果（E0/E5/E1/E2/E7/OOF）。
#
# 已修复的前置问题（否则 V4 会静默退化成 V3）：
#   1. PatchDataset(task="prior") 原先只返回 (rgb, target)，没有高度
#   2. train_prior 调 model(x) 不传 altitude
#      => AltitudeAwarePriorNet 走 altitude=None 降级分支，与普通 LiteHR 等价
#   3. infer_prior 同样需要传高度（现在从 split.csv 查）
#   已验证：零初始化 FiLM 下 50m/90m 输出差 = 0（初始恒等），
#          扰动 gamma 后差 = 5.86e-02（高度条件生效）
# ============================================================
cd /root/autodl-tmp/BCP-Vis
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CH=logs/phase3
mkdir -p logs

echo "[$(date)] PHASE3 START" | tee -a $CH.log
echo "[$(date)] waiting for phase2 ..." | tee -a $CH.log
while pgrep -f "run_chain_[p]hase2" > /dev/null; do sleep 60; done
echo "[$(date)] phase2 done" | tee -a $CH.log

# ---------- V4 PriorNet ----------
echo "[$(date)] STEP V4 Altitude-Aware PriorNet" | tee -a $CH.log
python train/train_prior.py --config configs/prior_v4_altitude.yaml \
    --name v4_altitude --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/prior_v4.log 2>&1 || echo "[FAIL] V4" | tee -a $CH.log
tail -6 logs/prior_v4.log | tee -a $CH.log

# ---------- V4 推理 ----------
V4DIR="$PROC/priors_v4"
mkdir -p "$V4DIR"
echo "[$(date)] STEP infer prior (V4)" | tee -a $CH.log
python scripts/infer_prior.py \
    --ckpt runs/priornet/v4_altitude/best.pth --arch altitude \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$V4DIR" --tile 1024 --overlap 192 \
    > logs/infer_prior_v4.log 2>&1 || echo "[FAIL] infer v4" | tee -a $CH.log
echo "v4 prior files: $(ls $V4DIR/*.png 2>/dev/null | wc -l)" | tee -a $CH.log

# ---------- E6 ----------
echo "[$(date)] STEP E6 (RGB + V4 altitude prior)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E6 --name E6_altitude --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$V4DIR" \
    > logs/final_E6.log 2>&1 || echo "[FAIL] E6" | tee -a $CH.log
tail -6 logs/final_E6.log | tee -a $CH.log

echo "[$(date)] STEP eval E6" | tee -a $CH.log
python eval/eval_final.py \
    --ckpt runs/bcp_vis/E6_altitude/best.pth --config configs/bcp_vis.yaml \
    --exp E6 --tag E6 --proc "$PROC" --raw "$RAW" \
    --prior-dir "$V4DIR" --batch-size 8 --num-workers 8 \
    > logs/eval_E6.log 2>&1 || echo "[FAIL] eval E6" | tee -a $CH.log
tail -3 logs/eval_E6.log | tee -a $CH.log

python eval/run_ablation.py --merge-only \
    > logs/ablation_v4.log 2>&1 || echo "[FAIL] ablation" | tee -a $CH.log
tail -20 logs/ablation_v4.log | tee -a $CH.log

echo "[$(date)] PHASE3 END" | tee -a $CH.log
