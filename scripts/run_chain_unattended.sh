#!/usr/bin/env bash
# ============================================================
# BCP-Vis 无人值守执行链（阶段 06 ~ 11）
#
# 设计目的：用 setsid + nohup 启动后，完全脱离本地终端。
# 用户关掉本地电脑 / 断开 SSH 后，服务器继续跑完所有步骤。
#
# 顺序（严格按执行说明第 44 节）：
#   06 make_patches
#   07 PriorNet V1 smoke test (2 epoch, 少量数据)
#   09 Lite-HR PriorNet (V3, 正式训练)
#   -- 用 V3 生成 predicted prior
#   10 RGB baseline (E0)
#   11 BCP-Vis (E5)
#
# 每步都会写 logs/chain_*.log；单步失败不会中断整条链。
# ============================================================
cd "<PROJECT_ROOT>"
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CHAIN=logs/chain
mkdir -p logs "$PROC/oof_priors"

echo "[$(date)] CHAIN START" | tee -a $CHAIN.log

# ---------- 等 soft prior 落盘完成 ----------
echo "[$(date)] waiting for soft priors ..." | tee -a $CHAIN.log
while pgrep -f build_soft_prior > /dev/null; do sleep 10; done
echo "[$(date)] soft priors ready: $(ls $PROC/soft_priors/*.png 2>/dev/null | wc -l)" | tee -a $CHAIN.log

# ---------- 06 patches ----------
echo "[$(date)] STEP 06 make_patches" | tee -a $CHAIN.log
python scripts/make_patches.py --root "$RAW" --proc "$PROC" \
    --patch 768 --overlap 192 --empty-keep-ratio 0.15 --n-vis 12 \
    > logs/chain_make_patches.log 2>&1 || echo "[FAIL] make_patches" | tee -a $CHAIN.log
python tools/check_patch_stats.py "$PROC/patch_manifest.csv" \
    >> logs/chain_make_patches.log 2>&1
tail -20 logs/chain_make_patches.log | tee -a $CHAIN.log

# ---------- 07 PriorNet V1 smoke ----------
echo "[$(date)] STEP 07 PriorNet V1 smoke (2 epoch)" | tee -a $CHAIN.log
python train/train_prior.py --config configs/prior_v1.yaml \
    --name v1_smoke --epochs 2 --limit 400 --batch-size 8 --num-workers 16 \
    > logs/chain_prior_v1_smoke.log 2>&1 || echo "[FAIL] prior v1 smoke" | tee -a $CHAIN.log
tail -25 logs/chain_prior_v1_smoke.log | tee -a $CHAIN.log

# ---------- 09 Lite-HR PriorNet (V3) ----------
echo "[$(date)] STEP 09 Lite-HR PriorNet V3" | tee -a $CHAIN.log
python train/train_prior.py --config configs/prior_v3_litehr.yaml \
    --name v3_litehr --batch-size 16 --num-workers 16 \
    > logs/chain_prior_v3.log 2>&1 || echo "[FAIL] prior v3" | tee -a $CHAIN.log
tail -15 logs/chain_prior_v3.log | tee -a $CHAIN.log

# ---------- 用 V3 生成 predicted prior ----------
echo "[$(date)] STEP infer prior (V3)" | tee -a $CHAIN.log
python scripts/infer_prior.py \
    --ckpt runs/priornet/v3_litehr/best.pth --arch litehr \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$PROC/oof_priors" --tile 1024 --overlap 192 \
    > logs/chain_infer_prior.log 2>&1 || echo "[FAIL] infer prior" | tee -a $CHAIN.log
tail -6 logs/chain_infer_prior.log | tee -a $CHAIN.log

# ---------- 10 RGB baseline (E0) ----------
echo "[$(date)] STEP 10 RGB baseline E0" | tee -a $CHAIN.log
python train/train_final.py --config configs/rgb_baseline.yaml \
    --exp E0 --name E0_rgb --batch-size 16 --num-workers 16 \
    > logs/chain_final_E0.log 2>&1 || echo "[FAIL] E0" | tee -a $CHAIN.log
tail -15 logs/chain_final_E0.log | tee -a $CHAIN.log

# ---------- 11 BCP-Vis (E5) ----------
echo "[$(date)] STEP 11 BCP-Vis E5" | tee -a $CHAIN.log
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E5 --name E5_bcp --batch-size 16 --num-workers 16 \
    --prior-dir "$PROC/oof_priors" \
    > logs/chain_final_E5.log 2>&1 || echo "[FAIL] E5" | tee -a $CHAIN.log
tail -15 logs/chain_final_E5.log | tee -a $CHAIN.log

echo "[$(date)] CHAIN END" | tee -a $CHAIN.log
