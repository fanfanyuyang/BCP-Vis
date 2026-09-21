#!/usr/bin/env bash
# ============================================================
# BCP-Vis Phase-2 无人值守执行链
#
# 背景：原 chain1 在 infer_prior 阶段写 float32 .npy（8.3MB/张 × 9371 = 78GB）
#       撑爆 50GB 数据盘，已改为 uint8 PNG（147KB/张，总量 ~1.4GB）。
#       同时原 chain1 / chain2 的 bash 父进程被终止，本链重新编排剩余步骤。
#
# 顺序：
#   0) 等待当前正在跑的 E0 (RGB baseline) 结束
#   1) 用 V3 Lite-HR PriorNet 生成 predicted prior（uint8 png）
#   2) E5  BCP-Vis      (RGB + predicted prior)
#   3) E1  RGB+全零通道  / E2 RGB+随机通道 / E7 Oracle 上界
#   4) 统一评估（总指标 + 按高度 + 按目标尺寸）
#   5) OOF 5-fold prior -> E5_OOF（论文级，放在最后，不阻塞主结果）
#
# 单步失败不中断整条链。
# ============================================================
cd "<PROJECT_ROOT>"
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CH=logs/phase2
mkdir -p logs "$PROC/oof_priors"

echo "[$(date)] PHASE2 START" | tee -a $CH.log

# ---------- 0) E0 RGB baseline ----------
# 原先这里是「等待遗留的 E0 孤儿进程结束」。遗留 E0 有两个问题：
#   (a) 未启用 channels_last / cudnn.benchmark，比优化后慢约 33%；
#   (b) 曾与同为孤儿的 E1 争抢同一张 GPU（GPU 已饱和，并发无吞吐收益）。
# 故改为**重新训练**，保证 E0 与后续 E5/E1/E2/E7 完全同条件、可比。
echo "[$(date)] STEP E0 (RGB baseline)" | tee -a $CH.log
python train/train_final.py --config configs/rgb_baseline.yaml \
    --exp E0 --name E0_rgb --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/final_E0.log 2>&1 || echo "[FAIL] E0" | tee -a $CH.log
tail -8 logs/final_E0.log | tee -a $CH.log

# ---------- 1) 生成 predicted prior ----------
echo "[$(date)] STEP infer prior (V3, uint8 png)" | tee -a $CH.log
python scripts/infer_prior.py \
    --ckpt runs/priornet/v3_litehr/best.pth --arch litehr \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$PROC/oof_priors" --tile 1024 --overlap 192 \
    > logs/infer_prior_v3.log 2>&1 || echo "[FAIL] infer prior" | tee -a $CH.log
echo "prior files: $(ls $PROC/oof_priors/*.png 2>/dev/null | wc -l)" | tee -a $CH.log
df -h <PROJECT_ROOT> | tail -1 | tee -a $CH.log

run_eval () {
    local exp=$1 ckpt=$2 extra=$3 cfg=$4 tag=$5
    tag=${tag:-$exp}
    echo "[$(date)] eval $exp ($ckpt) tag=$tag" | tee -a $CH.log
    python eval/eval_final.py \
        --ckpt "$ckpt" --config "configs/$cfg" --exp "$exp" --tag "$tag" \
        --proc "$PROC" --raw "$RAW" $extra --batch-size 8 --num-workers 8 \
        > "logs/eval_$tag.log" 2>&1 || echo "[FAIL] eval $tag" | tee -a $CH.log
    tail -3 "logs/eval_$tag.log" | tee -a $CH.log
}

# ---------- 2) E5 BCP-Vis ----------
echo "[$(date)] STEP E5 (RGB + predicted prior)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E5 --name E5_bcp --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/oof_priors" \
    > logs/final_E5.log 2>&1 || echo "[FAIL] E5" | tee -a $CH.log
tail -8 logs/final_E5.log | tee -a $CH.log
run_eval E0 runs/rgb_baseline/E0_rgb/best.pth "" rgb_baseline.yaml
run_eval E5 runs/bcp_vis/E5_bcp/best.pth "--prior-dir $PROC/oof_priors" bcp_vis.yaml

# ---------- 3) 对照组 ----------
echo "[$(date)] STEP E1 (zero prior)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E1 \
    --name E1_zero --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/final_E1.log 2>&1 || echo "[FAIL] E1" | tee -a $CH.log
tail -4 logs/final_E1.log | tee -a $CH.log
run_eval E1 runs/bcp_vis/E1_zero/best.pth "" bcp_vis.yaml

echo "[$(date)] STEP E2 (random prior)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E2 \
    --name E2_random --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/final_E2.log 2>&1 || echo "[FAIL] E2" | tee -a $CH.log
tail -4 logs/final_E2.log | tee -a $CH.log
run_eval E2 runs/bcp_vis/E2_random/best.pth "" bcp_vis.yaml

echo "[$(date)] STEP E7 (oracle GT prior, 仅理论上界)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E7 \
    --name E7_oracle --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/final_E7.log 2>&1 || echo "[FAIL] E7" | tee -a $CH.log
tail -4 logs/final_E7.log | tee -a $CH.log
run_eval E7 runs/bcp_vis/E7_oracle/best.pth "" bcp_vis.yaml

# ---------- 4) 消融汇总 ----------
echo "[$(date)] STEP ablation summary" | tee -a $CH.log
python eval/run_ablation.py --merge-only \
    > logs/ablation.log 2>&1 || echo "[FAIL] ablation" | tee -a $CH.log
tail -20 logs/ablation.log | tee -a $CH.log

# ---------- 5) OOF 5-fold prior（论文级） ----------
echo "[$(date)] STEP OOF 5-fold prior" | tee -a $CH.log
python scripts/generate_oof_prior.py \
    --proc "$PROC" --raw "$RAW" --folds 5 --seed 42 \
    --base-config configs/prior_v3_litehr.yaml --prior-cfg-name litehr \
    --batch-size 16 --num-workers 16 \
    > logs/oof_prior.log 2>&1 || echo "[FAIL] oof prior" | tee -a $CH.log
echo "oof prior files: $(ls $PROC/oof_priors/*.png 2>/dev/null | wc -l)" | tee -a $CH.log
tail -10 logs/oof_prior.log | tee -a $CH.log

echo "[$(date)] STEP E5_OOF (RGB + OOF prior)" | tee -a $CH.log
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E5 --name E5_oof --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$PROC/oof_priors" \
    > logs/final_E5_oof.log 2>&1 || echo "[FAIL] E5_oof" | tee -a $CH.log
tail -8 logs/final_E5_oof.log | tee -a $CH.log
# --exp 必须是 E0..E7 之一（eval_final 用 modes[exp] 查表，否则 KeyError 崩溃）；
# 自定义名字只能通过第 5 个参数 --tag 传。
run_eval E5 runs/bcp_vis/E5_oof/best.pth "--prior-dir $PROC/oof_priors" bcp_vis.yaml E5_OOF

# ---------- 6) 最终汇总（含 E5_OOF） ----------
echo "[$(date)] STEP final ablation" | tee -a $CH.log
python eval/run_ablation.py --merge-only \
    > logs/ablation_final.log 2>&1 || echo "[FAIL] ablation final" | tee -a $CH.log
tail -25 logs/ablation_final.log | tee -a $CH.log

echo "[$(date)] PHASE2 END" | tee -a $CH.log
