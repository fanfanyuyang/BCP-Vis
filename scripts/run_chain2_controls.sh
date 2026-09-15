#!/usr/bin/env bash
# ============================================================
# 对照组链（等主链 run_chain_unattended.sh 结束后自动接上）
#
# 为什么必须有这些对照：
#   E0 RGB          真正 baseline
#   E1 RGB+全零通道  排除"只是多一通道"带来的容量提升
#   E2 RGB+随机通道  排除"任意第 4 通道"的正则化效应
#   E5 RGB+BCP      主实验
#   E7 RGB+GT mask  理论上界（仅参考，严禁当部署方法）
#
# 结构公平性：E0/E1/E2/E5/E7 共用同一 ResNet18-U-Net，
# 唯一区别是 first conv 的 in_channels 3 vs 4；
# lr / epochs / batch / split / 增强 / seed 全部一致（由 yaml 保证）。
# ============================================================
cd /root/autodl-tmp/BCP-Vis
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CHAIN2=logs/chain2
mkdir -p logs

echo "[$(date)] CHAIN2 START" | tee -a $CHAIN2.log

# 等主链结束
echo "[$(date)] waiting for chain1 ..." | tee -a $CHAIN2.log
while pgrep -f run_chain_unattended > /dev/null; do sleep 30; done
echo "[$(date)] chain1 finished" | tee -a $CHAIN2.log

run_eval () {
    local exp=$1 name=$2 extra=$3
    echo "[$(date)] eval $exp ($name)" | tee -a $CHAIN2.log
    python eval/eval_final.py \
        --ckpt "runs/$name/best.pth" --config configs/$4 --exp "$exp" --tag "$exp" \
        --proc "$PROC" --raw "$RAW" $extra --batch-size 8 --num-workers 8 \
        >> logs/chain2.log 2>&1 || echo "[FAIL] eval $exp" | tee -a $CHAIN2.log
}

# E1: RGB + 全零通道
echo "[$(date)] STEP E1 (zero prior)" | tee -a $CHAIN2.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E1 \
    --name E1_zero --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/chain2_E1.log 2>&1 || echo "[FAIL] E1" | tee -a $CHAIN2.log
tail -6 logs/chain2_E1.log | tee -a $CHAIN2.log
run_eval E1 E1_zero "" bcp_vis.yaml

# E2: RGB + 随机通道
echo "[$(date)] STEP E2 (random prior)" | tee -a $CHAIN2.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E2 \
    --name E2_random --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/chain2_E2.log 2>&1 || echo "[FAIL] E2" | tee -a $CHAIN2.log
tail -6 logs/chain2_E2.log | tee -a $CHAIN2.log
run_eval E2 E2_random "" bcp_vis.yaml

# E7: Oracle 上界（仅供理论上界参考）
echo "[$(date)] STEP E7 (oracle GT prior, 仅参考)" | tee -a $CHAIN2.log
python train/train_final.py --config configs/bcp_vis.yaml --exp E7 \
    --name E7_oracle --batch-size 16 --num-workers 16 \
    --proc "$PROC" --raw "$RAW" \
    > logs/chain2_E7.log 2>&1 || echo "[FAIL] E7" | tee -a $CHAIN2.log
tail -6 logs/chain2_E7.log | tee -a $CHAIN2.log
run_eval E7 E7_oracle "" bcp_vis.yaml

# 评估主链产出的 E0 / E5
run_eval E0 rgb_baseline/E0_rgb "" rgb_baseline.yaml
run_eval E5 bcp_vis/E5_bcp "--prior-dir $PROC/oof_priors" bcp_vis.yaml

echo "[$(date)] CHAIN2 END" | tee -a $CHAIN2.log
