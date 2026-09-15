#!/bin/bash
# V4 Altitude-Aware PriorNet 训练（数据驱动的核心算法迭代）
# 动机：交叉实测 tiny@50m +0.081 / tiny@70m +0.042 / tiny@90m -0.001
#       ⇒ 固定 sigma 在高空相对极小目标过大，先验过度膨胀淹没目标
#       ⇒ V4 让网络自学 sigma(h)（FiLM 高度条件调制），改善 90m
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_v4.lock
flock -n 200 || { echo "V4 already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
# 资源：容器 CPU 配额 16 核 → workers=8（不过度争抢）；
#       GPU 24G 空闲 → batch=16（与 V3 一致，保证可比）
python train/train_prior.py --config configs/prior_v4_altitude.yaml \
  --device cuda --num-workers 8 --batch-size 16
echo DONE > logs/train_v4.done
