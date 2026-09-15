#!/bin/bash
# 统一重跑：干净评估 + 尺寸×高度交叉 + 生成统一报告
cd /root/autodl-tmp/BCP-Vis
# ★安全闸门：本脚本会 rm -f reports/*.csv（清空全部评测结果）并 pkill eval_final.py。
# 误跑 = 结果全丢。必须显式加 --force 才会执行。
if [ "${1:-}" != "--force" ]; then
  echo "!! 危险：本脚本会清空 reports/*.csv 并杀掉 eval_final.py。"
  echo "!! 确认要清空重跑请执行: bash run_eval_all.sh --force"
  exit 1
fi
# 防并发锁：若已有实例在运行，本次立即退出（幂等，避免多实例互相 rm CSV）
exec 200>/tmp/bcp_eval.lock
flock -n 200 || { echo "another instance running, exit"; exit 0; }
pkill -f eval_final.py 2>/dev/null || true
sleep 2
rm -f reports/final_metrics.csv reports/by_size.csv reports/by_altitude.csv \
      reports/by_snow.csv reports/by_size_altitude.csv logs/eval_all2.done
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
for spec in \
  "E0:runs/rgb_baseline/E0_rgb/best.pth:configs/rgb_baseline.yaml" \
  "E1:runs/bcp_vis/E1_zero/best.pth:configs/bcp_vis.yaml" \
  "E2:runs/bcp_vis/E2_random/best.pth:configs/bcp_vis.yaml" \
  "E5:runs/bcp_vis/E5_bcp/best.pth:configs/bcp_vis.yaml:--prior-dir datasets/EVD4UAV_processed/oof_priors" \
  "E7:runs/bcp_vis/E7_oracle/best.pth:configs/bcp_vis.yaml"; do
  IFS=":" read e ck cfg pd <<< "$spec"
  echo "=== $e ==="
  python eval/eval_final.py --ckpt "$ck" --config "$cfg" --exp "$e" --tag "$e" $pd
done
python eval/make_report.py
echo DONE > logs/eval_all2.done
