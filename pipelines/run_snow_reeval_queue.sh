#!/bin/bash
# 雪控制重评：对关键既有实验用"新增 by_size_snow 分组"的 eval 重评，
# 以便把"高度效应"与"雪效应"解耦（by_size_snow 只对补丁之后的评测生效）。
# 等 E29 完成后接手，串行不抢卡。只读 ckpt、幂等（同 tag 覆盖旧行）。
cd "<PROJECT_ROOT>"
exec 200>/tmp/bcp_snow.lock
flock -n 200 || { echo "snow reeval 已在运行，退出"; exit 0; }
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base
LOG=logs/snow_reeval_queue.log
PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
echo "[$(date)] 雪控制重评队列启动，等 logs/e29_queue.done" >> "$LOG"
while [ ! -f logs/e29_queue.done ]; do sleep 120; done
echo "[$(date)] 上游 E29 完成，开始雪控制重评" >> "$LOG"

ev () {   # ev <ckpt> <config> <exp> <tag> <prior_dir>
  ck="$1"; cfg="$2"; exp="$3"; tag="$4"; pd="$5"
  if [ ! -f "$ck" ]; then echo "[SKIP] $tag ckpt 缺失: $ck" >> "$LOG"; return; fi
  cmd="python eval/eval_final.py --ckpt $ck --config $cfg --exp $exp --tag $tag \
       --proc $PROC --raw $RAW --batch-size 8 --num-workers 8"
  if [ -n "$pd" ]; then cmd="$cmd --prior-dir $pd"; fi
  eval "$cmd" >> "$LOG" 2>&1 || echo "[FAIL] $tag" >> "$LOG"
  echo "[$(date)] $tag 重评完成" >> "$LOG"
}

ev runs/rgb_baseline/E0_rgb/best.pth    configs/rgb_baseline.yaml E0  E0  ""
ev runs/bcp_vis/E5_bcp/best.pth         configs/bcp_vis.yaml      E5  E5  "$PROC/oof_priors"
ev runs/bcp_vis/E6_altitude/best.pth    configs/bcp_vis.yaml      E6  E6  "$PROC/priors_v4"
ev runs/bcp_vis/E12_cascade/best.pth    configs/bcp_vis.yaml      E12 E12 "$PROC/priors_v8"
# E20 双先验：eval_final 无 --prior-dir2 参数，必须靠 config(内含 prior_dir/prior_dir2)
ev runs/bcp_vis/E20_dual/best.pth       configs/bcp_vis_dual.yaml E20 E20 ""

python eval/make_report.py >> "$LOG" 2>&1
echo DONE > logs/snow_reeval.done
echo "[$(date)] 雪控制重评完成，by_size_snow.csv 已生成" >> "$LOG"
