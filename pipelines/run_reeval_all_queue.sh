#!/bin/bash
# 修复 eval_final 的 BatchNorm 污染 bug 后，**全量重评**所有既有实验。
# 该 bug 让所有评测值被系统性低估（实测 E12 +0.0045、E16 +0.0755）。
# 关键的排前面，保证即使时间不够也先拿到核心对照。
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_reeval.lock
flock -n 200 || { echo "reeval 已在运行，退出"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
LOG=logs/reeval_all_queue.log
PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
echo "[$(date)] 全量重评队列启动，等 logs/resume_queue.done" >> "$LOG"
while [ ! -f logs/resume_queue.done ]; do sleep 120; done
echo "[$(date)] 开始全量重评（BN 修复后）" >> "$LOG"

ev () {   # ev <ckpt> <config> <exp> <tag> <prior_dir> [prior_dir2]
  ck="$1"; cfg="$2"; exp="$3"; tag="$4"; pd="$5"; pd2="$6"
  if [ ! -f "$ck" ]; then echo "[SKIP] $tag ckpt 缺失" >> "$LOG"; return; fi
  cmd="python eval/eval_final.py --ckpt $ck --config $cfg --exp $exp --tag $tag \
       --proc $PROC --raw $RAW --batch-size 8 --num-workers 8"
  if [ -n "$pd" ]; then cmd="$cmd --prior-dir $pd"; fi
  if [ -n "$pd2" ]; then cmd="$cmd --prior-dir2 $pd2"; fi
  eval "$cmd" >> "$LOG" 2>&1 || echo "[FAIL] $tag" >> "$LOG"
  echo "[$(date)] $tag 重评完成" >> "$LOG"
}

# ---- 核心对照（基线 / 因果链 / 主结果）----
ev runs/rgb_baseline/E0_rgb/best.pth            configs/rgb_baseline.yaml E0    E0       ""
ev runs/bcp_vis/E1_zero/best.pth                configs/bcp_vis.yaml      E1    E1       ""
ev runs/bcp_vis/E2_random/best.pth              configs/bcp_vis.yaml      E2    E2       ""
ev runs/bcp_vis/E5_bcp/best.pth                 configs/bcp_vis.yaml      E5    E5       "$PROC/oof_priors"
ev runs/bcp_vis/E5_oof/best.pth                 configs/bcp_vis.yaml      E5    E5_OOF   "$PROC/oof_priors"
ev runs/bcp_vis/E6_altitude/best.pth            configs/bcp_vis.yaml      E6    E6       "$PROC/priors_v4"
ev runs/bcp_vis/E8_dual/best.pth                configs/bcp_vis.yaml      E8    E8       "$PROC/priors_v5d"

# ---- 新实验（修复前评的，需重评）----
ev runs/bcp_vis/E13_uap/best.pth                configs/bcp_vis.yaml      E13   E13      "$PROC/priors_uap"
ev runs/bcp_vis/E13b_uap_indep/best.pth         configs/bcp_vis.yaml      E13b  E13b     "$PROC/priors_uap_indep"
ev runs/bcp_vis/E17_logit/best.pth              configs/bcp_vis.yaml      E17   E17      "$PROC/priors_v8_logit"
ev runs/bcp_vis/E12_cascade/best.pth            configs/bcp_vis.yaml      E18   E18      "$PROC/priors_v8_misaligned"

# ---- 消融/次要 ----
ev runs/bcp_vis/E14_cat/best.pth                configs/bcp_vis.yaml      E14   E14      "$PROC/priors_cat"
ev runs/bcp_vis/E9_moe/best.pth                 configs/bcp_vis.yaml      E9    E9       "$PROC/priors_v5moe"
ev runs/bcp_vis/E10_large/best.pth              configs/bcp_vis.yaml      E10   E10      "$PROC/priors_large"
ev runs/bcp_vis/E11_altmoe/best.pth             configs/bcp_vis.yaml      E11   E11      "$PROC/priors_v7"
ev runs/bcp_vis/E7_oracle/best.pth              configs/bcp_vis.yaml      E7    E7       ""

python eval/make_report.py >> "$LOG" 2>&1
echo DONE > logs/reeval_all_queue.done
echo "[$(date)] 全量重评完成" >> "$LOG"
