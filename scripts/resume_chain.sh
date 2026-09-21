#!/usr/bin/env bash
# ============================================================
# BCP-Vis 幂等续跑链
#
# 设计：每一步先检查「是否已完整完成」，完成则跳过，未完成才跑。
# 因此可安全重复执行，也可在任一时刻被中断后由 tools/watchdog.py 重新拉起。
#
# 完成判据：
#   训练   metrics.csv 行数-1 >= $EPOCHS 且 best.pth 存在
#   先验   oof_priors/*.png 数量 >= split.csv 行数 * 0.98
# ============================================================
cd "<PROJECT_ROOT>"
source "<CONDA_ROOT>/etc/profile.d/conda.sh"
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
CH=logs/resume
EPOCHS=40
mkdir -p logs

log() { echo "[$(date)] $*" | tee -a $CH.log; }

# ---- 某实验是否已完成
exp_done() {
    local d=$1
    [ -f "$d/best.pth" ] || return 1
    local n=0
    [ -f "$d/metrics.csv" ] && n=$(( $(wc -l < "$d/metrics.csv") - 1 ))
    [ "$n" -ge "$EPOCHS" ]
}

run_train() {
    local name=$1 exp=$2 cfg=$3 dir=$4 extra=$5
    if exp_done "$dir"; then log "SKIP $exp（已完成 $EPOCHS epoch）"; return 0; fi
    log "RUN  $exp  ($dir)"
    python train/train_final.py --config "$cfg" --exp "$exp" --name "$name" \
        --batch-size 16 --num-workers 16 --proc "$PROC" --raw "$RAW" $extra \
        > "logs/final_$name.log" 2>&1 || { log "[FAIL] $exp"; return 1; }
    tail -4 "logs/final_$name.log" | tee -a $CH.log
}

run_eval() {
    local tag=$1 exp=$2 ckpt=$3 cfg=$4 extra=$5
    [ -f "$ckpt" ] || { log "SKIP eval $tag（无 ckpt）"; return 0; }
    log "EVAL $tag"
    python eval/eval_final.py --ckpt "$ckpt" --config "$cfg" \
        --exp "$exp" --tag "$tag" --proc "$PROC" --raw "$RAW" \
        $extra --batch-size 8 --num-workers 8 \
        > "logs/eval_$tag.log" 2>&1 || log "[FAIL] eval $tag"
    tail -3 "logs/eval_$tag.log" | tee -a $CH.log
}

log "=========================================="
log "RESUME CHAIN START (epochs=$EPOCHS)"
log "=========================================="

# ---------- 0) 先验是否齐全 ----------
NTARGET=$(python - <<'PY'
import csv
print(sum(1 for _ in csv.DictReader(open("datasets/EVD4UAV_processed/split.csv",
                                          encoding="utf-8"))))
PY
)
NHAVE=$(ls $PROC/oof_priors/*.png 2>/dev/null | wc -l)
log "先验: 已有 $NHAVE / 需要 $NTARGET"
if [ "$NHAVE" -lt $((NTARGET * 98 / 100)) ]; then
    log "RUN  infer prior (V3)"
    python scripts/infer_prior.py \
        --ckpt runs/priornet/v3_litehr/best.pth --arch litehr \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$PROC/oof_priors" --tile 1024 --overlap 192 \
        > logs/infer_prior_v3.log 2>&1 || log "[FAIL] infer prior"
    log "先验完成: $(ls $PROC/oof_priors/*.png 2>/dev/null | wc -l)"
else
    log "SKIP infer prior（已齐全）"
fi
df -h <PROJECT_ROOT> | tail -1 | tee -a $CH.log

# ---------- 1) 主实验 ----------
run_train E0_rgb     E0 configs/rgb_baseline.yaml runs/rgb_baseline/E0_rgb ""
run_train E5_bcp     E5 configs/bcp_vis.yaml      runs/bcp_vis/E5_bcp      "--prior-dir $PROC/oof_priors"
run_eval  E0 E0 runs/rgb_baseline/E0_rgb/best.pth configs/rgb_baseline.yaml ""
run_eval  E5 E5 runs/bcp_vis/E5_bcp/best.pth      configs/bcp_vis.yaml      "--prior-dir $PROC/oof_priors"

# ---------- 2) 对照组 ----------
run_train E1_zero   E1 configs/bcp_vis.yaml runs/bcp_vis/E1_zero   ""
run_eval  E1 E1 runs/bcp_vis/E1_zero/best.pth   configs/bcp_vis.yaml ""
run_train E2_random E2 configs/bcp_vis.yaml runs/bcp_vis/E2_random ""
run_eval  E2 E2 runs/bcp_vis/E2_random/best.pth configs/bcp_vis.yaml ""
run_train E7_oracle E7 configs/bcp_vis.yaml runs/bcp_vis/E7_oracle ""
run_eval  E7 E7 runs/bcp_vis/E7_oracle/best.pth configs/bcp_vis.yaml ""

python eval/run_ablation.py --merge-only > logs/ablation.log 2>&1 || log "[FAIL] ablation"
tail -12 logs/ablation.log | tee -a $CH.log

# ---------- 3) OOF（论文级） ----------
if [ -f "$PROC/oof_priors/oof_manifest.csv" ]; then
    log "SKIP OOF（已有 oof_manifest.csv）"
else
    log "RUN  OOF 5-fold prior"
    python scripts/generate_oof_prior.py \
        --proc "$PROC" --raw "$RAW" --folds 5 --seed 42 \
        --base-config configs/prior_v3_litehr.yaml --prior-cfg-name litehr \
        --batch-size 16 --num-workers 16 \
        > logs/oof_prior.log 2>&1 || log "[FAIL] oof prior"
    log "OOF 先验: $(ls $PROC/oof_priors/*.png 2>/dev/null | wc -l)"
fi
run_train E5_oof E5 configs/bcp_vis.yaml runs/bcp_vis/E5_oof "--prior-dir $PROC/oof_priors"
run_eval  E5_OOF E5 runs/bcp_vis/E5_oof/best.pth configs/bcp_vis.yaml "--prior-dir $PROC/oof_priors"

# ---------- 4) V4 Altitude-Aware + E6 ----------
if exp_done runs/priornet/v4_altitude; then
    log "SKIP V4 PriorNet（已完成）"
else
    log "RUN  V4 Altitude-Aware PriorNet"
    python train/train_prior.py --config configs/prior_v4_altitude.yaml \
        --name v4_altitude --batch-size 16 --num-workers 16 \
        --proc "$PROC" --raw "$RAW" \
        > logs/prior_v4.log 2>&1 || log "[FAIL] V4"
    tail -4 logs/prior_v4.log | tee -a $CH.log
fi

V4DIR="$PROC/priors_v4"
mkdir -p "$V4DIR"
NV4=$(ls $V4DIR/*.png 2>/dev/null | wc -l)
if [ "$NV4" -lt $((NTARGET * 98 / 100)) ]; then
    log "RUN  infer prior (V4)  现有 $NV4"
    python scripts/infer_prior.py \
        --ckpt runs/priornet/v4_altitude/best.pth --arch altitude \
        --proc "$PROC" --raw "$RAW" \
        --out-dir "$V4DIR" --tile 1024 --overlap 192 \
        > logs/infer_prior_v4.log 2>&1 || log "[FAIL] infer v4"
else
    log "SKIP infer V4（已齐全 $NV4）"
fi
run_train E6_altitude E6 configs/bcp_vis.yaml runs/bcp_vis/E6_altitude "--prior-dir $V4DIR"
run_eval  E6 E6 runs/bcp_vis/E6_altitude/best.pth configs/bcp_vis.yaml "--prior-dir $V4DIR"

# ---------- 5) 最终汇总 + 可视化 ----------
python eval/run_ablation.py --merge-only > logs/ablation_final.log 2>&1 || log "[FAIL] ablation final"
tail -20 logs/ablation_final.log | tee -a $CH.log

python tools/compare_exps.py --baseline E0 --others E1,E2,E5,E5_OOF,E6,E7 \
    > logs/compare_exps.log 2>&1 || log "[FAIL] compare"
tail -40 logs/compare_exps.log | tee -a $CH.log

if [ -f runs/bcp_vis/E5_bcp/best.pth ]; then
    python eval/visualize_samples.py \
        --ckpt runs/bcp_vis/E5_bcp/best.pth --config configs/bcp_vis.yaml \
        --exp E5 --n 12 --proc "$PROC" --raw "$RAW" \
        --prior-dir "$PROC/oof_priors" \
        > logs/visualize_E5.log 2>&1 || log "[FAIL] visualize E5"
fi

log "=========================================="
log "RESUME CHAIN END"
log "=========================================="
