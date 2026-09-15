#!/usr/bin/env bash
# ============================================================
# 自动分析器：等某个实验训练完（metrics 达到目标轮数）并被评估完，
# 然后自动生成分组对比表，写入 logs/auto_analysis_<tag>.log。
#
# 用法： bash scripts/auto_analyze.sh E5 runs/bcp_vis/E5_bcp 40
# ============================================================
cd /root/autodl-tmp/BCP-Vis
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

TAG=$1
DIR=$2
EPOCHS=${3:-40}
OUT=logs/auto_analysis_$TAG.log

echo "[$(date)] auto_analyze 启动：等待 $TAG ($DIR) 完成 $EPOCHS 轮..." > $OUT

# ---- 1) 等训练完成
while true; do
    if [ -f "$DIR/metrics.csv" ]; then
        N=$(( $(wc -l < "$DIR/metrics.csv") - 1 ))
        if [ "$N" -ge "$EPOCHS" ]; then break; fi
    fi
    # 训练进程没了但没跑满 -> 视为失败/中断
    if ! pgrep -f "train_[f]inal.py" > /dev/null; then
        if [ ! -f "$DIR/metrics.csv" ] || [ "$N" -lt "$EPOCHS" ]; then
            echo "[$(date)] 训练进程已退出且未跑满 $EPOCHS 轮（当前 $N），放弃等待" >> $OUT
            exit 1
        fi
    fi
    sleep 60
done
echo "[$(date)] $TAG 训练完成" >> $OUT

# ---- 2) 等评估结果出现
for i in $(seq 1 120); do
    if grep -q "^$TAG," reports/final_metrics.csv 2>/dev/null; then
        echo "[$(date)] $TAG 评估已完成" >> $OUT
        break
    fi
    sleep 30
done

# ---- 3) 生成对比表
echo "" >> $OUT
echo "############ 分组对比（$TAG vs E0） ############" >> $OUT
python tools/compare_exps.py --baseline E0 --others E1,E2,E5,E5_OOF,E6,E7 >> $OUT 2>&1

echo "" >> $OUT
echo "############ by_size 明细 ############" >> $OUT
cat reports/by_size.csv >> $OUT 2>/dev/null
echo "" >> $OUT
echo "############ by_snow 明细 ############" >> $OUT
cat reports/by_snow.csv >> $OUT 2>/dev/null

echo "[$(date)] 分析完成" >> $OUT
