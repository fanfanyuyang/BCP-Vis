#!/bin/bash
# ============================================================
# E13：不确定性感知先验 UAP（第 3 个创新点）
#
# 思想：双专家头的【分歧】即先验的不确定性
#     U  = | P_f − (1 − P_b) |      两个专家越不一致，U 越大
#     P* = (1 − U) · P_f           不确定性高的地方自动降低先验权重
#
# 为什么能补上第 2 个创新点的漏洞：
#   V5/E8 用 P_f⊙(1−P_b) 相乘，抑制过强，把小目标也压掉了
#   （tiny IoU 0.709 → 0.687）。而小目标恰恰是"专家最不确定"的区域
#   （U 高）⇒ UAP 会自动在这些地方少信先验、多看原图 ⇒ 召回有望回升。
#
# 成本：不需重训 PriorNet（复用 V8 的 ckpt），只改先验组合方式。
# ============================================================
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e13.lock
flock -n 200 || { echo "e13 already running, exit"; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e13_queue.log
EXPECT=9371
CFG=configs/prior_v8_cascade.yaml
CKPT=runs/priornet/prior_v8_cascade/best.pth
OUTDIR="$PROC/priors_uap"

echo "[$(date)] E13 队列：等待 E12 完成（需要 V8 的 ckpt）..." >> $LOG
while [ ! -f logs/e12_queue.done ]; do sleep 180; done
echo "[$(date)] E13 队列：等待 E14 完成（e14_queue.done，E14 优先）..." >> $LOG
while [ ! -f logs/e14_queue.done ]; do sleep 180; done

if [ ! -f "$CKPT" ]; then
    echo "[ABORT] 缺少 V8 ckpt，无法生成 UAP 先验" >> $LOG
    exit 1
fi

mkdir -p "$OUTDIR"
echo "[$(date)] 生成 UAP 先验（prior-mode=uap）" >> $LOG
python scripts/infer_prior.py --ckpt "$CKPT" --arch cascade \
    --model-config $CFG \
    --prior-mode uap \
    --proc "$PROC" --raw "$RAW" \
    --out-dir "$OUTDIR" --tile 1024 --overlap 192 \
    >> $LOG 2>&1 || echo "[FAIL] infer uap" >> $LOG

N=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
echo "uap prior files: $N / $EXPECT" >> $LOG
if [ "$N" -lt "$EXPECT" ]; then
    echo "[ABORT] uap prior $N < $EXPECT" >> $LOG
    exit 1
fi

echo "[$(date)] 训练 E13" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml \
    --exp E13 --name E13_uap --batch-size 16 --num-workers 8 \
    --proc "$PROC" --raw "$RAW" --prior-dir "$OUTDIR" \
    >> $LOG 2>&1 || echo "[FAIL] E13 train" >> $LOG

if [ -f runs/bcp_vis/E13_uap/best.pth ]; then
    python eval/eval_final.py \
        --ckpt runs/bcp_vis/E13_uap/best.pth --config configs/bcp_vis.yaml \
        --exp E13 --tag E13 --proc "$PROC" --raw "$RAW" \
        --prior-dir "$OUTDIR" --batch-size 8 --num-workers 8 \
        >> $LOG 2>&1 || echo "[FAIL] E13 eval" >> $LOG
    python eval/make_report.py >> $LOG 2>&1
    echo "[$(date)] E13 done" >> $LOG
fi

echo DONE > logs/e13_queue.done
echo "[$(date)] E13 队列 DONE" >> $LOG
