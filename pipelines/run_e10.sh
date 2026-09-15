#!/bin/bash
# E10：大容量对照（骨干翻倍、无专家结构）—— 论文容量排除项
cd /root/autodl-tmp/BCP-Vis
exec 200>/tmp/bcp_e10.lock
flock -n 200 || { echo 'e10 already running'; exit 0; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
PROC=datasets/EVD4UAV_processed
RAW=datasets/EVD4UAV/raw/EVD4UAV
LOG=logs/e10.log
LP=runs/priornet/prior_large_plain/best.pth
LPDIR=$PROC/priors_large
mkdir -p $LPDIR
echo "[$(date)] STEP infer large prior" >> $LOG
python scripts/infer_prior.py --ckpt $LP --arch litehr     --model-config configs/prior_large_plain.yaml     --proc $PROC --raw $RAW --out-dir $LPDIR --tile 1024 --overlap 192     >> $LOG 2>&1 || echo '[FAIL] infer large' >> $LOG
N=$(ls $LPDIR/*.png 2>/dev/null | wc -l)
echo "large prior files: $N / 9371" >> $LOG
if [ "$N" -lt 9371 ]; then echo '[ABORT] large prior' >> $LOG; exit 1; fi
echo "[$(date)] STEP train E10" >> $LOG
python train/train_final.py --config configs/bcp_vis.yaml     --exp E10 --name E10_large --batch-size 16 --num-workers 8     --proc $PROC --raw $RAW --prior-dir $LPDIR     >> $LOG 2>&1 || echo '[FAIL] E10 train' >> $LOG
if [ ! -f runs/bcp_vis/E10_large/best.pth ]; then echo '[ABORT] E10 ckpt' >> $LOG; exit 1; fi
echo "[$(date)] STEP eval E10" >> $LOG
python eval/eval_final.py --ckpt runs/bcp_vis/E10_large/best.pth     --config configs/bcp_vis.yaml --exp E10 --tag E10     --proc $PROC --raw $RAW --prior-dir $LPDIR --batch-size 8 --num-workers 8     >> $LOG 2>&1 || echo '[FAIL] E10 eval' >> $LOG
python eval/make_report.py >> $LOG 2>&1
echo DONE > logs/e10.done
echo "[$(date)] E10 DONE" >> $LOG
