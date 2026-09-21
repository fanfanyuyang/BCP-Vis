#!/bin/bash
# supervise_equeue.sh —— e11~e14 队列守护自愈
#
# 每 120s 巡检一次：若某实验尚未完成(无 logs/eN_queue.done)且其对应的
# 队列守护进程(bash run_eN_queue.sh)缺失，则自动重拉。
#   - 已完成(.done 存在)的实验不重拉；
#   - 正在睡眠等待的进程(活着)不重拉；
#   - 同一实验 10 分钟内最多重拉一次，避免崩溃循环刷屏。
#
# 用法： setsid nohup bash supervise_equeue.sh >/dev/null 2>&1 </dev/null &
cd "<PROJECT_ROOT>"
LOG=logs/supervise.log
echo "[$(date)] supervisor 启动 (pid $$)" >> "$LOG"

declare -A last
while true; do
  now=$(date +%s)
  for e in 11 12 13 14; do
    donef="logs/e${e}_queue.done"
    [ -f "$donef" ] && continue
    if ! pgrep -f "bash run_e${e}_queue.sh" >/dev/null 2>&1; then
      prev=${last[$e]:-0}
      if [ $((now - prev)) -lt 600 ]; then
        continue   # 10 分钟内已重拉过，避免崩溃循环
      fi
      last[$e]=$now
      echo "[$(date)] 检测到 e$e 队列进程缺失，重拉" >> "$LOG"
      setsid nohup bash "run_e${e}_queue.sh" >> "$LOG" 2>&1 </dev/null &
    fi
  done
  sleep 120
done
