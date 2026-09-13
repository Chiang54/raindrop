#!/bin/bash
# 用法: bash run_training_multi.sh <day|night|both>
# day 已經有完整的 150-epoch 結果在 ./checkpoints，不需要重跑；
# 這支腳本主要是用來跑 night 跟 both 這兩個新的消融實驗分支。
set -e
MODE="$1"
if [[ "$MODE" != "day" && "$MODE" != "night" && "$MODE" != "both" ]]; then
  echo "用法: bash run_training_multi.sh <day|night|both>"
  exit 1
fi

export DATASET_MODE="$MODE"
LOGFILE="/workspace/train_loop_${MODE}.log"

cd /workspace/raindrop
for i in $(seq 1 20); do
  echo "=== [$MODE] Run $i starting at $(date) ===" >> "$LOGFILE"
  python3 train.py >> "$LOGFILE" 2>&1
  rc=$?
  echo "=== [$MODE] Run $i finished at $(date) with exit code $rc ===" >> "$LOGFILE"
  if [ $rc -ne 0 ]; then
    echo "=== [$MODE] Non-zero exit, stopping loop ===" >> "$LOGFILE"
    break
  fi
  # train.py 內部會在 start_epoch > total_target_epochs(150) 時直接空跑結束（exit 0），
  # 所以用「是否已經印出 Epoch [150/150]」判斷有沒有跑完，跑完就提早結束迴圈省時間。
  if grep -q "Epoch \[150/150\]" "$LOGFILE" 2>/dev/null; then
    break
  fi
done
echo "ALL RUNS COMPLETE [$MODE]" >> "$LOGFILE"
