#!/bin/zsh
# Keep an mlx_lm server alive (it may be killed externally); restart after 30 s.
# usage: serve_mlx_forever.sh <model_id> <port> [decode_concurrency]
# Uses $PYTHON (default python3 of the active environment). The model must already be in the
# Hugging Face cache unless HF_HUB_OFFLINE=0. Logs and PIDs go to run/.
CODE=${0:A:h:h}
PY=${PYTHON:-python3}
MODEL=$1; PORT=$2; DC=${3:-16}
cd $CODE
mkdir -p $CODE/run/pids
echo $$ > $CODE/run/pids/serve_forever_$PORT.pid
while true; do
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MLX_CACHE_GB=${MLX_CACHE_GB:-4} $PY $CODE/scripts/mlx_server_capped.py --model "$MODEL" --port $PORT \
    --decode-concurrency $DC --prompt-concurrency 4 --prompt-cache-size 16 --prompt-cache-bytes 8589934592 --max-tokens 1024 --log-level WARNING \
    >> $CODE/run/mlx_server_$PORT.log 2>&1 &
  echo $! > $CODE/run/pids/mlx_server_$PORT.pid
  wait $!
  echo "[$(date '+%F %T')] server on port $PORT exited ($?); restarting in 30 s" >> $CODE/run/mlx_server_$PORT.log
  sleep 30
done
