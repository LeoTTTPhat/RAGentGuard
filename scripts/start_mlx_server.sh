#!/bin/zsh
# Start an mlx_lm.server for AgentDojo and register its PID with memguard.
# usage: start_mlx_server.sh <model_id> <port> [decode_concurrency]
# Uses $PYTHON (default python3 of the active environment). The model must already be in the
# Hugging Face cache unless HF_HUB_OFFLINE=0. Logs and PIDs go to run/.
CODE=${0:A:h:h}
PY=${PYTHON:-python3}
MODEL=$1; PORT=$2; DC=${3:-16}
cd $CODE
mkdir -p $CODE/run/pids
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MLX_CACHE_GB=${MLX_CACHE_GB:-4} nohup $PY $CODE/scripts/mlx_server_capped.py --model "$MODEL" --port $PORT --decode-concurrency $DC \
  --prompt-concurrency 4 --prompt-cache-size 48 --max-tokens 1024 --log-level WARNING \
  > $CODE/run/mlx_server_$PORT.log 2>&1 &
echo $! > $CODE/run/pids/mlx_server_$PORT.pid
echo "started pid $(cat $CODE/run/pids/mlx_server_$PORT.pid) on port $PORT"
