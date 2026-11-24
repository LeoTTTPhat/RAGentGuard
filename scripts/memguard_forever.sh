#!/bin/zsh
# Keep memguard.py alive (restart after 5 s if it ever exits). The name contains "memguard", so memguard exempts it.
# Uses $PYTHON (default python3 of the active environment). Logs and PIDs go to run/.
CODE=${0:A:h:h}
PY=${PYTHON:-python3}
cd $CODE
mkdir -p $CODE/run/pids
echo $$ > $CODE/run/pids/memguard_forever.pid
while true; do
  $PY $CODE/scripts/memguard.py --interval 5 --server-cap-gb 75 >> $CODE/run/memguard.out 2>&1
  echo "[$(date '+%F %T')] memguard exited ($?); restarting in 5 s" >> $CODE/run/memguard.out
  sleep 5
done
