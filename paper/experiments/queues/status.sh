#!/bin/bash
# One-screen status of the gap experiments.
G=${LFPIV_WORK}
cd $G
date "+%H:%M:%S"
for f in logs/train_*.log; do
  n=$(basename $f .log); n=${n#train_}
  [[ $n == *aborted* ]] && continue
  last=$(grep EPOCH $f | tail -1 | awk '{print $3, $4}')
  f1=$(grep EPOCH $f | tail -1 | sed -E "s/.*'f1': ([0-9.]+).*/\1/")
  best=$(grep EPOCH $f | sed -E "s/.*'f1': ([0-9.]+).*/\1/" | sort -n | tail -1)
  done=$([ -f runs/$n/trained.json ] && echo DONE || (pgrep -u $(id -u) -f "train.py --name $n " > /dev/null && echo run || echo DEAD))
  echo "$n ep/sec: $last val_f1: $f1 best: $best $done"
done
echo "fits std $(ls fits/sealed_std 2>/dev/null | grep -c npz) motion $(ls fits/sealed_motion 2>/dev/null | grep -c npz)"
grep -v Warn logs/classical.log | tail -3 | cut -c1-230
grep -v Warn logs/classical_up2.log 2>/dev/null | tail -2 | cut -c1-230
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'
echo
