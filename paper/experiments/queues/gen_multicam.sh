#!/bin/bash
cd "$(dirname "$0")/.."
PY=python
i=0
for f in 0.8 0.9 1.0 1.1; do
  tag=$(printf "%03d" $(echo "$f*100/1" | bc))
  $PY gen.py static --out mc_f$tag/train --n 240 --seed0 $((97000000 + i * 100000)) --f $f --workers 24
  $PY gen.py static --out mc_f$tag/val --n 15 --seed0 $((97050000 + i * 100000)) --f $f --workers 15
  i=$((i + 1))
done
i=0
for f in 0.85 0.95 1.05; do
  tag=$(printf "%03d" $(echo "$f*100/1" | bc))
  $PY gen.py static --out sealed_cam$tag --n 40 --seed0 $((98000000 + i * 100000)) --f $f --workers 20
  i=$((i + 1))
done
echo MULTICAM_DATA_DONE
