#!/bin/bash
cd "$(dirname "$0")/.."
PY=python
while [ $(ls ../fits/ds9train 2>/dev/null | grep -c "\.npz$") -lt 960 ] || pgrep -u $(id -u) -f "fit.py --data ds9train" > /dev/null; do sleep 60; done
$PY make_pseudo.py
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=${LFPIV_WORK}/data/ds9_pseudo
CUDA_VISIBLE_DEVICES=6 $PY train.py --name student_v3 --seed 0 --model v3 --data $D > ../logs/train_student_v3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY train.py --name student_projlift --seed 0 --model projlift --data $D > ../logs/train_student_projlift.log 2>&1 &
wait
export CUDA_VISIBLE_DEVICES=6
for r in student_v3 student_projlift; do
  $PY evaluate.py --run runs/$r --which last.pt --data sealed_std 2>&1 | grep -v -i warn
  $PY evaluate.py --run runs/$r --which last.pt --data sealed_std --decode heatmap 2>&1 | grep -v -i warn
  $PY evaluate.py --run runs/$r --which last.pt --data sealed_std --snr 30 --condition 2>&1 | grep -v -i warn
  $PY evaluate.py --run runs/$r --which last.pt --data sealed_camB --convert-to 1.0 2>&1 | grep -v -i warn
done
for r in v3_long projlift_long lift_long; do
  while [ ! -f ../runs/$r/trained.json ]; do sleep 60; done
  for a in ${r}_a $r; do
    $PY evaluate.py --run runs/$a --data sealed_std 2>&1 | grep -v -i warn
    for c in "sealed_camB 1.0" "sealed_cam090 0.9" "sealed_cam110 1.1"; do set -- $c; $PY evaluate.py --run runs/$a --data $1 --convert-to $2 2>&1 | grep -v -i warn; done
    $PY evaluate.py --run runs/$a --data sealed_std --snr 30 --condition 2>&1 | grep -v -i warn
  done
done
echo STUDENT_AND_LONG_DONE
