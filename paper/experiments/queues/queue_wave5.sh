#!/bin/bash
# Wave 5: harder tests of the networks trained on four cameras (no retraining).
#   T1  microlens focal lengths outside the training range (0.70, 0.75, 1.15, 1.20 mm)
#   T2  camera parameters that never varied in training (main-lens focal length, object distance), with the full
#       calibration in the lookup, and with the views rescaled to the training magnification
#   T3  a second set of sealed fields at the training camera, generated after all architectures were fixed
cd "$(dirname "$0")/.."
PY=python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=7
E="$PY evaluate.py"
i=0
for f in 0.70 0.75 1.15 1.20; do
  tag=$(printf "%03d" $(echo "$f*100/1" | bc))
  $PY gen.py static --out sealed_cam$tag --n 40 --seed0 $((98300000 + i * 100000)) --f $f --workers 20 | tail -1
  i=$((i + 1))
done
# name F L f   f_eq(=f/M^2, for the lifting baseline without rescaling)   f_rect(=f/M, after rescaling)
T2="sealed_F095:95:200:0.8:0.97729:0.88421 sealed_F105:105:200:0.8:0.65488:0.72381 sealed_L190:100:190:0.8:0.648:0.72 sealed_L210:100:210:0.8:0.968:0.88 sealed_F095_f095:95:200:0.95:1.16053:1.05 sealed_F105_f095:105:200:0.95:0.77767:0.859524"
i=0
for t in $T2; do
  IFS=: read name F L f feq frect <<< "$t"
  $PY gen.py static --out $name --n 40 --seed0 $((98700000 + i * 100000)) --f $f --F $F --L $L --workers 20 | tail -1
  i=$((i + 1))
done
$PY gen.py static --out sealed2_std --n 120 --seed0 99500000 --workers 30 | tail -1
echo DATA_DONE $(date +%H:%M:%S)
(
  for run in mc_viewlift mc_liftcal mc_lift mc_v3 mc_projlift; do
    for c in "sealed_cam070 0.7" "sealed_cam075 0.75" "sealed_cam115 1.15" "sealed_cam120 1.2"; do set -- $c; $E --run runs/$run --data $1 --f-lookup $2; done
  done 2>&1 | grep -v -i warn; echo T1_DONE
) > ../logs/wave5_T1.log 2>&1 &
(
  for run in mc_viewlift mc_lift mc_v3 mc_projlift; do
    for t in $T2; do IFS=: read name F L f feq frect <<< "$t"; $E --run runs/$run --data $name --f-lookup $f --F-lookup $F --L-lookup $L; done
  done 2>&1 | grep -v -i warn
  for t in $T2; do IFS=: read name F L f feq frect <<< "$t"
    $E --run runs/mc_liftcal --data $name --f-lookup $f --F-lookup $F --L-lookup $L
    $E --run runs/mc_liftcal --data $name --f-lookup $feq --F-lookup $F --L-lookup $L --tag-suffix feq
  done 2>&1 | grep -v -i warn; echo T2RAW_DONE
) > ../logs/wave5_T2raw.log 2>&1 &
(
  for run in mc_viewlift mc_lift mc_v3 mc_projlift; do
    for t in $T2; do IFS=: read name F L f feq frect <<< "$t"; $E --run runs/$run --data $name --f-lookup $f --F-lookup $F --L-lookup $L --rectify; done
  done 2>&1 | grep -v -i warn
  for t in $T2; do IFS=: read name F L f feq frect <<< "$t"
    $E --run runs/mc_liftcal --data $name --f-lookup $frect --F-lookup $F --L-lookup $L --rectify --tag-suffix frect
  done 2>&1 | grep -v -i warn; echo T2RECT_DONE
) > ../logs/wave5_T2rect.log 2>&1 &
(
  for run in p7 scratch_s0 runs/seed1 runs/seed2 runs/projlift_s0 runs/projlift_s1 runs/lift_s0 runs/lift_s1 runs/viewlift_s0 runs/viewlift_w_s0 runs/mc_viewlift runs/mc_liftcal; do
    $E --run $run --data sealed2_std
  done 2>&1 | grep -v -i warn
  $E --run p7 --data sealed2_std --decode heatmap 2>&1 | grep -v -i warn
  $E --run runs/viewlift_s0 --data sealed2_std --decode heatmap 2>&1 | grep -v -i warn; echo T3_DONE
) > ../logs/wave5_T3.log 2>&1 &
wait
echo WAVE5_DONE $(date +%H:%M:%S)
