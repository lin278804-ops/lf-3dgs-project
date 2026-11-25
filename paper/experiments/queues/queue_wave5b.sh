#!/bin/bash
# Wave 5b: (1) a view-preserving network trained on the 1.20-mm camera alone, as the ceiling for that camera;
#          (2) the second initializations on the cameras of wave 5, once they are trained.
cd "$(dirname "$0")/.."
PY=python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=7
E="$PY evaluate.py"
while ! grep -q WAVE5_DONE ../logs/queue_wave5.log; do sleep 30; done
$PY gen.py static --out mc_f120/train --n 960 --seed0 97600000 --f 1.2 --workers 30 | tail -1
$PY gen.py static --out mc_f120/val --n 30 --seed0 97680000 --f 1.2 --workers 30 | tail -1
(
  $PY train.py --name viewlift_f120 --seed 0 --model viewlift --multi 1.2 > ../logs/train_viewlift_f120.log 2>&1
  $E --run runs/viewlift_f120 --data sealed_cam120 --f-lookup 1.2 2>&1 | grep -v -i warn
  echo ORACLE_DONE $(date +%H:%M:%S)
) &
T1="sealed_cam070:0.7 sealed_cam075:0.75 sealed_cam115:1.15 sealed_cam120:1.2"
T2="sealed_F095:95:200:0.8:0.97729 sealed_F105:105:200:0.8:0.65488 sealed_L190:100:190:0.8:0.648 sealed_L210:100:210:0.8:0.968 sealed_F095_f095:95:200:0.95:1.16053 sealed_F105_f095:105:200:0.95:0.77767"
PENDING="viewlift_s1 mc_viewlift_s1 mc_liftcal_s1"
while [ -n "$PENDING" ]; do
  NEXT=""
  for n in $PENDING; do
    if [ -f ../runs/$n/trained.json ]; then
      echo "=== EVAL $n $(date +%H:%M:%S)"
      case $n in
        viewlift_s1) $E --run runs/$n --data sealed2_std; $E --run runs/$n --data sealed_std --condition
                     $E --run runs/$n --data sealed_std --intervene reverse --limit 40
                     for c in sealed_camB sealed_cam090 sealed_cam110; do $E --run runs/$n --data $c; done ;;
        mc_*) $E --run runs/$n --data sealed2_std
              for c in $T1; do $E --run runs/$n --data ${c%%:*} --f-lookup ${c##*:}; done
              for t in $T2; do IFS=: read name F L f feq <<< "$t"
                $E --run runs/$n --data $name --f-lookup $f --F-lookup $F --L-lookup $L
                [ $n = mc_liftcal_s1 ] && $E --run runs/$n --data $name --f-lookup $feq --F-lookup $F --L-lookup $L --tag-suffix feq
              done ;;
      esac 2>&1 | grep -v -i warn
      echo "=== EVALDONE $n $(date +%H:%M:%S)"
    else NEXT="$NEXT $n"; fi
  done
  PENDING="$NEXT"; [ -n "$PENDING" ] && sleep 60
done
wait
echo WAVE5B_DONE $(date +%H:%M:%S)
