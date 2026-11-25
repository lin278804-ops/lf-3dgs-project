#!/bin/bash
# Capacity-matched comparison: evaluate each run as soon as it finishes. Usage: eval_wave2.sh <gpu>
cd "$(dirname "$0")"
PY=python
export CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
STRATA="sealed_conc050 sealed_conc075 sealed_diam30 sealed_diam70"
E="$PY evaluate.py"

cameras() {   # run, has_lookup
  for cam in "sealed_camB 1.0" "sealed_cam090 0.9" "sealed_cam110 1.1"; do
    set -- $1 $2 $cam
    $E --run $1 --data $3
    $E --run $1 --data $3 --convert-to $4
    [ "$2" = "1" ] && $E --run $1 --data $3 --f-lookup $4
  done
}

run_evals() {
  case $1 in
    existing)         cameras p7 1; cameras runs/lift_s0 0 ;;
    projlift_s0|projlift_s1)
                      $E --run runs/$1 --data sealed_std $STRATA; $E --run runs/$1 --data sealed_std --decode heatmap
                      $E --run runs/$1 --data sealed_std --snr 50 40 35 30 25 20 --condition
                      cameras runs/$1 1
                      $PY motion.py --run runs/$1 --decode heatmap
                      $PY real_pairs.py --run runs/$1 --decode heatmap --data realpairs_all | grep SUMMARY
                      $PY real_plate.py --run runs/$1 --decode heatmap | grep SUMMARY
                      $PY timing.py runs/$1 ;;
    lift_s1)          $E --run runs/$1 --data sealed_std $STRATA; cameras runs/$1 0
                      $E --run runs/$1 --data sealed_std --snr 30 --condition ;;
    projlift_n120)    $E --run runs/$1 --data sealed_std; cameras runs/$1 1 ;;
  esac
}

while [ ! -f ../data/sealed_cam110/manifest.json ]; do sleep 20; done
echo "=== EVAL existing $(date +%H:%M:%S)"; run_evals existing 2>&1 | grep -v -i warn; echo "=== EVALDONE existing $(date +%H:%M:%S)"
PENDING="projlift_n120 projlift_s0 projlift_s1 lift_s1"
while [ -n "$PENDING" ]; do
  NEXT=""
  for n in $PENDING; do
    if [ -f ../runs/$n/trained.json ]; then
      echo "=== EVAL $n $(date +%H:%M:%S)"
      run_evals $n 2>&1 | grep -v -i warn
      echo "=== EVALDONE $n $(date +%H:%M:%S)"
    else
      NEXT="$NEXT $n"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 60
done
echo WAVE2_DONE
