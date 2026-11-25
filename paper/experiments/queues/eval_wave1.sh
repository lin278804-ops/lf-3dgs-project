#!/bin/bash
# Evaluate each training run as soon as it finishes. Usage: eval_wave1.sh <gpu>
cd "$(dirname "$0")"
PY=python
export CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
STRATA="sealed_conc050 sealed_conc075 sealed_diam30 sealed_diam70"
NOISE="--snr 50 40 35 30 25 20"
E="$PY evaluate.py"

run_evals() {
  case $1 in
    seed1|seed2)      $E --run runs/$1 --data sealed_std ds9test; $E --run runs/$1 --data sealed_std --decode heatmap ;;
    v3_n120)          $E --run runs/$1 --data sealed_std; $E --run runs/$1 --data sealed_camB --f-lookup 1.0
                      $E --run runs/$1 --data sealed_camB --convert-to 1.0 ;;
    lift_s0|lift_compact|lift_n120)
                      $E --run runs/$1 --data sealed_std; $E --run runs/$1 --data sealed_std --snr 40 30
                      $E --run runs/$1 --data sealed_camB; $E --run runs/$1 --data sealed_camB --convert-to 1.0
                      $E --run runs/$1 --data $STRATA ;;
    p7_ctrl)          $E --run runs/$1 --data sealed_std; $E --run runs/$1 --data sealed_std --decode heatmap
                      $PY motion.py --run runs/$1 ;;
    p7_node)          $E --run runs/$1 --data sealed_std $STRATA; $E --run runs/$1 --data sealed_std $NOISE
                      $E --run runs/$1 --data sealed_std --decode heatmap
                      $E --run runs/$1 --data sealed_camB --f-lookup 1.0; $E --run runs/$1 --data sealed_camB --convert-to 1.0
                      $PY motion.py --run runs/$1 ;;
    p7_node_d128)     $E --run runs/$1 --data sealed_std; $PY motion.py --run runs/$1 ;;
    p7_aug)           $E --run runs/$1 --data sealed_std $NOISE; $E --run runs/$1 --data $STRATA
                      $E --run runs/$1 --data sealed_std --gain 0.7 --tag-suffix gain0.7
                      $E --run runs/$1 --data sealed_std --gain 1.3 --tag-suffix gain1.3
                      $E --run runs/$1 --data sealed_std --decode heatmap; $PY motion.py --run runs/$1 ;;
    p7_mixconc)       $E --run runs/$1 --data sealed_std $STRATA; $E --run runs/$1 --data sealed_std --snr 40 30 ;;
    final)            $E --run runs/$1 --data sealed_std $STRATA; $E --run runs/$1 --data sealed_std $NOISE
                      $E --run runs/$1 --data sealed_camB --f-lookup 1.0; $E --run runs/$1 --data sealed_camB --convert-to 1.0
                      $E --run runs/$1 --data sealed_std --gain 0.7 --tag-suffix gain0.7
                      $E --run runs/$1 --data sealed_std --gain 1.3 --tag-suffix gain1.3
                      $PY motion.py --run runs/$1 ;;
  esac
}

PENDING="$2"
[ -z "$PENDING" ] && PENDING="p7_node_d128 p7_node p7_ctrl p7_aug seed1 seed2 v3_n120 lift_n120 p7_mixconc lift_compact lift_s0"
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
echo WAVE1_DONE
