#!/bin/bash
# Wave 3: (i) evaluate the view-preserving projection network when trained; (ii) train and evaluate
# five architectures on four cameras at once. Each multi-camera training starts when its GPU is released.
cd "$(dirname "$0")/.."
PY=python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STRATA="sealed_conc050 sealed_conc075 sealed_diam30 sealed_diam70"
E="$PY evaluate.py"

eval_single() {      # network trained on the training camera only
  r=$1
  $E --run runs/$r --data sealed_std $STRATA; $E --run runs/$r --data sealed_std --decode heatmap
  $E --run runs/$r --data sealed_std --snr 40 30 20 --condition
  for c in "sealed_camB 1.0" "sealed_cam090 0.9" "sealed_cam110 1.1"; do
    set -- $c; $E --run runs/$r --data $1; $E --run runs/$r --data $1 --convert-to $2; $E --run runs/$r --data $1 --f-lookup $2
  done
  $PY motion.py --run runs/$r --decode heatmap
  $PY real_pairs.py --run runs/$r --decode heatmap --data realpairs_all | grep SUMMARY
  $PY real_plate.py --run runs/$r --decode heatmap | grep SUMMARY
  $PY timing.py runs/$r
}

eval_multi() {       # network trained on four cameras: each camera read with its own calibration
  r=$1
  $E --run runs/$r --data sealed_std
  for c in "sealed_cam085 0.85" "sealed_cam090 0.9" "sealed_cam095 0.95" "sealed_camB 1.0" "sealed_cam105 1.05" "sealed_cam110 1.1"; do
    set -- $c; $E --run runs/$r --data $1 --f-lookup $2
  done
  $E --run runs/$r --data sealed_std --snr 30 --condition
}

worker() {           # gpu, wait-for run, name, model args...
  gpu=$1; after=$2; name=$3; shift 3
  while [ ! -f ../runs/$after/trained.json ]; do sleep 60; done
  while [ ! -f ../data/sealed_cam105/manifest.json ]; do sleep 60; done
  export CUDA_VISIBLE_DEVICES=$gpu
  if [[ $after == viewlift* ]]; then
    echo "=== EVAL $after $(date +%H:%M:%S)"; eval_single $after 2>&1 | grep -v -i warn; echo "=== EVALDONE $after $(date +%H:%M:%S)"
  fi
  $PY train.py --name $name --seed 0 --multi 0.8,0.9,1.0,1.1 "$@" > ../logs/train_$name.log 2>&1
  echo "=== EVAL $name $(date +%H:%M:%S)"; eval_multi $name 2>&1 | grep -v -i warn; echo "=== EVALDONE $name $(date +%H:%M:%S)"
}

worker 1 v3_long        mc_v3        --model v3        > ../logs/wave3_mc_v3.log 2>&1 &
worker 2 projlift_long  mc_projlift  --model projlift  > ../logs/wave3_mc_projlift.log 2>&1 &
worker 5 lift_long      mc_lift      --model lift      > ../logs/wave3_mc_lift.log 2>&1 &
worker 3 viewlift_s0    mc_viewlift  --model viewlift  > ../logs/wave3_mc_viewlift.log 2>&1 &
worker 4 viewlift_w_s0  mc_liftcal   --model liftcal   > ../logs/wave3_mc_liftcal.log 2>&1 &
wait
echo WAVE3_DONE
