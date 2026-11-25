#!/bin/bash
# Evaluations that need no new training: archived principal network (p7) and archived seed-0
# scratch run on the sealed sets. Usage: eval_wave0.sh <gpu>
cd "$(dirname "$0")"
PY=python
export CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
STRATA="sealed_conc050 sealed_conc075 sealed_diam30 sealed_diam70"
$PY evaluate.py --run p7 --data sealed_std --snr 50 40 35 30 25 20
$PY evaluate.py --run p7 --data sealed_std --decode heatmap
$PY evaluate.py --run scratch_s0 --data sealed_std ds9test
$PY evaluate.py --run p7 --data $STRATA
$PY evaluate.py --run p7 --data sealed_conc050 sealed_conc075 --topk 4000
$PY evaluate.py --run p7 --data sealed_std --gain 0.7 --tag-suffix gain0.7
$PY evaluate.py --run p7 --data sealed_std --gain 1.3 --tag-suffix gain1.3
echo WAVE0_DONE
