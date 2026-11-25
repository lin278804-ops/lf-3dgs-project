#!/bin/bash
cd "$(dirname "$0")/.."
PY=python
while [ ! -f ../data/realpairs_full.uploaded ]; do sleep 60; done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
g=(1 2 1 2)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=${g[$k]} $PY fit.py --data realpairs_full --shard $k --nshards 4 > ../logs/fit_real_$k.log 2>&1 & done
wait
CUDA_VISIBLE_DEVICES=1 $PY real_pairs.py --fit --data realpairs_full 2>&1 | grep -v -i warn
echo REALFIT_DONE
