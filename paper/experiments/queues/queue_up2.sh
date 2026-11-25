#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -u $(id -u) -f "classical.py (tune|run)$|classical.py run --data sealed_std ds9test|classical.py tune$" > /dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python classical.py tune --up 2 && python classical.py run --up 2 --data sealed_std
echo UP2_DONE
