#!/bin/bash
cd "$(dirname "$0")/.."
PY=python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=7
E="$PY evaluate.py"
cams="sealed_cam085:0.85 sealed_cam090:0.9 sealed_cam095:0.95 sealed_camB:1.0 sealed_cam105:1.05 sealed_cam110:1.1"
PENDING="viewlift_n120 viewlift_s1 mc_viewlift_s1 mc_liftcal_s1"
while [ -n "$PENDING" ]; do
  NEXT=""
  for n in $PENDING; do
    if [ -f ../runs/$n/trained.json ]; then
      echo "=== EVAL $n $(date +%H:%M:%S)"
      case $n in
        mc_*) $E --run runs/$n --data sealed_std; for c in $cams; do $E --run runs/$n --data ${c%%:*} --f-lookup ${c##*:}; done
              $E --run runs/$n --data sealed_std --snr 30 --condition ;;
        viewlift_n120) $E --run runs/$n --data sealed_std; $E --run runs/$n --data sealed_camB --convert-to 1.0; $E --run runs/$n --data sealed_camB --f-lookup 1.0 ;;
        viewlift_s1) $E --run runs/$n --data sealed_std sealed_conc050 sealed_conc075 sealed_diam30 sealed_diam70; $E --run runs/$n --data sealed_std --decode heatmap
              $E --run runs/$n --data sealed_std --snr 40 30 20 --condition
              for c in "sealed_camB 1.0" "sealed_cam090 0.9" "sealed_cam110 1.1"; do set -- $c; $E --run runs/$n --data $1 --convert-to $2; $E --run runs/$n --data $1 --f-lookup $2; done
              $PY motion.py --run runs/$n --decode heatmap; $PY real_pairs.py --run runs/$n --decode heatmap --data realpairs_all | grep SUMMARY ;;
      esac 2>&1 | grep -v -i warn
      echo "=== EVALDONE $n $(date +%H:%M:%S)"
    else NEXT="$NEXT $n"; fi
  done
  PENDING="$NEXT"; [ -n "$PENDING" ] && sleep 60
done
echo WAVE4_DONE
