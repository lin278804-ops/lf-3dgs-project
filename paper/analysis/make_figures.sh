#!/bin/bash
# Regenerate every number, table and figure of the manuscript from the result records in source_data/.
# source_data/ is the Source Data archive of the manuscript; unpack it next to this script first.
set -euo pipefail
cd "$(dirname "$0")"
[ -d source_data ] || { echo "unpack the Source Data archive into $(pwd)/source_data first"; exit 1; }
mkdir -p figures manuscript/tables
cd scripts
for s in compute_numbers make_tables fig1_concept fig2_reconstruction fig3_mechanism fig4_camera fig4_architectures si_generated \
         fig5_motion fig6_recordings extended_data gap_figures supp_fig1_depth_sampling compute_gap_numbers; do
  python "$s.py" > /dev/null
  echo "ok  $s"
done
