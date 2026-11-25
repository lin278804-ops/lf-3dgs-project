# Experiments of the manuscript

This directory holds the code behind *An explicit optical projection makes learned light-field particle
reconstruction compact, accurate and testable*: the network architectures that are compared, their training and
evaluation, the classical baselines, per-exposure fitting, the two-exposure experiments, and the scripts that turn
the result records into the figures, tables and quoted numbers.

It builds on the package `lf_3dgs` at the top of this repository (camera model, renderer, pooled network, losses) and
on the ray tracer in [lf-3dgs-datagen](https://github.com/lin278804-ops/lf-3dgs-datagen).

## Layout

| Path | Content |
|---|---|
| `experiments/gapcommon.py` | The architectures (`LiftNet`, `ProjLiftNet`, `ViewLiftNet`, `LiftCalNet`, `NodeOffsetNet`), input conditioning, detection and scoring |
| `experiments/train.py` | Training on one camera or on several cameras (`--multi`) |
| `experiments/evaluate.py` | Scoring on sealed fields: noise, conditioning, occupancy-peak decoder, calibration in the lookup, conversion of output coordinates, lookup intervention |
| `experiments/classical.py` | Backprojection, calibrated plane sweep and multiplicative algebraic reconstruction |
| `experiments/fit.py` | Per-exposure Gaussian fitting (uses `experiments/fitcore/`, see below) |
| `experiments/motion.py` | Two-exposure displacement on sealed fields |
| `experiments/gen.py` | Generation of every data set from its seeds |
| `experiments/real_pairs.py`, `real_plate.py`, `recorded/` | Analyses of the recorded images (the images are not distributed) |
| `experiments/queues/` | The batch scripts as they were run; they list the exact command of every evaluation |
| `experiments/manifests/` | Seeds and SHA-256 digests of every generated file |
| `analysis/` | Figures, tables and numbers from the result records |
| `CHECKPOINTS.md`, `checkpoints.json` | Trained networks, how each was trained and where it is used |

## Environment

The experiments were run with Python 3.10, PyTorch 2.4.1 (CUDA 11.8), NumPy 2.2, SciPy 1.15 and PyYAML 6.0 on NVIDIA
A800 GPUs. `pip install -r ../requirements.txt` in the repository root installs the dependencies of the package; the
analysis scripts need `analysis/requirements.txt`.

Three environment variables locate everything else (defaults in parentheses):

| Variable | Meaning |
|---|---|
| `LFPIV_PROJECT` | checkout of this repository (two levels above `experiments/`) |
| `LFPIV_DATAGEN` | checkout of lf-3dgs-datagen (`../lf-3dgs-datagen` next to this repository) |
| `LFPIV_WORK` | working directory with `runs/`, `data/`, `results/`, `fits/` (`<repository>/work`) |
| `LFPIV_TRAIN_DATA` | training archive with `train/`, `val/`, `test/` (`$LFPIV_WORK/data/lightfield_dataset9`) |

## Data

All simulated data are regenerated from seeds; nothing needs to be downloaded. Run from `experiments/`.

The training archive (960 / 120 / 120 fields, seeds 30000000-30001199) is rebuilt with

```bash
python gen.py archive --manifest manifests/lightfield_dataset9_seeds.json --out lightfield_dataset9
python gen.py verify --out lightfield_dataset9      # images and coordinates against the digests of the files we used
```

`gen.py verify --out <set>` works for every set below; the digests are in `manifests/content/` and cover the images
and the reference coordinates, so they do not depend on the compression of the files.

The other sets:

| Set | Files | Command |
|---|---|---|
| `mc_f080/train` | 240 | `python gen.py static --out mc_f080/train --n 240 --seed0 97000000` |
| `mc_f080/val` | 15 | `python gen.py static --out mc_f080/val --n 15 --seed0 97050000` |
| `mc_f090/train` | 240 | `python gen.py static --out mc_f090/train --n 240 --seed0 97100000 --f 0.9` |
| `mc_f090/val` | 15 | `python gen.py static --out mc_f090/val --n 15 --seed0 97150000 --f 0.9` |
| `mc_f100/train` | 240 | `python gen.py static --out mc_f100/train --n 240 --seed0 97200000 --f 1` |
| `mc_f100/val` | 15 | `python gen.py static --out mc_f100/val --n 15 --seed0 97250000 --f 1` |
| `mc_f110/train` | 240 | `python gen.py static --out mc_f110/train --n 240 --seed0 97300000 --f 1.1` |
| `mc_f110/val` | 15 | `python gen.py static --out mc_f110/val --n 15 --seed0 97350000 --f 1.1` |
| `mc_f120/train` | 960 | `python gen.py static --out mc_f120/train --n 960 --seed0 97600000 --f 1.2` |
| `mc_f120/val` | 30 | `python gen.py static --out mc_f120/val --n 30 --seed0 97680000 --f 1.2` |
| `sealed2_std` | 120 | `python gen.py static --out sealed2_std --n 120 --seed0 99500000` |
| `sealed_F095` | 40 | `python gen.py static --out sealed_F095 --n 40 --seed0 98700000 --F 95` |
| `sealed_F095_f095` | 40 | `python gen.py static --out sealed_F095_f095 --n 40 --seed0 99100000 --f 0.95 --F 95` |
| `sealed_F105` | 40 | `python gen.py static --out sealed_F105 --n 40 --seed0 98800000 --F 105` |
| `sealed_F105_f095` | 40 | `python gen.py static --out sealed_F105_f095 --n 40 --seed0 99200000 --f 0.95 --F 105` |
| `sealed_L190` | 40 | `python gen.py static --out sealed_L190 --n 40 --seed0 98900000 --L 190` |
| `sealed_L210` | 40 | `python gen.py static --out sealed_L210 --n 40 --seed0 99000000 --L 210` |
| `sealed_cam070` | 40 | `python gen.py static --out sealed_cam070 --n 40 --seed0 98300000 --f 0.7` |
| `sealed_cam075` | 40 | `python gen.py static --out sealed_cam075 --n 40 --seed0 98400000 --f 0.75` |
| `sealed_cam085` | 40 | `python gen.py static --out sealed_cam085 --n 40 --seed0 98000000 --f 0.85` |
| `sealed_cam090` | 40 | `python gen.py static --out sealed_cam090 --n 40 --seed0 94000000 --f 0.9` |
| `sealed_cam095` | 40 | `python gen.py static --out sealed_cam095 --n 40 --seed0 98100000 --f 0.95` |
| `sealed_cam105` | 40 | `python gen.py static --out sealed_cam105 --n 40 --seed0 98200000 --f 1.05` |
| `sealed_cam110` | 40 | `python gen.py static --out sealed_cam110 --n 40 --seed0 95000000 --f 1.1` |
| `sealed_cam115` | 40 | `python gen.py static --out sealed_cam115 --n 40 --seed0 98500000 --f 1.15` |
| `sealed_cam120` | 40 | `python gen.py static --out sealed_cam120 --n 40 --seed0 98600000 --f 1.2` |
| `sealed_camB` | 40 | `python gen.py static --out sealed_camB --n 40 --seed0 93000000 --f 1` |
| `sealed_conc050` | 40 | `python gen.py static --out sealed_conc050 --n 40 --seed0 91100000 --conc 0.05` |
| `sealed_conc075` | 40 | `python gen.py static --out sealed_conc075 --n 40 --seed0 91200000 --conc 0.075` |
| `sealed_diam30` | 40 | `python gen.py static --out sealed_diam30 --n 40 --seed0 91300000 --diam 30` |
| `sealed_diam70` | 40 | `python gen.py static --out sealed_diam70 --n 40 --seed0 91400000 --diam 70` |
| `sealed_motion` | 24 | `python gen.py motion --out sealed_motion --n 24 --seed0 92000000` |
| `sealed_std` | 120 | `python gen.py static --out sealed_std --n 120 --seed0 91000000` |
| `simplate_d050` | 19 | `python gen.py plate --out simplate_d050 --seed0 96100000 --diam-mm 0.05 --pitch-mm 1.224` |
| `simplate_d245` | 19 | `python gen.py plate --out simplate_d245 --seed0 96000000 --diam-mm 0.245 --pitch-mm 1.224` |

`sealed_std` and `sealed2_std` are the two sets of sealed fields at the training camera, `sealed_conc*` and
`sealed_diam*` the density and tracer-size strata, `sealed_cam*`, `sealed_F*` and `sealed_L*` the cameras absent from
training, `mc_f*` the fields for training on several cameras and `sealed_motion` the two-exposure fields.
`ds9_pseudo` (training on fitted particles) is produced by `fit.py --data ds9train` followed by `make_pseudo.py`.

## Trained networks

Unpack `checkpoints_v1.tar` (see `CHECKPOINTS.md`) into `$LFPIV_WORK`, which gives `$LFPIV_WORK/runs/<name>/`.

## Reproducing the evaluations

```bash
cd experiments
python evaluate.py --run runs/viewlift_s0 --data sealed_std                       # view-preserving network, F1 0.919
python evaluate.py --run p7 --data sealed_std --decode heatmap                    # pooled network, occupancy-peak decoder
python evaluate.py --run runs/viewlift_s0 --data sealed_std --intervene reverse --limit 40   # lookup intervention
python evaluate.py --run runs/lift_s0 --data sealed_camB --convert-to 1.0         # another camera, output converted
python evaluate.py --run runs/mc_viewlift --data sealed_cam070 --f-lookup 0.7     # camera outside the training range
python evaluate.py --run runs/mc_viewlift --data sealed_F105 --F-lookup 105       # another main lens
python evaluate.py --run runs/viewlift_s0 --data sealed_std --snr 30 --condition  # sensor noise, conditioned input
python classical.py run --data sealed_std                                         # classical reconstructions
python fit.py --data sealed_std && python score_preds.py                          # per-exposure fitting
python motion.py --run p7 --decode heatmap                                        # two-exposure displacement
```

Each call writes `results/<run>__<data>[__options].json` (summary and per-field scores) and an `.npz` with the
detections. `queues/` contains every call that produced a result used in the manuscript, and `CHECKPOINTS.md` the
training command of every network.

## Figures and numbers

`analysis/make_figures.sh` regenerates every figure, table and quoted number from the result records. Unpack the
Source Data archive of the manuscript to `analysis/source_data/` first; `source_data/gap/results/` has the same
content as `$LFPIV_WORK/results/`.

## Per-exposure fitting

`fit.py` runs the optimizer from `experiments/fitcore/`, a later revision of `lf_3dgs` in which the optimizer cannot
read reference labels: labels are not loaded, the number of initial candidates is fixed, and the pruning floor does not
depend on the number of reference particles. `lf_3dgs/optimizer.py` at the top of the repository is the earlier
revision. The settings are in `experiments/configs/e7_star_physics_tuned.yaml`.

## Licence

Apache-2.0; see `LICENSE` at the top of the repository.

## Not included

- The recorded plenoptic images (target, flow pairs, tank). They were acquired by collaborators; see the Data
  availability statement of the manuscript.
- The archive of mixed seeding density used to tune `p7_mixconc`, and the networks of the learned-lookup and
  flow-family studies (Extended Data Figs 1 and 4), which predate this code; they are available from the authors.
