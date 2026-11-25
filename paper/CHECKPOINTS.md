# Trained networks

`checkpoints_v1.tar` unpacks to `runs/<name>/` (`best.pt`, `config.json`, training history); place `runs/` in `$LFPIV_WORK`.
The last column is the beginning of the SHA-256 of `best.pt`; full digests are in `checkpoints.json`.

| Run | Architecture | Parameters (millions) | Training | Used in | SHA-256 |
|---|---|---|---|---|---|
| `ab_skip_bcast_s0` | Pooled, unwarped stream replicated along depth | 0.25 | last stage of the principal schedule | ED Table 2; Supp. Note 7 | `2b7b4aaea0726710` |
| `lift_long` | Lifting | 9.65 | continued to 100 epochs | ED Fig. 5b | `b6babbfca0d7a84a` |
| `lift_long_a` | Lifting | 9.65 | lift_s0 continued to 80 epochs | ED Fig. 5b | `afa7ca9c3b56fbae` |
| `lift_n120` | Lifting | 9.65 | 120 training fields | Fig. 4e | `fc34038b60cb9ce4` |
| `lift_s0` | Lifting | 9.65 | single stage | Figs 2c, 4, 5e; ED Figs 5, 6d, 7 | `a1de4cf8a966d720` |
| `lift_s1` | Lifting | 9.65 | second initialization | Figs 2c, 4 | `941b6087aa9f0bfd` |
| `mc_lift` | Lifting, no calibration | 9.65 | four cameras | Fig. 4d; Supp. Tables 3, 4 | `47fc4a61d8595769` |
| `mc_liftcal` | Lifting told the focal length | 9.85 | four cameras | Fig. 4d; Supp. Tables 3, 4 | `a4709bb12967f750` |
| `mc_liftcal_s1` | Lifting told the focal length | 9.85 | four cameras, second initialization | Fig. 4d; Supp. Tables 3, 4 | `da3b4c3ac8419cac` |
| `mc_projlift` | Pooled, added capacity | 9.39 | four cameras | Supp. Tables 3, 4 | `c620488a7af3c653` |
| `mc_v3` | Pooled | 0.25 | four cameras | Fig. 4d; Supp. Tables 3, 4 | `8c6ba8d095474f6a` |
| `mc_viewlift` | View-preserving | 0.51 | four cameras | Fig. 4d; Supp. Tables 3, 4 | `37bce07a07904982` |
| `mc_viewlift_s1` | View-preserving | 0.51 | four cameras, second initialization | Fig. 4d; Supp. Tables 3, 4 | `df7a43dcc0841f08` |
| `p7_aug` | Pooled, tuned with noise | 0.25 | principal network tuned for 20 epochs | ED Fig. 6d | `8888ff934d9e711c` |
| `p7_ctrl` | Pooled | 0.25 | control continuation of the principal network, 20 epochs | Supp. Notes 2, 7 | `1713584292151b23` |
| `p7_margin_s0` | Pooled (principal) | 0.25 | five stages, 220 epochs | Figs 2, 3, 5, 6; ED Figs 5c, 6, 7, 8 | `e0c63496ced8ce62` |
| `p7_mixconc` | Pooled, tuned on mixed density | 0.25 | principal network tuned for 8 epochs | ED Fig. 6b | `3134bfe409cc6a28` |
| `p7_node` | Pooled, offsets referenced to nodes | 0.25 | principal network tuned for 20 epochs | ED Fig. 6c; Supp. Note 2 | `d259be807b0798d0` |
| `p7_node_d128` | Pooled, 128 depth planes | 0.26 | principal network tuned for 12 epochs | Fig. 5e | `33ed7ca5336011a8` |
| `projlift_long` | Pooled, added capacity | 9.39 | continued to 100 epochs | ED Fig. 5b | `c614cc76a4d9c9d8` |
| `projlift_long_a` | Pooled, added capacity | 9.39 | projlift_s0 continued to 80 epochs | ED Fig. 5b | `f396270d94e23f8c` |
| `projlift_n120` | Pooled, added capacity | 9.39 | 120 training fields | Fig. 4e | `1904a65e36bc2376` |
| `projlift_s0` | Pooled, added capacity | 9.39 | single stage | Figs 2c, 4 | `517656fef815d3a4` |
| `projlift_s1` | Pooled, added capacity | 9.39 | second initialization | Figs 2c, 4 | `6c70bef42ed1aae7` |
| `scratch_s0` | Pooled | 0.25 | single stage, 40 epochs, first initialization | Figs 2c, 4; ED Figs 2, 6a | `aa022d25286417ed` |
| `scratch_s0_no_render_loss` | Pooled, no rendering loss | 0.25 | as scratch_s0 with the rendered-image loss weighted zero | ED Fig. 2 | `6c6eddfb0be710c4` |
| `seed1` | Pooled | 0.25 | single stage, second initialization | Figs 2c, 4; ED Fig. 6a | `a99c4b53d3387ab6` |
| `seed2` | Pooled | 0.25 | single stage, third initialization | Figs 2c, 4; ED Fig. 6a | `adfbf15f58e0dc08` |
| `student_projlift` | Pooled, added capacity | 9.39 | trained on fitted particles | Supp. Note 7 | `957a32b240954730` |
| `student_v3` | Pooled | 0.25 | trained on fitted particles | Supp. Note 7 | `071c23ffdb92431a` |
| `v3_long` | Pooled | 0.25 | continued to 100 epochs | ED Fig. 5b | `111ae0b853d689da` |
| `v3_long_a` | Pooled | 0.25 | seed1 continued to 80 epochs | ED Fig. 5b | `e8610026bff613e2` |
| `v3_n120` | Pooled | 0.25 | 120 training fields, 160 epochs | Fig. 4e | `b9e5d013134b5611` |
| `viewlift_f120` | View-preserving | 0.51 | 1.20-mm camera alone | Supp. Note 7 | `673a3ae18c9982cc` |
| `viewlift_n120` | View-preserving | 0.51 | 120 training fields | Fig. 4e | `11f3aae04e1b72f2` |
| `viewlift_s0` | View-preserving | 0.51 | single stage | Figs 2c, 4, 5e | `d26b076e678eb4b2` |
| `viewlift_s1` | View-preserving | 0.51 | second initialization | Figs 2c, 4 | `0046fec2bc7f6b31` |
| `viewlift_w_s0` | View-preserving, 256 channels | 1.10 | single stage | Fig. 4b | `836d4c5529dd5034` |

## Training commands

Run from `paper/experiments/`. The single-stage recipe is 40 epochs at a learning rate of 1e-4 (`recipe.json`).
`p7_margin_s0`, `scratch_s0`, `scratch_s0_no_render_loss` and `ab_skip_bcast_s0` were trained earlier with `lf_3dgs/run_s2.py`; their `config.json` records the settings.

| Run | Command |
|---|---|
| `lift_long` | `python train.py --name lift_long --model lift --init $LFPIV_WORK/runs/lift_long_a --epochs 20 --lr 3e-05` |
| `lift_long_a` | `python train.py --name lift_long_a --model lift --init $LFPIV_WORK/runs/lift_s0` |
| `lift_n120` | `python train.py --name lift_n120 --model lift --epochs 160 --lr 0.0001 --train-fields 120 --val-every 8` |
| `lift_s0` | `python train.py --name lift_s0 --model lift` |
| `lift_s1` | `python train.py --name lift_s1 --seed 1 --model lift` |
| `mc_lift` | `python train.py --name mc_lift --model lift --multi 0.8,0.9,1.0,1.1` |
| `mc_liftcal` | `python train.py --name mc_liftcal --model liftcal --multi 0.8,0.9,1.0,1.1` |
| `mc_liftcal_s1` | `python train.py --name mc_liftcal_s1 --seed 1 --model liftcal --multi 0.8,0.9,1.0,1.1` |
| `mc_projlift` | `python train.py --name mc_projlift --model projlift --multi 0.8,0.9,1.0,1.1` |
| `mc_v3` | `python train.py --name mc_v3 --model v3 --multi 0.8,0.9,1.0,1.1` |
| `mc_viewlift` | `python train.py --name mc_viewlift --model viewlift --multi 0.8,0.9,1.0,1.1` |
| `mc_viewlift_s1` | `python train.py --name mc_viewlift_s1 --seed 1 --model viewlift --multi 0.8,0.9,1.0,1.1` |
| `p7_aug` | `python train.py --name p7_aug --model v3 --init p7 --epochs 20 --lr 3e-05 --augment` |
| `p7_ctrl` | `python train.py --name p7_ctrl --model v3 --init p7 --epochs 20 --lr 3e-05` |
| `p7_mixconc` | `python train.py --name p7_mixconc --model v3 --init p7 --epochs 8 --lr 3e-05 --data <archive of mixed seeding density>` |
| `p7_node` | `python train.py --name p7_node --model node --init p7 --epochs 20 --lr 3e-05` |
| `p7_node_d128` | `python train.py --name p7_node_d128 --model node --init p7 --epochs 12 --lr 3e-05 --depth 128` |
| `projlift_long` | `python train.py --name projlift_long --model projlift --init $LFPIV_WORK/runs/projlift_long_a --epochs 20 --lr 3e-05` |
| `projlift_long_a` | `python train.py --name projlift_long_a --model projlift --init $LFPIV_WORK/runs/projlift_s0` |
| `projlift_n120` | `python train.py --name projlift_n120 --model projlift --epochs 160 --lr 0.0001 --train-fields 120 --val-every 8` |
| `projlift_s0` | `python train.py --name projlift_s0 --model projlift` |
| `projlift_s1` | `python train.py --name projlift_s1 --seed 1 --model projlift` |
| `seed1` | `python train.py --name seed1 --seed 1 --model v3` |
| `seed2` | `python train.py --name seed2 --seed 2 --model v3` |
| `student_projlift` | `python train.py --name student_projlift --model projlift --data $LFPIV_WORK/data/ds9_pseudo` |
| `student_v3` | `python train.py --name student_v3 --model v3 --data $LFPIV_WORK/data/ds9_pseudo` |
| `v3_long` | `python train.py --name v3_long --model v3 --init $LFPIV_WORK/runs/v3_long_a --epochs 20 --lr 3e-05` |
| `v3_long_a` | `python train.py --name v3_long_a --model v3 --init $LFPIV_WORK/runs/seed1` |
| `v3_n120` | `python train.py --name v3_n120 --model v3 --epochs 160 --lr 0.0001 --train-fields 120 --val-every 8` |
| `viewlift_f120` | `python train.py --name viewlift_f120 --model viewlift --multi 1.2` |
| `viewlift_n120` | `python train.py --name viewlift_n120 --model viewlift --epochs 160 --lr 0.0001 --train-fields 120 --val-every 8` |
| `viewlift_s0` | `python train.py --name viewlift_s0 --model viewlift` |
| `viewlift_s1` | `python train.py --name viewlift_s1 --seed 1 --model viewlift` |
| `viewlift_w_s0` | `python train.py --name viewlift_w_s0 --model viewlift --lift-hidden 256` |
