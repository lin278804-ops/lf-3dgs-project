"""Recompute every number quoted in the manuscript from source_data/.

Writes source_data/numbers.json. Each entry is either recomputed here from
per-field or per-particle records, or copied from an archived experiment
summary (flagged with the file it came from).
"""
from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np

from lfmetrics import match
from lfstyle import DATA, boot_ci, load_json, read_csv

N = {}


def put(key, value, source):
    N[key] = {"value": value, "source": source}


def r(x, k=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in x]
    return round(float(x), k)


# ---------------------------------------------------------------- Fig. 2
audit = load_json("fig2/gaussian_fitting_8fields.json")["records"]
for name, key in (("backprojection", "backprojection"), ("fit", "stage1"), ("network8", "network")):
    f1 = np.array([a[key]["f1"] for a in audit])
    put(f"f1_{name}_8fields", r(f1.mean(), 5), "fig2/gaussian_fitting_8fields.json")
    put(f"f1_{name}_8fields_range", r([f1.min(), f1.max()]), "fig2/gaussian_fitting_8fields.json")
    put(f"precision_{name}_8fields", r(np.mean([a[key]["precision"] for a in audit])), "same")
    put(f"recall_{name}_8fields", r(np.mean([a[key]["recall"] for a in audit])), "same")
    put(f"err3d_mm_{name}_8fields", r(np.mean([a[key]["mean_dist"] for a in audit])), "same")
    for ax in "xyz":
        put(f"sigma_{ax}_um_{name}_8fields", r(1000 * np.mean([a[key][f"sigma_{ax}"] for a in audit]), 1), "same")
        put(f"bias_{ax}_um_{name}_8fields", r(1000 * np.mean([a[key][f"bias_{ax}"] for a in audit]), 1), "same")
d = np.array([a["network"]["f1"] - a["stage1"]["f1"] for a in audit])
put("f1_network_minus_fit_8fields", {"mean": r(d.mean()), "ci95": r(boot_ci(d))}, "recomputed, paired field bootstrap")
d = np.array([a["stage1"]["f1"] - a["backprojection"]["f1"] for a in audit])
put("f1_fit_minus_backprojection_8fields", {"mean": r(d.mean()), "ci95": r(boot_ci(d))}, "recomputed")
t_fit = np.array([a["seconds"]["stage1"] for a in audit])
t_net = np.array([a["seconds"]["network"] for a in audit])
put("seconds_fit_8fields", {"mean": r(t_fit.mean(), 2), "min": r(t_fit.min(), 2), "max": r(t_fit.max(), 2)},
    "fig2/gaussian_fitting_8fields.json (includes loading, initialization and output writing)")
put("seconds_network_8fields", r(t_net.mean(), 4), "same audit, host arrays to detections")
rt = load_json("fig2/pipeline_runtime.json")
put("ms_network_one_frame", {"mean": r(rt["decode_nms_one_frame"]["mean_ms"], 1),
                             "sd": r(rt["decode_nms_one_frame"]["std_ms"], 1), "n": rt["decode_nms_one_frame"]["n"]},
    "fig2/pipeline_runtime.json")
put("ms_network_pair_with_linking", {"mean": r(rt["decode_nms_track_one_pair"]["mean_ms"], 1),
                                     "sd": r(rt["decode_nms_track_one_pair"]["std_ms"], 1),
                                     "n": rt["decode_nms_track_one_pair"]["n"]}, "fig2/pipeline_runtime.json")
put("time_ratio_fit_over_network", r(t_fit.mean() / (rt["decode_nms_one_frame"]["mean_ms"] / 1000), 0),
    "mean fit seconds / 158.2 ms; scopes differ, approximate")
n120 = read_csv("fig2/network_120_fields.csv")
for k in ("f1", "precision", "recall", "mean_dist", "sigma_x", "sigma_y", "sigma_z"):
    v = np.array([float(x[k]) for x in n120])
    put(f"network120_{k}", {"mean": r(v.mean()), "sd": r(v.std(ddof=1)), "min": r(v.min()), "max": r(v.max())},
        "fig2/network_120_fields.csv")
put("network120_counts", {k: int(sum(float(x[k]) for x in n120)) for k in ("tp", "fp", "fn")}, "same")

fields8 = sorted(glob.glob(str(DATA / "fig2/fields/sample_*")))
for key, name in (("stage1", "fit"), ("network", "network"), ("backprojection", "backprojection")):
    per = np.zeros((len(fields8), 5))
    lat = np.zeros((len(fields8), 2))
    for i, dd in enumerate(fields8):
        fd8 = np.load(os.path.join(dd, "figure_data.npz"))
        m8 = match(fd8[key], fd8["gt"])
        hit = np.zeros(len(fd8["gt"]), bool)
        hit[m8["gt_idx"]] = True
        az = np.abs(fd8["gt"][:, 2])
        per[i] = [hit[(az >= k) & (az < k + 1)].mean() for k in range(5)]
        ax_ = np.abs(fd8["gt"][:, 0])
        lat[i] = [hit[ax_ < 2].mean(), hit[ax_ >= 9].mean()]
    put(f"recall_by_depth_{name}_8fields", r(100 * per.mean(0), 1), "fig2/fields, recomputed with the archived matching rule")
    put(f"recall_centre_vs_edge_x_{name}_8fields", r(100 * lat.mean(0), 1), "same; |x|<2 mm versus |x|>=9 mm")

# ---------------------------------------------------------------- Fig. 3
rows = read_csv("fig3/lookup_interventions_120_fields.csv")
f1 = collections.defaultdict(list)
cnt = collections.defaultdict(lambda: np.zeros(3, int))
for x in rows:
    f1[x["arm"]].append(float(x["f1"]))
    cnt[x["arm"]] += np.array([int(x["tp"]), int(x["fp"]), int(x["fn"])])
for arm in f1:
    put(f"intervention_f1_{arm}", r(np.mean(f1[arm]), 5), "fig3/lookup_interventions_120_fields.csv")
    put(f"intervention_counts_{arm}", {"tp": int(cnt[arm][0]), "fp": int(cnt[arm][1]), "fn": int(cnt[arm][2]),
                                       "candidates": int(cnt[arm][0] + cnt[arm][1])}, "same")
d = np.array(f1["mismatched"]) - np.array(f1["calibrated"])
put("intervention_reversed_minus_calibrated", {"mean": r(d.mean()), "ci95": r(boot_ci(d))}, "recomputed")
light = read_csv("fig3/observed_light_at_projection.csv")
reg = np.array([float(x["registered"]) for x in light])
rev = np.array([float(x["reversed_labels"]) for x in light])
put("light_registered", r(reg.mean(), 3), "fig3/observed_light_at_projection.csv")
put("light_reversed", r(rev.mean(), 3), "same")
put("light_difference", {"mean": r((reg - rev).mean(), 3), "ci95": r(boot_ci(reg - rev), 3),
                         "fields_favouring_registered": int((reg > rev).sum())}, "recomputed")
fd = load_json("fig3/feature_depth_summary.json")
put("feature_depth", {"eligible": fd["eligible"], "truth": fd["truth"],
                      "cosine_calibrated": r(fd["statistics"]["cosine"]["pairwise"]["calibrated"]),
                      "cosine_reversed": r(fd["statistics"]["cosine"]["pairwise"]["mismatched"]),
                      "variance_calibrated": r(fd["statistics"]["negative_variance"]["pairwise"]["calibrated"]),
                      "variance_reversed": r(fd["statistics"]["negative_variance"]["pairwise"]["mismatched"])},
    "fig3/feature_depth_summary.json")
se = load_json("fig3/single_emitter_raytrace.json")
put("single_emitter", {"observations": se["observations"], "with_light": se["with_light"],
                       "median_px_calibrated": r(se["absolute_centroid_error_native_px"]["error_px"]["median"], 3),
                       "median_px_reversed": r(se["absolute_centroid_error_native_px"]["reverse_error_px"]["median"], 3),
                       "median_px_focus_plane": r(se["absolute_centroid_error_native_px"]["focus_error_px"]["median"], 3)},
    "fig3/single_emitter_raytrace.json")
cc = load_json("fig3/camera_coordinate_check.json")
put("projection_implementations_max_px", cc["errors"]["projection_grid"]["max_native_px"],
    "fig3/camera_coordinate_check.json")

# ---------------------------------------------------------------- Fig. 4
cs = load_json("fig4/camera_cross_summary.json")
for arm in ("AA", "AB", "BA", "BB"):
    a = cs["arms"][arm]
    put(f"cross_{arm}", {"f1": r(a["f1"]["mean"]), "recall": r(a["recall"]["mean"]), "precision": r(a["precision"]["mean"]),
                         "z_mae_um": r(1000 * a["z_mae_mm"]["mean"], 1), **a["total"]}, "fig4/camera_cross_summary.json")
put("cross_BB_minus_BA_f1", {"mean": r(cs["contrasts"]["BB-BA"]["f1"]["mean"]), "ci95": r(cs["contrasts"]["BB-BA"]["f1"]["ci95"])},
    "same")
st = cs["strata"]["BB-BA"]
put("cross_BB_BA_common", st["all"]["total"], "same")
for band in ("abs_z_0_to_1", "abs_z_4_to_5"):
    put(f"cross_recall_gain_{band}", r(st[band]["delta_recall"]["mean"]), "same")
shared = read_csv("fig4/camera_cross_shared.csv")
dz = np.array([float(x["delta_common_z_mae_mm"]) for x in shared if x["contrast"] == "BB-BA" and x["stratum"] == "all"])
put("cross_common_axial_change_um", {"mean": r(1000 * dz.mean(), 2), "ci95": r([1000 * v for v in boot_ci(dz)], 2)},
    "fig4/camera_cross_shared.csv, recomputed")
scan = read_csv("fig4/focal_scan_by_field.csv")
by = collections.defaultdict(lambda: collections.defaultdict(dict))
for x in scan:
    by[x["observed"]][x["scene"]][float(x["lookup"])] = float(x["f1"])
true_f = {"A": 0.80, "C": 0.90, "B": 1.00}
peaks, means = 0, {}
for cam in by:
    for sc in by[cam]:
        peaks += int(max(by[cam][sc], key=by[cam][sc].get) == true_f[cam])
    means[cam] = {str(k): r(np.mean([by[cam][s][k] for s in by[cam]])) for k in sorted(next(iter(by[cam].values())))}
put("focal_scan_peak_at_truth", f"{peaks} of {sum(len(v) for v in by.values())}", "fig4/focal_scan_by_field.csv")
put("focal_scan_means", means, "same")
loc = read_csv("fig4/calibration_location_fields.csv")
for arm in ("old", "output", "lookup"):
    put(f"location_{arm}", {
        "tp": sum(int(x[f"{arm}_tp"]) for x in loc), "fp": sum(int(x[f"{arm}_fp"]) for x in loc),
        "fn": sum(int(x[f"{arm}_fn"]) for x in loc), "pred": sum(int(x[f"{arm}_n_pred"]) for x in loc),
        "recall": r(np.mean([float(x[f"{arm}_recall"]) for x in loc])),
        "precision": r(np.mean([float(x[f"{arm}_precision"]) for x in loc])),
        "f1": r(np.mean([float(x[f"{arm}_f1"]) for x in loc]))}, "fig4/calibration_location_fields.csv")
c3 = {k: 1000 * np.array([float(x[f"common3_{k}_mae_mm"]) for x in loc]) for k in ("old", "output", "lookup")}
c2 = {k: 1000 * np.array([float(x[f"common2_{k}_mae_mm"]) for x in loc]) for k in ("output", "lookup")}
put("location_common3", {"n": sum(int(x["common3_n"]) for x in loc), **{k: r(v.mean(), 2) for k, v in c3.items()},
                         "output_minus_lookup": r((c3["output"] - c3["lookup"]).mean(), 2),
                         "ci95": r(boot_ci(c3["output"] - c3["lookup"]), 2)}, "same, recomputed")
put("location_common2", {"n": sum(int(x["common2_n"]) for x in loc), **{k: r(v.mean(), 2) for k, v in c2.items()},
                         "lookup_minus_output": r((c2["lookup"] - c2["output"]).mean(), 2),
                         "ci95": r(boot_ci(c2["lookup"] - c2["output"]), 2)}, "same, recomputed")
rec = {arm: np.zeros((24, 5, 2)) for arm in ("old", "output", "lookup")}
frac_ok = []
for i, f in enumerate(sorted(glob.glob(str(DATA / "fig4/calibration_location_particles/scene_*.npz")))):
    p = np.load(f)
    az = np.abs(p["gt_xyz"][:, 2])
    for arm in rec:
        hit = p[f"{arm}_pred_for_gt"] >= 0
        for k in range(5):
            m = (az >= k) & (az < k + 1)
            rec[arm][i, k] = [hit[m].sum(), m.sum()]
for arm in rec:
    put(f"location_recall_by_depth_{arm}", r(100 * (rec[arm][:, :, 0] / rec[arm][:, :, 1]).mean(0), 2),
        "fig4/calibration_location_particles, recomputed")
eq = load_json("fig4/calibration_location_equal_fp.json")
put("location_equal_fp_note", "see fig4/calibration_location_equal_fp.json", "archived")
ps = load_json("fig4/plane_sweep_vs_network.json")
for arm in ("AA", "BB"):
    put(f"plane_sweep_{arm}", {"f1_sweep": r(ps[arm]["classical"]["f1"]["mean"]), "f1_network": r(ps[arm]["p7"]["f1"]["mean"]),
                               "recall_equal_fp_sweep": r(ps[arm]["equal_fp"]["p7_official"]["classical"]["mean"]),
                               "recall_equal_fp_network": r(ps[arm]["equal_fp"]["p7_official"]["p7"]["mean"]),
                               "common_ids": ps[arm]["common_truth_ids"],
                               "common_z_mae_um_sweep": r(1000 * ps[arm]["classical"]["common_z_mae_mm"]["mean"], 1),
                               "common_z_mae_um_network": r(1000 * ps[arm]["p7"]["common_z_mae_mm"]["mean"], 1)},
        "fig4/plane_sweep_vs_network.json")
sw = read_csv("fig4/plane_sweep_fields.csv")
for arm in ("old", "output_xyz", "sampling"):
    rs = [x for x in sw if x["arm"] == arm]
    put(f"plane_sweep_location_{arm}", {"f1": r(np.mean([float(x["f1"]) for x in rs])),
                                        "recall": r(np.mean([float(x["recall"]) for x in rs])),
                                        "tp": sum(int(x["tp"]) for x in rs), "fp": sum(int(x["fp"]) for x in rs)},
        "fig4/plane_sweep_fields.csv")

# ---------------------------------------------------------------- Fig. 5
links = collections.defaultdict(list)
for dd in sorted(glob.glob(str(DATA / "fig5/motion_links/scene_*"))):
    for cond in ("x_plus", "z_plus", "z_minus", "still"):
        z = np.load(os.path.join(dd, f"{cond}.npz"))
        links[cond].append((z["error_mm_B"], z["correct_B"]))
STEP = 0.2
for cond, comp, sg in (("x_plus", 0, 1), ("z_plus", 2, 1), ("z_minus", 2, -1)):
    fa = np.array([np.mean(sg * e[:, comp] + STEP) for e, ok in links[cond]])
    fo = np.array([np.mean(sg * e[ok, comp] + STEP) for e, ok in links[cond]])
    put(f"motion_{cond}", {"n_scoreable": int(sum(len(e) for e, _ in links[cond])),
                           "correct_link_fraction": r(np.mean(np.concatenate([ok for _, ok in links[cond]]))),
                           "mean_measured_mm_all": r(sg * fa.mean()), "recovered_all": r(fa.mean() / STEP),
                           "recovered_correct": r(fo.mean() / STEP)}, "fig5/motion_links, recomputed")
za = np.array([np.mean([np.mean(sg * e[:, 2] + STEP) for e, _ in [links[c][i]]]) for i in range(24)
               for c, sg in (("z_plus", 1), ("z_minus", -1))]).reshape(24, 2).mean(1) / STEP
zo = np.array([np.mean(sg * links[c][i][0][links[c][i][1], 2] + STEP) for i in range(24)
               for c, sg in (("z_plus", 1), ("z_minus", -1))]).reshape(24, 2).mean(1) / STEP
put("motion_axial_recovered", {"all": r(za.mean()), "all_ci95": r(boot_ci(za)), "correct": r(zo.mean()),
                               "correct_ci95": r(boot_ci(zo))}, "recomputed, signs pooled within field")
still = links["still"]
put("motion_still", {"n_scoreable": int(sum(len(e) for e, _ in still)),
                     "epe_mm": r(np.mean([np.linalg.norm(e, axis=1).mean() for e, _ in still])),
                     "mae_um_xyz": r(1000 * np.mean([np.abs(e).mean(0) for e, _ in still], axis=0), 1)},
    "fig5/motion_links, recomputed")
# quantization of axial displacement to the depth-bin spacing
BIN = 10.0 / 64
allz = np.concatenate([sg * e[:, 2] + STEP for c, sg in (("z_plus", 1), ("z_minus", -1)) for e, _ in links[c]])
resid = np.abs(allz / BIN - np.round(allz / BIN))
put("motion_axial_quantization", {"bin_mm": BIN, "n": int(len(allz)),
                                  "fraction_within_0.1_bin_of_integer": r(np.mean(resid < 0.1)),
                                  "expected_if_uniform": 0.2,
                                  "share_at_0_1_2_bins": r([np.mean(np.round(allz / BIN) == k) for k in (0, 1, 2)])},
    "fig5/motion_links, recomputed")
ea = load_json("fig5/endpoint_attribution.json")["axial_signed_pooled"]
put("motion_tracks", {"endpoint_recovery_old": r(ea["metrics"]["endpoint_recovery"]["BA"]["mean"]),
                      "endpoint_recovery_matched": r(ea["metrics"]["endpoint_recovery"]["BB"]["mean"]),
                      "conditional_link_old": r(ea["metrics"]["conditional_correct_link"]["BA"]["mean"]),
                      "conditional_link_matched": r(ea["metrics"]["conditional_correct_link"]["BB"]["mean"]),
                      "track_recall_old": r(ea["metrics"]["actual_track_recall"]["BA"]["mean"]),
                      "track_recall_matched": r(ea["metrics"]["actual_track_recall"]["BB"]["mean"]),
                      "identity_axial_recovered_matched": r(ea["metrics"]["oracle_axis_recovered_fraction"]["BB"]["mean"]),
                      "totals_matched": ea["totals"]["BB"], "totals_old": ea["totals"]["BA"]},
    "fig5/endpoint_attribution.json")
ms = load_json("fig5/motion_summary.json")["conditions"]["axial_signed_pooled"]["metrics"]
put("motion_common_tracks", {"dz_mae_um_old": r(1000 * ms["common_correct_dz_mae_mm"]["BA"], 1),
                             "dz_mae_um_matched": r(1000 * ms["common_correct_dz_mae_mm"]["BB"], 1),
                             "ci95_um": r([1000 * v for v in ms["common_correct_dz_mae_mm"]["paired_interval_95"]], 1),
                             "epe_mm_old": r(ms["epe3d_mm"]["BA"]), "epe_mm_matched": r(ms["epe3d_mm"]["BB"])},
    "fig5/motion_summary.json")
# sub-voxel offsets: are they informative?
corr = []
frac_z = []
for f in sorted(glob.glob(str(DATA / "fig4/camera_cross_predictions/AA/scene_*.npz"))):
    p = np.load(f)
    t = np.load(str(DATA / "fig4/camera_cross_truth" / os.path.basename(f)))
    gz, pz = t["gt_vox"][p["gt_idx"], 0], p["vox"][p["pred_idx"], 0]
    same = np.floor(gz) == np.floor(pz)
    corr.append(np.corrcoef(gz[same] % 1, pz[same] % 1)[0, 1])
    frac_z.append(p["vox"][:, 0] % 1)
frac_z = np.concatenate(frac_z)
put("subvoxel_depth", {"corr_true_vs_predicted_fraction_same_bin": r(np.mean(corr), 3),
                       "sd_predicted_fraction": r(frac_z.std(), 3), "sd_uniform": r(1 / np.sqrt(12), 3)},
    "fig4/camera_cross_predictions/AA, recomputed")
fl = read_csv("fig5/flow_families_independent.csv")
put("flow_families", {"pairs": len(fl), "epe_mm": r(np.mean([float(x["epe3d_mm"]) for x in fl])),
                      "track_recall": r(np.mean([float(x["track_recall"]) for x in fl])),
                      "f1": r(np.mean([float(x["f1"]) for x in fl]))}, "fig5/flow_families_independent.csv")
T = load_json("fig5/flow_families_temporal_models.json")["diagnostic_comparison"]
put("flow_families_postprocessing", {"smoothing_epe_mm": r(T["control_summary"]["epe3d_mm"]),
                                     "learned_temporal_epe_mm": r(T["candidate_summary"]["epe3d_mm"])},
    "fig5/flow_families_temporal_models.json")

# ---------------------------------------------------------------- Fig. 6
ang = {x["method"]: x for x in read_csv("fig6/target_angular_prediction.csv")}
put("target_angular", {"readable": int(ang["constant"]["readable_count"]), "eligible": int(ang["constant"]["total_opportunities"]),
                       **{k: r(float(ang[m]["readable"]), 4) for k, m in (("constant", "constant"), ("affine", "linear"),
                                                                           ("quadratic", "quadratic"),
                                                                           ("quadratic_optical", "quadratic_mixed"))}},
    "fig6/target_angular_prediction.csv")
mag = load_json("fig6/target_step_magnitude.json")
put("target_steps", {k: {"mean": r([mag[k]["matched_all"][i]["mean"] for i in range(3)], 3),
                         "mae": r([mag[k]["matched_all"][i]["mae"] for i in range(3)], 3),
                         "strict_order": r(mag[k]["matched_trends"]["strict_order"])}
                     for k in ("gaussian_pair", "weighted_centroid", "original_shared")} | {"n": mag["matched_n"]},
    "fig6/target_step_magnitude.json")
fit = load_json("fig6/target_blind_fitting_baseline.json")
net = load_json("fig6/target_blind_network.json")["shared"]["pair"]
stg = load_json("fig6/target_blind_network_stages.json")["counts"]
put("target_blind", {"opportunities": fit["all_opportunities"], "verifiable": fit["admitted"],
                     "fit_correct": fit["status"]["correct_id_and_direction"], "fit_links": fit["predicted_links"],
                     "network_correct": net["correct"], "network_links": net["anonymous_links"], "network_stages": stg},
    "fig6/target_blind_*.json")
tank = read_csv("fig6/tank_proposals.csv")
put("tank_proposals", {"min": min(int(x["n_pred_min"]) for x in tank), "max": max(int(x["n_pred_max"]) for x in tank),
                       "sequences": len(tank)}, "fig6/tank_proposals.csv")
near = load_json("fig6/tank_nearest_view_reference.json")["records"]
put("tank_heldout_nmse", {"network_reconstruction": 0.9997, "nearest_view": r(np.mean([x["heldout_nmse_mean"] for x in near])),
                          "pairs": len(near)}, "fig6/tank_heldout_view_findings.txt; fig6/tank_nearest_view_reference.json")
pl = read_csv("fig6/planar_closure.csv")
put("planar_closure", {"pairs": len(pl), "median_px": r(np.median([float(x["forward_backward_closure_px_median"]) for x in pl])),
                       "mean_coverage": r(np.mean([float(x["closure_assessed_fraction_of_all_vectors"]) for x in pl]))},
    "fig6/planar_closure.csv")

# ---------------------------------------------------------------- Extended Data
for k in ("ablation_no_hard_negatives", "ablation_broadcast_skip", "ablation_p6_parent", "p7_test_hungarian_nms"):
    a = load_json(f"extended/{k}.json")
    put(k, {"f1": r(a["mean_f1"]), "precision": r(a["mean_precision"]), "recall": r(a["mean_recall"]),
            "sigma_z_um": r(1000 * a["sigma_z"], 1)}, f"extended/{k}.json")
dv = load_json("extended/learned_lookup_summary.json")["camera_results"]
put("learned_lookup", {cam: {"analytic": r(dv[cam]["analytic_mean_f1"]), "learned": r(dv[cam]["learned_mean_f1"]),
                             "diff_pp": r(100 * dv[cam]["learned_minus_analytic_f1"], 2),
                             "ci95_pp": r([100 * v for v in dv[cam]["paired_field_ci95"]], 2)} for cam in ("A", "B")},
    "extended/learned_lookup_summary.json")
rc = load_json("extended/render_loss_cross_summary.json")
put("render_loss_cross", {"common": rc["common"], "interaction_um": r(rc["contrasts"]["axial_mae_um"]["interaction"]["mean"], 2),
                          "interaction_ci95": r(rc["contrasts"]["axial_mae_um"]["interaction"]["ci95"], 2),
                          "matched_on_benefit_um": r(rc["contrasts"]["axial_mae_um"]["matched_on_benefit"]["mean"], 2),
                          "matched_recall_diff_pp": r(100 * rc["contrasts"]["recall"]["matched_on_benefit"]["mean"], 2)},
    "extended/render_loss_cross_summary.json")
pcmp = load_json("extended/render_loss_paired_comparison.json")["comparisons"]
put("render_loss_paired", {"f1_on": r(pcmp["static_best"]["f1"]["on"]), "f1_off": r(pcmp["static_best"]["f1"]["off"]),
                           "z_mae_um_diff": r(1000 * pcmp["static_best"]["z_mae_mm"]["paired_on_minus_off"], 2),
                           "z_mae_um_ci95": r([1000 * v for v in pcmp["static_best"]["z_mae_mm"]["ci95"]], 2),
                           "track_recall_on": r(pcmp["motion_best"]["track_recall"]["on"]),
                           "track_recall_off": r(pcmp["motion_best"]["track_recall"]["off"])},
    "extended/render_loss_paired_comparison.json")

(DATA / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
for k, v in N.items():
    print(f"{k}: {json.dumps(v['value'], ensure_ascii=False)}")
