"""Numbers quoted for the sealed-field, robustness, displacement and recorded-pair experiments.

Reads the unmodified experiment records in source_data/gap/ and source_data/gapA/ and writes
source_data/gap_numbers.json ({key: {"value": ..., "source": ...}}). Entries whose records are
absent are omitted, so the manuscript check reports them as missing instead of guessing.
"""
from __future__ import annotations

import json

import numpy as np

from lfstyle import DATA, boot_ci

RES = DATA / "gap" / "results"
OUT = {}


def put(key, value, source):
    OUT[key] = {"value": value, "source": source}


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f1(name, key=None):
    r = load(name)
    if not r:
        return None
    v = np.array([f["f1"] for f in r["fields"]])
    s = r["summary"]
    put(key or f"f1__{name}", dict(mean=float(v.mean()), ci95=boot_ci(v), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                   n_fields=len(v), precision=s["precision"], recall=s["recall"],
                                   lateral_f1=s.get("lateral_f1"), lateral_dz_median_mm=s.get("lateral_abs_dz_median_mm"),
                                   err_sd_um=s.get("err_sd_um"), err_bias_um=s.get("err_bias_um"),
                                   seconds_per_field=s.get("seconds_per_field")), f"gap/results/{name}.json")
    return v


def main():
    # ---- baselines and runs on the sealed fields
    for n in ("p7__sealed_std", "p7__sealed_std__heatmap", "p7__ds9test", "fit__sealed_std", "backprojection__sealed_std",
              "planesweep__sealed_std", "mart32__sealed_std", "mart484__sealed_std", "mart484_up2__sealed_std",
              "scratch_s0__sealed_std", "seed1__sealed_std", "seed2__sealed_std", "lift_s0__sealed_std", "v3_n120__sealed_std",
              "lift_n120__sealed_std", "p7_ctrl__sealed_std", "p7_node__sealed_std", "p7_node_d128__sealed_std",
              "p7_aug__sealed_std", "p7_mixconc__sealed_std", "p7__sealed_std__cond", "lift_s0__sealed_std__cond",
              "p7__sealed_camB", "p7__sealed_camB__f1", "p7__sealed_camB__conv1", "lift_s0__sealed_camB",
              "lift_s0__sealed_camB__conv1", "v3_n120__sealed_camB__conv1", "p7_node__sealed_camB__conv1"):
        f1(n)
    for run in ("projlift_s0", "projlift_s1", "lift_s1", "projlift_n120"):
        for sfx in ("", "__heatmap", "__cond", "__snr30__cond"):
            f1(f"{run}__sealed_std{sfx}")
        for ds in ("sealed_conc050", "sealed_conc075", "sealed_diam30", "sealed_diam70"):
            f1(f"{run}__{ds}")
    for run in ("seed1", "seed2"):
        for nm in ("sealed_std__cond", "sealed_std__snr30__cond", "sealed_conc050", "sealed_conc075"):
            f1(f"{run}__{nm}")
    for run in ("viewlift_s0", "viewlift_s1", "viewlift_w_s0", "viewlift_n120", "v3_long", "v3_long_a", "projlift_long", "projlift_long_a",
                "lift_long", "lift_long_a", "student_v3", "student_projlift"):
        for sfx in ("sealed_std", "sealed_std__heatmap", "sealed_std__cond", "sealed_std__snr20__cond", "sealed_std__snr30__cond",
                    "sealed_std__snr40__cond", "sealed_conc050", "sealed_conc075", "sealed_diam30", "sealed_diam70", "sealed_std__reverse",
                    "sealed_camB", "sealed_camB__f1", "sealed_camB__conv1", "sealed_cam090", "sealed_cam090__f0.9", "sealed_cam090__conv0.9",
                    "sealed_cam110", "sealed_cam110__f1.1", "sealed_cam110__conv1.1"):
            f1(f"{run}__{sfx}")
    for run in ("p7", "projlift_s0", "lift_s0"):
        f1(f"{run}__sealed_std__reverse")
    for run in ("mc_viewlift", "mc_viewlift_s1", "mc_liftcal", "mc_liftcal_s1", "mc_lift", "mc_v3", "mc_projlift"):
        for sfx in ("sealed_std", "sealed_cam085__f0.85", "sealed_cam090__f0.9", "sealed_cam095__f0.95", "sealed_camB__f1",
                    "sealed_cam105__f1.05", "sealed_cam110__f1.1", "sealed_std__cond", "sealed_std__snr30__cond"):
            f1(f"{run}__{sfx}")
        for sfx in ("sealed_cam070__f0.7", "sealed_cam075__f0.75", "sealed_cam115__f1.15", "sealed_cam120__f1.2", "sealed_F095__F95",
                    "sealed_F105__F105", "sealed_L190__L190", "sealed_L210__L210", "sealed_F095_f095__f0.95__F95",
                    "sealed_F105_f095__f0.95__F105", "sealed_F095__f0.97729__F95__feq", "sealed_F105__f0.65488__F105__feq",
                    "sealed_L190__f0.648__L190__feq", "sealed_L210__f0.968__L210__feq", "sealed_F095_f095__f1.16053__F95__feq",
                    "sealed_F105_f095__f0.77767__F105__feq", "sealed2_std"):
            f1(f"{run}__{sfx}")
    for run in ("p7", "scratch_s0", "seed1", "seed2", "projlift_s0", "projlift_s1", "lift_s0", "lift_s1", "viewlift_s0", "viewlift_s1",
                "viewlift_w_s0"):
        f1(f"{run}__sealed2_std")
    f1("viewlift_f120__sealed_cam120__f1.2")
    fp = {}
    for run in ("p7", "ab_skip_bcast_s0"):
        for sfx in ("", "__f1", "__conv1"):
            r = load(f"{run}__sealed_camB{sfx}__fpsrc")
            if r:
                fp[f"{run}{sfx or '__raw'}"] = dict(f1=r["summary"]["f1"], tp=r["summary"]["tp"], fp=r["summary"]["fp"])
    r = load("ab_skip_bcast_s0__sealed_std__fpsrc")
    if fp and r:
        put("lookup_fp_source", dict(runs=fp, bcast_sealed_f1=r["summary"]["f1"],
                                     bcast_conv_minus_lookup=fp["ab_skip_bcast_s0__conv1"]["f1"] - fp["ab_skip_bcast_s0__f1"]["f1"],
                                     principal_conv_minus_lookup=fp["p7__conv1"]["f1"] - fp["p7__f1"]["f1"]),
            "gap/results/*__sealed_camB*__fpsrc.json")
    sp = {}
    for key, nm in (("d245", "plate__p7__heatmap__simplate_d245"), ("d050", "plate__p7__heatmap__simplate_d050"),
                    ("d245_raw", "plate__p7__heatmap__raw__simplate_d245")):
        r = load(nm)
        if r:
            sp[key] = {k: r["summary"][k] for k in ("depth_gain", "depth_linearity_residual_mm", "depth_sd_within_plane_mm_median",
                                                     "holes_found_median", "planes")}
    if sp:
        put("sim_plate", sp, "gap/results/plate__p7__heatmap__simplate_*.json")
    p = DATA / "gap" / "gatefree_axial.json"
    if p.exists():
        put("gatefree_axial", json.loads(p.read_text()), "gap/gatefree_axial.json")
    for run in ("p7", "projlift_s0", "projlift_s1", "lift_s0", "lift_s1", "projlift_n120", "v3_n120", "lift_n120", "seed1", "seed2"):
        for cam, fl in (("camB", "1"), ("cam090", "0.9"), ("cam110", "1.1")):
            for sfx in ("", f"__f{fl}", f"__conv{fl}"):
                f1(f"{run}__sealed_{cam}{sfx}")
    for ds in ("sealed_conc050", "sealed_conc075", "sealed_diam30", "sealed_diam70"):
        for run in ("p7", "p7_mixconc", "p7_node", "p7_aug", "lift_s0"):
            f1(f"{run}__{ds}")
        f1(f"p7__{ds}__topk4000")
    for q in (20, 25, 30, 35, 40, 50):
        for run, sfx in (("p7", ""), ("p7", "__cond"), ("p7_aug", ""), ("p7_node", ""), ("lift_s0", ""), ("lift_s0", "__cond"),
                         ("fit", ""), ("p7_mixconc", "")):
            f1(f"{run}__sealed_std__snr{q}{sfx}")
    a, b = f1("p7__sealed_std", "tmp"), f1("fit__sealed_std", "tmp")
    OUT.pop("tmp", None)
    if a is not None and b is not None:
        put("network_minus_fit_sealed", dict(mean=float((a - b).mean()), ci95=boot_ci(a - b)), "paired over 120 sealed fields")
    runs = [OUT[k]["value"]["mean"] for k in ("f1__scratch_s0__sealed_std", "f1__seed1__sealed_std", "f1__seed2__sealed_std") if k in OUT]
    if len(runs) == 3:
        put("independent_runs_sealed", dict(values=runs, mean=float(np.mean(runs)), sd=float(np.std(runs, ddof=1)),
                                            range=[float(min(runs)), float(max(runs))]), "three trainings from random weights")
    tune = load("mart_tuning")
    if tune:
        put("mart_settings", tune["chosen"], "gap/results/mart_tuning.json")
    tim = load("timing")
    if tim:
        put("timing_ms", {k: dict(mean=v["ms_mean"], sd=v["ms_sd"], n_params=v["n_params"]) for k, v in tim.items()},
            "gap/results/timing.json")
    for run in ("lift_s0", "v3_n120", "lift_n120", "seed1", "seed2", "p7_node", "p7_ctrl", "p7_aug", "p7_mixconc", "p7_node_d128"):
        p = DATA / "gap" / "runs" / run / "trained.json"
        m = DATA / "gap" / "runs" / run / "meta.json"
        if p.exists() and m.exists():
            t, mm = json.loads(p.read_text()), json.loads(m.read_text())
            put(f"run__{run}", dict(epochs=t["epochs"], best_epoch=t["best_epoch"], best_val_f1=t["best_val_f1"],
                                    n_params=mm["n_params"], n_train=mm["n_train"]), f"gap/runs/{run}")

    # ---- displacement (written by fig5_motion.py) and decoding
    p = DATA / "gap" / "fig5_numbers.json"
    if p.exists():
        put("motion", json.loads(p.read_text()), "gap/fig5_numbers.json (from gap/results/motion__*)")
    for nm, key in (("motion__p7", "head"), ("motion__p7__heatmap", "peak")):
        f = RES / f"{nm}.npz"
        if f.exists():
            z = np.load(f)
            m = np.isin(z["motion"], [list(z["motions"]).index(k) for k in ("z_p020", "z_m020")])
            dz = z["meas"][m, 2] / (10 / 64)
            put(f"plane_locked_share_{key}", float((np.abs(dz - np.round(dz)) < 0.1).mean()), f"gap/results/{nm}.npz, steps of 0.20 mm")
    for n in ("p7__sealed_std__gain0.7", "p7__sealed_std__gain1.3"):
        f1(n)
    for nm in ("motion__p7__heatmap", "motion__p7", "motion__fit", "motion__p7_node", "motion__p7_ctrl", "motion__p7_node_d128",
               "motion__lift_s0__heatmap", "motion__projlift_s0__heatmap", "motion__viewlift_s0__heatmap", "motion__viewlift_s1__heatmap"):
        r = load(nm)
        if r:
            put(nm, {k: {q: v[q] for q in ("track_recall", "transfer", "blind_correct", "blind_transfer_all", "sd_disp_mm", "on_plane")
                         if q in v} for k, v in r["summary"].items()}, f"gap/results/{nm}.json")

    # ---- recorded pairs, plate and scale
    for nm in ("real__p7__heatmap", "real__p7", "real__fit", "real__lift_s0__heatmap", "real__projlift_s0__heatmap",
               "real__viewlift_s0__heatmap", "real__viewlift_s1__heatmap"):
        r = load(nm)
        if r and "all" in r["summary"]:
            rows = [q for q in r["pairs"] if q.get("tracks")]
            geo = json.loads((DATA / "gapA/camera_geometry.json").read_text())["plate_june"]
            mps = geo["um_per_sai_px"] * 1e-6 / 1e-5
            sp = np.array([np.hypot(q["track_row"], q["track_col"]) for q in rows])
            ctrl = [q["ctrl_tracks"] for q in rows if "ctrl_tracks" in q]
            by = {g: float(np.median(sp[[q["group"] == g for q in rows]]) * mps) for g in ("30", "40", "50")}
            put(nm, dict(r["summary"]["all"], pairs_total=r["summary"]["pairs"], speed_mps_median=float(np.median(sp) * mps),
                         speed_mps_iqr=[float(v * mps) for v in np.percentile(sp, [25, 75])], speed_mps_median_by_recording=by,
                         displacement_px_median=float(np.median(sp)), ctrl_tracks_median=float(np.median(ctrl)) if ctrl else None,
                         clear_reference=r["summary"].get("clear_reference")), f"gap/results/{nm}.json")
    r = load("plate__p7__heatmap")
    if r:
        near = [q for q in r["planes"] if -2.5 <= q["stage_mm"] <= 1.5]
        put("plate_network", dict(summary=r["summary"], near_focus=dict(
            stage_range_mm=[-2.5, 1.5], holes_found=[min(q["holes_found"] for q in near), max(q["holes_found"] for q in near)],
            lateral_residual_um=[min(q["lateral_residual_um"] for q in near), max(q["lateral_residual_um"] for q in near)],
            depth_median_mm=[min(q["z_median"] for q in near), max(q["z_median"] for q in near)],
            um_per_px=float(99.0 / np.median([q["scale_nominal_per_mm"] for q in near])))), "gap/results/plate__p7__heatmap.json")
    r = load("real__p7__heatmap__realtank")
    if r:
        n = [q[k] for q in r["pairs"] for k in ("n0", "n1")]
        put("tank_conditioned_candidates", dict(frames=len(n), min=int(min(n)), max=int(max(n))), "gap/results/real__p7__heatmap__realtank.json")
    for nm in ("camera_geometry", "bench_speed", "scale_check"):
        p = DATA / "gapA" / f"{nm}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if nm == "camera_geometry":
                put("plate_geometry", {k: v for k, v in d["plate_june"].items() if k != "rows"}, "gapA/camera_geometry.json")
            elif nm == "bench_speed":
                put("recorded_pairs_correlation", {g: dict(pairs=v["pairs"], ensemble=v["views"]["centre"]["ensemble"],
                                                           mismatched=v["views"]["centre"]["mismatched_pairs"],
                                                           per_pair=v["per_pair_clear"], expected_px_if_label_is_mps=v["expected_px_if_label_is_mps"],
                                                           expected_px_plate_scale=v["expected_px_if_label_is_mps_plate_scale"])
                                                       for g, v in d["groups"].items()}, "gapA/bench_speed.json")
            else:
                put("planar_camera", d["planar_camera"], "gapA/scale_check.json")
    (DATA / "gap_numbers.json").write_text(json.dumps(OUT, indent=1))
    print(len(OUT), "entries")
    for k in ("f1__p7__sealed_std", "f1__fit__sealed_std", "f1__lift_s0__sealed_std", "independent_runs_sealed", "real__p7__heatmap"):
        if k in OUT:
            print(k, json.dumps(OUT[k]["value"])[:420])


if __name__ == "__main__":
    main()
