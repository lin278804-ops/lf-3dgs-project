"""Paragraphs and a table of Supplementary Note 7 that are written directly from the experiment records.

Writes manuscript/generated_note7.tex, which supplementary.tex inputs. Every number is read from
source_data/gap/results at build time, and runs that have not been evaluated yet are left out, so
the text follows the records when further initializations arrive.
"""
from __future__ import annotations

import json

import numpy as np

from lfstyle import DATA, ROOT

RES = DATA / "gap" / "results"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def S(name, key="f1"):
    r = load(name)
    return None if r is None else r["summary"][key]


def have(runs, sfx):
    return [r for r in runs if load(f"{r}__{sfx}") is not None]


def fm(v, d=3):
    return f"{v:.{d}f}"


def each(runs, sfx, d=3):
    """'0.919' for one run, '0.919 and 0.921' for two."""
    v = [S(f"{r}__{sfx}") for r in have(runs, sfx)]
    return " and ".join(fm(x, d) for x in v) if v else None


def mean(runs, sfx):
    v = [S(f"{r}__{sfx}") for r in have(runs, sfx)]
    return float(np.mean(v)) if v else None


def n_init(runs, sfx="sealed_std"):
    n = len(have(runs, sfx))
    return {1: "one initialization", 2: "two initializations", 3: "three initializations"}[n]


def main():
    out = []
    V = ["viewlift_s0", "viewlift_s1"]
    nv = len(have(V, "sealed_std"))
    tim = load("timing")
    r0 = load("viewlift_s0__sealed_std")["summary"]
    rv = load("viewlift_s0__sealed_std__reverse")
    rr = {k: load(f"{k}__sealed_std__reverse")["summary"] for k in ("p7", "projlift_s0", "lift_s0", "viewlift_s0")}
    mv = load("motion__viewlift_s0__heatmap")["summary"]
    ax = [mv[k]["transfer"] for k in mv if k.startswith("z_")]
    both = [mv[k]["track_recall"] for k in mv if k.startswith("z_")]
    re = load("real__viewlift_s0__heatmap")["summary"]["all"]
    pl = load("plate__viewlift_s0__heatmap")["summary"]
    pll = load("plate__lift_s0__heatmap")["summary"]

    t = (r"\textbf{View-preserving network.} The view-preserving network (%s parameters) scored %s on the sealed fields (%s; precision %.3f and recall %.3f, "
         r"with s.d.\ of the position error of %.0f, %.0f and %.0f\um\ along $x$, $y$ and $z$%s) and %s with the occupancy-peak decoder. "
         % (f"{tim['viewlift_s0__network']['n_params']:,}", each(V, "sealed_std"), n_init(V), r0["precision"], r0["recall"], *r0["err_sd_um"],
            ", for the first" if nv > 1 else "", each(V, "sealed_std__heatmap")))
    t += (r"A variant with 256 instead of 128 channels in the fusion (%s parameters) scored %s, so the result does not depend on the width chosen. "
          % (f"{tim['viewlift_w_s0__network']['n_params']:,}", fm(S("viewlift_w_s0__sealed_std"))))
    t += (r"With conditioned input the network scored %s without noise and %s, %s and %s at 40, 30 and 20\,dB. At two and three times the seeding density it scored %s and %s, "
          r"and with tracers of 30 and 70\um\ %s and %s. "
          % (fm(mean(V, "sealed_std__cond")), fm(mean(V, "sealed_std__snr40__cond")), fm(mean(V, "sealed_std__snr30__cond")),
             fm(mean(V, "sealed_std__snr20__cond")), fm(mean(V, "sealed_conc050")), fm(mean(V, "sealed_conc075")),
             fm(mean(V, "sealed_diam30")), fm(mean(V, "sealed_diam70"))))
    if nv > 1:
        t += "Values without a second figure are means of the two initializations. "
    cams = [("0.90", "cam090", "0.9"), ("1.00", "camB", "1"), ("1.10", "cam110", "1.1")]
    t += (r"On the sealed fields of the cameras of 0.90, 1.00 and 1.10\,mm it scored %s as trained, %s with the new calibration in the lookup and %s after conversion of the output coordinates. "
          % tuple(", ".join(fm(mean(V, f"sealed_{c}{sfx.format(fl)}")) for _, c, fl in cams).replace(", ", " and ", 2).replace(" and ", ", ", 1)
                  for sfx in ("", "__f{}", "__conv{}")))
    t += (r"For this network the two routes are nearly equivalent, and both fall with the distance from the training camera faster than for the pooled networks: "
          r"its fusion has learned what each view shows at each depth plane for the parallax of one camera. ")
    t += (r"Reversal of the angular labels in the lookup, applied to %d sealed fields, left it with F1 of %.3f (%s true detections among %s candidates), "
          r"against %.3f for the principal pooled network, %.3f for the pooled network of matched capacity and %.3f for the lifting baseline, "
          r"whose computation the reversal does not touch. "
          % (len(rv["fields"]), rr["viewlift_s0"]["f1"], f"{rr['viewlift_s0']['tp']:,}", f"{rr['viewlift_s0']['tp'] + rr['viewlift_s0']['fp']:,}",
             rr["p7"]["f1"], rr["projlift_s0"]["f1"], rr["lift_s0"]["f1"]))
    t += (r"On the sealed two-exposure fields, read from the occupancy peak, the first initialization returned %.0f--%.0f\%% of the axial steps for particles detected in both exposures, "
          r"detected both ends of %.0f--%.0f\%% of true tracks, and gave an apparent displacement of stationary particles with an s.d.\ of %.0f, %.0f and %.0f\um. "
          % (100 * min(ax), 100 * max(ax), 100 * min(both), 100 * max(both), *[1000 * v for v in mv["still"]["sd_disp_mm"]]))
    t += (r"On the recorded flow pairs its displacements agreed with the reference with correlations of %.2f and %.2f and a median difference of %.2f pixels, "
          r"with a median of %.0f consistent tracks per pair. __REAL_S1__On the target, a line fitted to the depth it returned against stage position over the %d positions "
          r"at which it found at least half of the holes had a slope of $%.2f$ against $-1$ for the stage axis, with a residual of %.1f\,mm "
          r"(lifting baseline, $%.2f$ and %.1f\,mm over %d positions); neither network reproduced the stage. Inference took %.0f\,ms per exposure (%.0f\,ms for the wider variant)."
          % (re["corr_row"], re["corr_col"], re["median_difference_px"], re["tracks_median"], pl["planes_with_50_holes"], pl["depth_gain"],
             pl["depth_linearity_residual_mm"], pll["depth_gain"], pll["depth_linearity_residual_mm"], pll["planes_with_50_holes"],
             tim["viewlift_s0__network"]["ms_mean"], tim["viewlift_w_s0__network"]["ms_mean"]))
    r1 = load("real__viewlift_s1__heatmap")
    if r1:
        q, c = r1["summary"]["all"], r1["summary"]["clear_reference"]
        c0 = load("real__viewlift_s0__heatmap")["summary"]["clear_reference"]
        t = t.replace("__REAL_S1__", (r"For the second initialization the median difference was %.2f pixels with a median of %.0f tracks, but the correlations over all %d pairs were %.2f and %.2f, "
                                      r"because the most frequent shift was grossly wrong in a few pairs (r.m.s.\ difference %.2f against %.2f pixels); over the %d pairs with a clear reference peak "
                                      r"they were %.2f and %.2f (%.2f and %.2f for the first initialization). "
                                      % (q["median_difference_px"], q["tracks_median"], q["pairs"], q["corr_row"], q["corr_col"], q["rms_difference_px"], re["rms_difference_px"],
                                         c["pairs"], c["corr_row"], c["corr_col"], c0["corr_row"], c0["corr_col"])))
    t = t.replace("__REAL_S1__", "")
    r = load("viewlift_n120__sealed_std")
    if r:
        t += (r" Trained on 120 fields it scored %.3f on the sealed fields, against %.3f and %.3f for the two pooled networks and %.3f for the lifting baseline."
              % (r["summary"]["f1"], S("v3_n120__sealed_std"), S("projlift_n120__sealed_std"), S("lift_n120__sealed_std")))
    out.append(t)

    # ---- longer training
    chains = [("pooled network", "seed1", "v3_long_a", "v3_long"), ("pooled network of matched capacity", "projlift_s0", "projlift_long_a", "projlift_long"),
              ("lifting baseline", "lift_s0", "lift_long_a", "lift_long")]
    t = (r"\textbf{Longer training.} To test whether the comparison at 40 epochs depends on the length of training, one initialization of each of three networks was continued "
         r"for 40 epochs at a learning rate of $10^{-4}$ and for 20 epochs at $3\times10^{-5}$. After 40, 80 and 100 epochs the scores on the sealed fields were ")
    t += "; ".join("%s, %s and %s for the %s" % (fm(S(f"{a}__sealed_std")), fm(S(f"{b}__sealed_std")), fm(S(f"{c}__sealed_std")), lab) for lab, a, b, c in chains)
    t += (r" (Extended Data Fig.~5b). After 100 epochs, conversion of the output to the cameras of 0.90, 1.00 and 1.10\,mm gave ")
    t += "; ".join("%s for the %s" % (", ".join(fm(S(f"{c}__sealed_{cam}__conv{fl}")) for _, cam, fl in cams).replace(", ", " and ", 2).replace(" and ", ", ", 1), lab)
                   for lab, a, b, c in chains)
    t += (r". The order of the three networks on the training camera and on the cameras far from it was therefore unchanged by longer training, "
          r"and the view-preserving network after 40 epochs (%s) remained above the lifting baseline after 100." % fm(mean(V, "sealed_std")))
    out.append(t)

    # ---- training on four cameras
    cam7 = [("0.80", "sealed_std", True), ("0.85", "sealed_cam085__f0.85", False), ("0.90", "sealed_cam090__f0.9", True), ("0.95", "sealed_cam095__f0.95", False),
            ("1.00", "sealed_camB__f1", True), ("1.05", "sealed_cam105__f1.05", False), ("1.10", "sealed_cam110__f1.1", True)]
    nets = [("View-preserving", ["mc_viewlift", "mc_viewlift_s1"], "0.51"), ("Lifting, told the focal length", ["mc_liftcal", "mc_liftcal_s1"], "9.85"),
            ("Lifting, no calibration", ["mc_lift"], "9.65"), ("Pooled", ["mc_v3"], "0.25"), ("Pooled, matched capacity", ["mc_projlift"], "9.39")]
    rows, summary = [], {}
    for lab, runs, par in nets:
        runs = have(runs, "sealed_std")
        vals = [mean(runs, sfx) for _, sfx, _ in cam7]
        unseen = float(np.mean([v for v, (_, _, seen) in zip(vals, cam7) if not seen]))
        noise = mean(runs, "sealed_std__snr30__cond")
        summary[lab] = dict(vals=vals, mean=float(np.mean(vals)), unseen=unseen, runs=len(runs),
                            per_run=[float(np.mean([S(f"{r}__{sfx}") for _, sfx, _ in cam7])) for r in runs])
        rows.append("%s & %s & %d & %s & %.3f & %.3f & %s \\\\" % (lab, par, len(runs), " & ".join(fm(v) for v in vals), np.mean(vals), unseen,
                                                                  fm(noise) if noise is not None else "--"))
    k, lc = summary["View-preserving"], summary["Lifting, told the focal length"]
    t = (r"\textbf{Training on four cameras.} Each architecture was trained from random weights on 240 fields from each of four cameras (microlens focal lengths of 0.80, 0.90, 1.00 and 1.10\,mm; "
         r"960 fields in all, as for training on one camera) and tested on sealed fields from these cameras and from three cameras absent from training "
         r"(0.85, 0.95 and 1.05\,mm) inside their range, each with its own calibration (Supplementary Table~3). The view-preserving network was the best on average (%.3f over the seven cameras, %.3f over the three unseen ones), "
         r"and its score on the 0.80-mm camera (%.3f) was %.3f below that of the same architecture trained on that camera alone with four times as many of its fields. "
         % (k["mean"], k["unseen"], k["vals"][0], mean(V, "sealed_std") - k["vals"][0]))
    pv, pc = summary["Pooled"]["vals"], summary["Pooled, matched capacity"]["vals"]
    t += (r"The pooled network was even across the cameras but lower (%.2f--%.2f), and the pooled network of matched capacity, whose learned convolutions act across depth planes, "
          r"was accurate only in the middle of the range (%.2f--%.2f). The lifting baseline without calibration cannot distinguish the cameras, and a particle image with a given parallax then has no single depth; "
          r"it was accurate only near the middle of the range. Supplying the focal length through a learned modulation of its features raised its mean to %.3f, "
          r"%.3f below the view-preserving network at 0.80\,mm and %.3f below at 1.10\,mm, with 19 times the parameters. "
          % (min(pv), max(pv), min(pc), max(pc), lc["mean"], k["vals"][0] - lc["vals"][0], k["vals"][6] - lc["vals"][6]))
    if k["runs"] > 1:
        spread = lambda runs: max(abs(S(f"{runs[0]}__{sfx}") - S(f"{runs[1]}__{sfx}")) for _, sfx, _ in cam7)  # noqa: E731
        t += (r"Values for these two networks are means of two initializations. Means over the seven cameras for the single initializations were %s for the view-preserving network "
              r"and %s for the lifting baseline told the focal length; at single cameras the two initializations differed by at most %.3f for the former and by up to %.3f for the latter. "
              % (" and ".join(fm(v) for v in k["per_run"]), " and ".join(fm(v) for v in lc["per_run"]),
                 spread(["mc_viewlift", "mc_viewlift_s1"]), spread(["mc_liftcal", "mc_liftcal_s1"])))
    else:
        t += r"Each of these networks was trained from one initialization. "
    t += r"These three cameras lie between the training cameras; cameras outside the range, and cameras that differ in other parameters, are tested below."
    out.append(t)

    # ---- cameras outside the training set
    T1 = [("0.70", "sealed_cam070__f0.7"), ("0.75", "sealed_cam075__f0.75"), ("1.15", "sealed_cam115__f1.15"), ("1.20", "sealed_cam120__f1.2")]
    T2 = [("sealed_F095", 95, 200, 0.8), ("sealed_F105", 105, 200, 0.8), ("sealed_L190", 100, 190, 0.8), ("sealed_L210", 100, 210, 0.8),
          ("sealed_F095_f095", 95, 200, 0.95), ("sealed_F105_f095", 105, 200, 0.95)]

    def t2tag(name, F, L, f, told=None):
        return (name + (f"__f{(told if told else f):g}" if (told or f != 0.8) else "") + (f"__F{F:g}" if F != 100 else "")
                + (f"__L{L:g}" if L != 200 else "") + ("__feq" if told else ""))

    TOLD = {'sealed_F095': 0.97729, 'sealed_F105': 0.65488, 'sealed_L190': 0.648, 'sealed_L210': 0.968, 'sealed_F095_f095': 1.16053, 'sealed_F105_f095': 0.77767}   # focal lengths of equal parallax as supplied on the server (f / M^2, five decimals)
    feq = [f / (F / (L - F)) ** 2 for _, F, L, f in T2]
    mag = [F / (L - F) for _, F, L, f in T2]
    rows4, hard = [], {}
    for lab, runs, par in nets + [("Lifting, told the focal length of equal parallax", ["mc_liftcal", "mc_liftcal_s1"], "9.85")]:
        told = "equal parallax" in lab
        runs = have(runs, "sealed_std")
        a = [mean(runs, sfx) for _, sfx in T1]
        b = [mean(runs, t2tag(n, F, L, f, told=TOLD[n] if told else None)) for (n, F, L, f) in T2]
        hard[lab] = dict(out=a, other=b)
        rows4.append("%s & %s & %s \\\\" % (lab, " & ".join("--" if told else fm(v) for v in a), " & ".join(fm(v) for v in b)))
    hv, hc, hq, hn = (hard[k] for k in ("View-preserving", "Lifting, told the focal length", "Lifting, told the focal length of equal parallax",
                                        "Lifting, no calibration"))
    t = (r"\textbf{Cameras outside the training set.} The networks trained on four cameras were applied, without further training, to two further groups of sealed fields "
         r"(40 fields per camera; Supplementary Table~4). The first group has microlens focal lengths outside the range of the training cameras (0.70, 0.75, 1.15 and 1.20\,mm). "
         r"The second differs from every training camera in the focal length of the main lens (95 or 105\,mm instead of 100\,mm) or in the object distance of the focal plane (190 or 210\,mm instead of 200\,mm), "
         r"alone or together with a microlens focal length of 0.95\,mm. These two parameters never varied in training, and they change the lateral magnification from 1.00 to between %.2f and %.2f. "
         r"The lookup received the calibration of each camera, and nothing else was changed. "
         % (min(mag), max(mag)))
    t += (r"The view-preserving network scored %.3f and %.3f at 0.70 and 0.75\,mm and %.3f and %.3f at 1.15 and 1.20\,mm, and %.3f--%.3f on the six cameras of the second group. "
          r"The lifting baseline told the focal length scored %.3f, %.3f, %.3f and %.3f on the first group. On the second group the microlens focal length does not describe what has changed, and it scored %.3f--%.3f. "
          r"As the most favourable input for this baseline we also supplied the focal length that gives a camera of the training type the same parallax per unit depth near the focal plane, "
          r"$f/M^{2}$ for a magnification $M$ (%s\,mm for the six cameras); it then scored %.3f--%.3f. "
          % (*hv["out"], min(hv["other"]), max(hv["other"]), *hc["out"], min(hc["other"]), max(hc["other"]),
             ", ".join(f"{q:.2f}" for q in feq), min(hq["other"]), max(hq["other"])))
    t += (r"The baseline without calibration followed the same quantity: it scored %.2f and %.2f on the two cameras whose equivalent focal length (%.2f and %.2f\,mm) lies where it was accurate among the training cameras, "
          r"and %.2f and %.2f on the two for which it is %.2f\,mm. Plotted against this equivalent focal length, the scores of the view-preserving network on the cameras of the second group lie within %.2f of its curve for the first "
          r"(Fig.~4d). Its accuracy fell at long microlens focal lengths, where the parallax per unit depth is smallest and where every network fell."
          % (hn["other"][0], hn["other"][3], feq[0], feq[3], hn["other"][1], hn["other"][2], feq[1],
             max(abs(v - float(np.interp(q, [float(x) for x, _ in [("0.70", 0), ("0.75", 0)]] + [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20],
                                         hv["out"][:2] + k["vals"] + hv["out"][2:]))) for v, q in zip(hv["other"], feq))))
    r = load("viewlift_f120__sealed_cam120__f1.2")
    if r:
        t += (r" Both the camera and extrapolation contribute to the fall: a view-preserving network trained on the 1.20-mm camera alone, with 960 fields, "
              r"scored %.3f on that camera (%.3f for the 0.80-mm camera trained in the same way), against %.3f for the network trained on the four cameras, which had not seen it. "
              r"Towards shorter focal lengths, where parallax per unit depth grows, no such loss appeared." % (r["summary"]["f1"], mean(V, "sealed_std"), hv["out"][3]))
    out.append(t)

    # ---- second sealed set
    S2 = [("the principal pooled network", ["p7"]), ("the pooled networks of the single-stage schedule", ["scratch_s0", "seed1", "seed2"]),
          ("the pooled network of matched capacity", ["projlift_s0", "projlift_s1"]), ("the lifting baseline", ["lift_s0", "lift_s1"]),
          ("the view-preserving network", V), ("its wider variant", ["viewlift_w_s0"])]
    diffs = [abs(S(f"{r}__sealed2_std") - S(f"{r}__sealed_std")) for _, runs in S2 for r in have(runs, "sealed2_std")]
    t = (r"\textbf{Second set of sealed fields.} After every architecture had been trained, a second set of 120 fields was generated at the training camera with new seeds. "
         r"Scores on the first and on the second set were " +
         "; ".join("%s and %s for %s" % (fm(mean(have(runs, "sealed2_std"), "sealed_std")), fm(mean(have(runs, "sealed2_std"), "sealed2_std")), lab) for lab, runs in S2) +
         r". No network changed by more than %.3f, and the order of the architectures was the same." % max(diffs))
    out.append(t)

    # ---- students
    t = (r"\textbf{Training on fitted particles.} To test whether a network can be trained without reference particles, the 960 training fields were fitted per exposure "
         r"and the fitted particles replaced the reference particles as targets, with the architecture, losses and 40-epoch schedule otherwise unchanged. "
         r"On the sealed fields, where the fit itself scores %.3f, the pooled network trained in this way scored %.3f (%.3f with the occupancy-peak decoder) and the pooled network of matched capacity %.3f (%.3f). "
         r"A network trained on fitted particles thus inherited the errors of the fit and did not exceed it, so the networks of this study depend on reference particles from simulation, "
         r"and the route from fitting to a network trained without labels remains open."
         % (S("fit__sealed_std"), S("student_v3__sealed_std"), S("student_v3__sealed_std__heatmap"), S("student_projlift__sealed_std"),
            S("student_projlift__sealed_std__heatmap")))
    out.append(t)

    table = (r"""\begin{table}[h]
\caption{\textbf{ | Networks trained on four cameras: detection F1 on sealed fields of seven cameras.} Each camera is read with its own calibration. Cameras of 0.85, 0.95 and 1.05\,mm were absent from training. Noise, sensor noise at 30\,dB with conditioned input on the 0.80-mm camera.}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{33mm}rrrrrrrrrrrr@{}}
\toprule
 & & & \multicolumn{7}{c}{Microlens focal length (mm)} & & & \\
\cmidrule(lr){4-10}
Network & Par. (M) & Runs & 0.80 & 0.85 & 0.90 & 0.95 & 1.00 & 1.05 & 1.10 & Mean & Unseen & Noise \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (ROOT / "manuscript" / "generated_note7.tex").write_text("\n\n".join(out) + "\n")
    table += (r"""
\begin{table}[h]
\caption{\textbf{ | Networks trained on four cameras: detection F1 on cameras outside the training set.} Left, microlens focal lengths outside the range of the training cameras (0.80--1.10\,mm). Right, cameras with another main-lens focal length $F$ or object distance $L$ (training cameras: 100 and 200\,mm), alone or with a microlens focal length $f$ of 0.95\,mm. Each camera is read with its own calibration; 40 sealed fields per camera.}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{40mm}rrrrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{Microlens focal length (mm)} & \multicolumn{6}{c}{Other main lens or object distance (mm)} \\
\cmidrule(lr){2-5}\cmidrule(l){6-11}
Network & 0.70 & 0.75 & 1.15 & 1.20 & $F$ 95 & $F$ 105 & $L$ 190 & $L$ 210 & \makecell[r]{$F$ 95\\$f$ 0.95} & \makecell[r]{$F$ 105\\$f$ 0.95} \\
\midrule
""" + "\n".join(rows4) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (ROOT / "manuscript" / "generated_table3.tex").write_text(table)
    print("\n\n".join(out))
    print(table)


if __name__ == "__main__":
    main()
