"""Write the Extended Data table bodies as LaTeX from the archived summaries."""
from __future__ import annotations

from pathlib import Path

from lfstyle import load_json

OUT = Path(__file__).resolve().parent.parent / "manuscript" / "tables"
OUT.mkdir(exist_ok=True)


def fmt(n):
    return f"{int(n):,}".replace(",", "{,}")


def ed_table1():
    S = load_json("fig4/calibration_location_summary.json")
    E = load_json("fig4/calibration_location_equal_fp.json")
    names = {"p7": "Principal", "off": "No rendering loss", "on": "With rendering loss"}
    arms = {"old": "Previous calibration", "output": "New, applied to output", "lookup": "New, applied to lookup"}
    lines, contrasts = [], []
    for ck in ("p7", "off", "on"):
        t, m = S[ck]["totals"], S[ck]["field_mean_and_paired_ci"]
        for i, arm in enumerate(("old", "output", "lookup")):
            c3 = 1000 * m[f"common3_{arm}_mae_mm"]["mean"]
            c2 = "--" if arm == "old" else f"{1000 * m[f'common2_{arm}_mae_mm']['mean']:.1f}"
            first = (r"\multirow{3}{*}{\shortstack[l]{" + names[ck] + r"\\(" + fmt(t["common3_n"]) + "; "
                     + fmt(t["common2_n"]) + ")}}") if i == 0 else ""
            lines.append(" & ".join([
                first, arms[arm], fmt(t[f"{arm}_tp"]), fmt(t[f"{arm}_fp"]),
                f"{100 * m[f'{arm}_recall']['mean']:.1f}", f"{100 * m[f'{arm}_precision']['mean']:.1f}",
                f"{m[f'{arm}_f1']['mean']:.3f}", f"{c3:.1f}", c2]) + r" \\")
        g3, g2 = m["common3_gain_mm"], m["common2_gain_mm"]
        eq = E[ck]["field_means"]["recall_delta_pp"]
        contrasts.append(" & ".join([
            names[ck],
            f"${-1000 * g3['mean']:+.1f}$ (${-1000 * g3['ci95'][1]:.1f}$ to ${-1000 * g3['ci95'][0]:.1f}$)",
            f"${-1000 * g2['mean']:+.1f}$ (${-1000 * g2['ci95'][1]:.1f}$ to ${-1000 * g2['ci95'][0]:.1f}$)",
            f"${m['lookup_minus_output_recall_pp']['mean']:+.1f}$",
            f"${eq['mean']:+.1f}$ (${eq['ci95'][0]:.1f}$ to ${eq['ci95'][1]:.1f}$)"]) + r" \\")
        if ck != "on":
            lines.append(r"\addlinespace[3pt]")
    (OUT / "ed_table1b_body.tex").write_text("\n".join(contrasts) + "\n\\bottomrule\n")
    (OUT / "ed_table1_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")


def ed_table2():
    rows = [("Parent checkpoint", "ablation_p6_parent"),
            ("Continuation without hard-negative loss", "ablation_no_hard_negatives"),
            ("Continuation with depth-independent skip", "ablation_broadcast_skip"),
            ("Principal network (hard negatives, depth-aware skip)", "p7_test_hungarian_nms")]
    lines = []
    for name, key in rows:
        a = load_json(f"extended/{key}.json")
        lines.append(" & ".join([name, f"{a['mean_f1']:.3f}", f"{100 * a['mean_precision']:.1f}",
                                 f"{100 * a['mean_recall']:.1f}", f"{1000 * a['sigma_x']:.0f}",
                                 f"{1000 * a['sigma_y']:.0f}", f"{1000 * a['sigma_z']:.0f}"]) + r" \\")
    (OUT / "ed_table2_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")


if __name__ == "__main__":
    ed_table1()
    ed_table2()
    print((OUT / "ed_table1_body.tex").read_text())
    print((OUT / "ed_table2_body.tex").read_text())
