"""Gap A: what the supplier's scale (1 px = 39.209 um, dt = 10 us) implies for the 30/40/50
plenoptic recordings, and whether it is consistent with the only other scale evidence available
(the perforated-plate recording of the same camera; hole centre distance 1 mm per the supplier's README).
Run gapA_camera_geometry.py first.

Reads archived, unmodified analyses; writes source_data/gapA/scale_check.json.
No new image processing of the flow recordings is done here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
PROJ = Path.home() / "Documents/ChatGPT/lf-piv/real_data_20261108/analysis"
PLATE = Path.home() / "Desktop/光场/标定板解码sai及真值npz/80010271289_20260614_191203_0_sai.npz"
SCALE_M, DT_S, SENSOR_PX_UM, MLA_PITCH_UM = 3.9209e-5, 1e-5, 4.5, 100.0


def plate_scale():
    g = json.loads((HERE / "source_data/gapA/camera_geometry.json").read_text())["plate_june"]
    return dict(hole_pitch_mm=g["hole_pitch_mm"], hole_pitch_source=g["hole_pitch_source"],
                hole_pitch_sai_px=g["hole_pitch_px_at_z0"], um_per_sai_px=g["um_per_sai_px"],
                um_per_sai_px_range_within_2p5mm=g["um_per_sai_px_range_within_2p5mm"])


def main():
    pre = json.loads((PROJ / "lightfield_preflight/summary.json").read_text())
    planar = json.loads((PROJ / "planar_piv/summary.json").read_text())
    plate = plate_scale()
    out = {"supplier": dict(metres_per_px=SCALE_M, dt_s=DT_S, mps_per_px_per_frame=SCALE_M / DT_S,
                            pairing="odd frame with the following even frame (21-22, 23-24, ...)"),
           "plate": plate, "groups": {}}
    readings = {
        # px = native sensor pixel of the plenoptic camera
        "supplier_px_is_sensor_px": lambda pitch_px: pitch_px * SCALE_M,
        # px = pixel of the decoded sub-aperture image
        "supplier_px_is_sai_px": lambda pitch_px: SCALE_M,
        # object-space scale of the same camera measured on the plate (June session)
        "plate_scale": lambda pitch_px: plate["um_per_sai_px"] * 1e-6,
    }
    for g, rec in pre["groups"].items():
        pitch_px = 0.5 * (rec["lattice"]["pitch_x_sensor_px"] + rec["lattice"]["pitch_y_sensor_px"])
        cen = [r["view_summaries"][0] for r in rec["records"]]
        ok = [c for c in cen if c["valid_fraction"] >= 0.5]
        sp = np.array([c["speed_sai_px_median"] for c in cen])
        sp_ok = np.array([c["speed_sai_px_median"] for c in ok])
        u = np.array([c["u_sai_px_median"] for c in ok])
        v = np.array([c["v_sai_px_median"] for c in ok])
        row = dict(pairs=len(cen), pairs_valid_fraction_ge_0p5=len(ok), microlens_pitch_sensor_px=pitch_px,
                   speed_sai_px_median_all_pairs=float(np.median(sp)),
                   speed_sai_px_median=float(np.median(sp_ok)),
                   speed_sai_px_iqr=[float(x) for x in np.percentile(sp_ok, [25, 75])],
                   u_sai_px_median=float(np.median(u)), v_sai_px_median=float(np.median(v)), readings={})
        for name, fn in readings.items():
            m_per_sai = fn(pitch_px)
            row["readings"][name] = dict(um_per_sai_px=1e6 * m_per_sai, magnification=MLA_PITCH_UM / (1e6 * m_per_sai),
                                         field_of_view_mm=[1e3 * m_per_sai * 230, 1e3 * m_per_sai * 184],
                                         speed_mps_median=float(np.median(sp_ok) * m_per_sai / DT_S),
                                         speed_mps_iqr=[float(x * m_per_sai / DT_S) for x in np.percentile(sp_ok, [25, 75])])
        out["groups"][g] = row
    sp = np.array([r["speed_mps_median"] for r in planar["records"]])
    out["planar_camera"] = dict(serial="80010271290", pairs=len(sp), speed_mps_median_of_pairs=float(np.median(sp)),
                                speed_mps_range=[float(sp.min()), float(sp.max())],
                                speed_px_median_of_pairs=float(np.median([r["speed_px_median"] for r in planar["records"]])),
                                magnification_if_supplier_scale=SENSOR_PX_UM * 1e-6 / SCALE_M)
    out["notes"] = [
        "Author's statements (21 Nov 2026): the November flow recordings kept the June configuration of the plenoptic "
        "camera; the pulse separation is 10 us; the plate is the one documented by the supplier (hole centre distance 1 mm).",
        "With these, the supplier's 39.209 um cannot be the sub-aperture pixel of this camera (80.9 um measured on the plate; "
        "the depth response of the plate is one third of what a 39.2-um pixel would give). It equals a 200.7-mm field over "
        "5,120 sensor pixels and matches the conventional camera (serial ...290, frames 21-50) to which it was first applied.",
        "No reading provides an independent velocity reference: all speeds come from cross-correlation of the same images.",
    ]
    dest = HERE / "source_data/gapA"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scale_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({g: {k: (round(v["speed_mps_median"], 1), round(v["um_per_sai_px"], 1), round(v["magnification"], 3))
                          for k, v in r["readings"].items()} | {"sai_px": round(r["speed_sai_px_median"], 3),
                                                                "n": (r["pairs_valid_fraction_ge_0p5"], r["pairs"])}
                      for g, r in out["groups"].items()}, indent=1))
    print(out["plate"], out["planar_camera"])


if __name__ == "__main__":
    main()
