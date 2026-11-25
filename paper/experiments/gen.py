"""Generate sealed evaluation data with the project ray tracer (unchanged).

  python gen.py check                                   # reproduce an archived dataset9 field bit-for-bit
  python gen.py static --out sealed_std --n 120 --seed0 91000000
  python gen.py static --out sealed_conc050 --n 40 --seed0 91100000 --conc 0.05
  python gen.py motion --out sealed_motion --n 24 --seed0 92000000
  python gen.py archive --manifest manifests/lightfield_dataset9_seeds.json --out lightfield_dataset9
  python gen.py verify --out sealed_std                 # compare images and coordinates with the recorded digests

Seeds are disjoint from every archived dataset (dataset9 uses 30000000 + index). Files are written
once and never overwritten; a manifest with SHA-256 of every file is written at the end.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("LFPIV_PROJECT", Path(__file__).resolve().parents[2]))    # checkout of lf-3dgs-project
DATAGEN = Path(os.environ.get("LFPIV_DATAGEN", PROJECT.parent / "lf-3dgs-datagen"))     # checkout of lf-3dgs-datagen
GAP = Path(os.environ.get("LFPIV_WORK", PROJECT / "work"))
DS9 = Path(os.environ.get("LFPIV_TRAIN_DATA", GAP / "data" / "lightfield_dataset9"))
sys.path.insert(0, str(DATAGEN))

CAMERA = dict(main_lens_focal_length_mm=100.0, main_lens_aperture_diameter_mm=25.0, object_distance_mm=200.0,
              mla_focal_length_mm=0.8, num_mla_x=232, num_mla_y=186, microlens_aperture_mm=0.099,
              ccd_resolution_x=5120, ccd_resolution_y=4096, ccd_pixel_pitch_mm=0.0045)
XR, YR, ZR = (-10, 10), (-8.0, 8.0), (-5.0, 5.0)
GRID = (64, 128, 128)
MOTIONS = {"still": (0, 0, 0), "x_p020": (0.2, 0, 0),
           "z_p005": (0, 0, 0.05), "z_m005": (0, 0, -0.05), "z_p010": (0, 0, 0.10), "z_m010": (0, 0, -0.10),
           "z_p020": (0, 0, 0.20), "z_m020": (0, 0, -0.20), "z_p040": (0, 0, 0.40), "z_m040": (0, 0, -0.40)}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def content_digest(path):
    """SHA-256 of the images (as 8-bit) and reference coordinates of one file, independent of the container."""
    h = hashlib.sha256()
    with np.load(path) as z:
        for k in ("sai_stack", "sai"):
            if k in z.files:
                h.update(np.ascontiguousarray(z[k]).astype(np.uint8).tobytes())
        for k in ("particle_centers", "holes_mm"):
            if k in z.files:
                h.update(np.ascontiguousarray(z[k]).astype(np.float32).tobytes())
    return h.hexdigest()


def digest_job(args):
    root, rel = args
    return rel, content_digest(Path(root) / rel)


def verify(name, write, workers):
    """Compare the content of data/<name> with manifests/content/<name>.json, or write that file (--write)."""
    root = (DS9 if name == "lightfield_dataset9" else GAP / "data" / name)
    rels = sorted(str(p.relative_to(root)) for p in root.rglob("*.npz"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        got = dict(ex.map(digest_job, [(str(root), r) for r in rels], chunksize=8))
    ref = Path(__file__).with_name("manifests") / "content" / (name.replace("/", "__") + ".json")
    if write:
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(json.dumps(got, indent=0, sort_keys=True))
        print("wrote", ref, len(got))
        return
    want = json.loads(ref.read_text())
    bad = sorted(k for k in want if got.get(k) != want[k])
    print(json.dumps(dict(set=name, files_expected=len(want), files_found=len(got), identical=len(want) - len(bad), differing=bad[:10])))


def trace(points, cam, gsa):
    with contextlib.redirect_stdout(io.StringIO()):
        raw = cam.generate_light_field_image(points, num_rays_per_particle=1000)
        sai = gsa(raw, cam)
    return sai.reshape(cam.pixels_per_microlens_y * cam.pixels_per_microlens_x, cam.num_mla_y, cam.num_mla_x)


def base_field(seed, conc, diam, f_mla, F=100.0, L=200.0):
    from generate_particle import generate_particle_point_clouds
    from simulate_lightfield import LightFieldCamera, generate_sai_array
    np.random.seed(seed % 2147483648)
    with contextlib.redirect_stdout(io.StringIO()):
        cam = LightFieldCamera(**dict(CAMERA, mla_focal_length_mm=f_mla, main_lens_focal_length_mm=F, object_distance_mm=L))
        cloud, centres = generate_particle_point_clouds(
            ppm_concentration=conc, num_microlenses_total=186 * 232, x_spatial_range=XR, y_spatial_range=YR,
            z_spatial_range=ZR, particle_diameter_um=diam, points_per_particle=50)
    return cam, cloud, centres, generate_sai_array


def one_static(args):
    idx, seed, conc, diam, f_mla, out, F, L = args
    path = Path(out) / (idx if isinstance(idx, str) else f"field_{idx:04d}.npz")
    if path.exists():
        return path.name, sha_file(path)
    cam, cloud, centres, gsa = base_field(seed, conc, diam, f_mla, F, L)
    sai = trace(cloud, cam, gsa)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, sai_stack=sai.astype(np.uint8), particle_centers=centres.astype(np.float32),
                        particle_diameter_um=np.float32(diam), ppm_concentration=np.float32(conc),
                        mla_focal_length_mm=np.float32(f_mla), rng_seed=np.int64(seed),
                        main_lens_focal_length_mm=np.float32(F), object_distance_mm=np.float32(L),
                        x_range=np.array(XR, np.float32), y_range=np.array(YR, np.float32),
                        z_range=np.array(ZR, np.float32), grid_shape=np.array(GRID, np.int32))
    tmp.rename(path)
    return path.name, sha_file(path)


def one_motion(args):
    idx, seed, f_mla, out = args
    folder = Path(out) / f"field_{idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "done.json"
    if done.exists():
        return folder.name, json.loads(done.read_text())
    cam, cloud, centres, gsa = base_field(seed, 0.025, 50.0, f_mla)
    lim = np.array([XR, YR, ZR], float)
    rec = {}
    first = trace(cloud, cam, gsa)
    np.savez_compressed(folder / "frame0.npz", sai_stack=first.astype(np.uint8),
                        particle_centers=centres.astype(np.float32), rng_seed=np.int64(seed))
    rec["frame0"] = sha_file(folder / "frame0.npz")
    for c, (name, delta) in enumerate(MOTIONS.items()):
        d = np.array(delta, float)
        moved = centres + d
        keep = ((moved >= lim[:, 0]) & (moved <= lim[:, 1])).all(1)
        pts = cloud.reshape(len(centres), 50, 3)[keep].reshape(-1, 3) + d
        np.random.seed((seed + 7919 * (c + 1)) % 2147483648)      # independent ray sample per exposure
        img = trace(pts, cam, gsa)
        np.savez_compressed(folder / f"{name}.npz", sai_stack=img.astype(np.uint8),
                            particle_centers=moved[keep].astype(np.float32), ids=np.flatnonzero(keep),
                            displacement_mm=d.astype(np.float32))
        rec[name] = sha_file(folder / f"{name}.npz")
    done.write_text(json.dumps(rec))
    return folder.name, rec


def one_plate(args):
    """Simulated perforated plate: 10 x 10 uniformly lit discs at one depth (same ray tracer and camera)."""
    z, seed, diam_mm, pitch_mm, out = args
    path = Path(out) / f"plate_{z:+05.1f}.npz".replace("+", "p").replace("-", "m")
    if path.exists():
        return path.name, sha_file(path)
    from simulate_lightfield import LightFieldCamera, generate_sai_array
    np.random.seed(seed % 2147483648)
    with contextlib.redirect_stdout(io.StringIO()):
        cam = LightFieldCamera(**dict(CAMERA, mla_focal_length_mm=0.8))
    g = (np.arange(10) - 4.5) * pitch_mm
    centres = np.array([(x, y, z) for y in g for x in g], float)
    n_pts = 200
    r = 0.5 * diam_mm * np.sqrt(np.random.rand(len(centres), n_pts))
    th = 2 * np.pi * np.random.rand(len(centres), n_pts)
    pts = centres[:, None, :] + np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(r)], -1)
    sai = trace(pts.reshape(-1, 3), cam, generate_sai_array)
    np.savez_compressed(path, sai=sai.astype(np.uint8), view_idx=np.arange(sai.shape[0]), holes_mm=centres.astype(np.float32),
                        stage_mm=np.float32(z), diam_mm=np.float32(diam_mm), pitch_mm=np.float32(pitch_mm))
    return path.name, sha_file(path)


def run(jobs, fn, out, workers, meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0, manifest = time.time(), {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(fn, j) for j in jobs])):
            name, h = fut.result()
            manifest[name] = h
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(meta, files=dict(sorted(manifest.items())),
                                                        seconds=time.time() - t0), indent=1))
    print("done", out, len(manifest), f"{time.time() - t0:.0f}s")


def check():
    """Regenerate dataset9 test sample_00000 (seed 30000000) and compare with the archive."""
    ref = np.load(DS9 / "test/sample_00000_diam50um_conc0p025.npz")
    cam, cloud, centres, gsa = base_field(int(ref["rng_seed"]), 0.025, 50.0, 0.8)
    sai = trace(cloud, cam, gsa)
    same_c = np.array_equal(centres.astype(np.float32), ref["particle_centers"])
    same_i = np.array_equal(sai.astype(np.float32), ref["sai_stack"].astype(np.float32))
    print(json.dumps(dict(seed=int(ref["rng_seed"]), centres_identical=bool(same_c), images_identical=bool(same_i),
                          differing_pixels=int((sai.astype(np.float32) != ref["sai_stack"]).sum()))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "static", "motion", "plate", "archive", "verify"])
    ap.add_argument("--write", action="store_true", help="verify: write the content digests instead of comparing")
    ap.add_argument("--manifest", help="archive: JSON with splits -> [[file name, seed, ...], ...]")
    ap.add_argument("--diam-mm", type=float, default=0.245, help="plate: disc diameter")
    ap.add_argument("--pitch-mm", type=float, default=1.224, help="plate: disc spacing")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=91000000)
    ap.add_argument("--conc", type=float, default=0.025)
    ap.add_argument("--diam", type=float, default=50.0)
    ap.add_argument("--f", type=float, default=0.8)
    ap.add_argument("--F", type=float, default=100.0, help="main-lens focal length (mm)")
    ap.add_argument("--L", type=float, default=200.0, help="object distance of the focal plane (mm)")
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    if a.mode == "check":
        check()
    elif a.mode == "verify":
        verify(a.out, a.write, a.workers)
    elif a.mode == "static":
        out = GAP / "data" / a.out
        run([(i, a.seed0 + i, a.conc, a.diam, a.f, str(out), a.F, a.L) for i in range(a.n)], one_static, out, a.workers,
            dict(kind="static", n=a.n, seed0=a.seed0, conc=a.conc, diam_um=a.diam, f_mla=a.f, main_lens_focal_length_mm=a.F,
                 object_distance_mm=a.L))
    elif a.mode == "archive":      # rebuild the training archive from the seed of every field (same images and centres)
        for split, rows in json.loads(Path(a.manifest).read_text())["splits"].items():
            out = GAP / "data" / a.out / split
            run([(r[0], int(r[1]), 0.025, 50.0, 0.8, str(out), 100.0, 200.0) for r in rows], one_static, out, a.workers,
                dict(kind="archive", split=split, n=len(rows)))
    elif a.mode == "plate":
        out = GAP / "data" / a.out
        zs = [round(-4.5 + 0.5 * i, 1) for i in range(19)]
        run([(z, a.seed0 + i, a.diam_mm, a.pitch_mm, str(out)) for i, z in enumerate(zs)], one_plate, out, a.workers,
            dict(kind="plate", seed0=a.seed0, diam_mm=a.diam_mm, pitch_mm=a.pitch_mm, depths=zs))
    else:
        out = GAP / "data" / a.out
        run([(i, a.seed0 + i, a.f, str(out)) for i in range(a.n)], one_motion, out, a.workers,
            dict(kind="motion", n=a.n, seed0=a.seed0, f_mla=a.f, motions=MOTIONS))
