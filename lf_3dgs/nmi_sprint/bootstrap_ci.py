"""Scene/pair-level bootstrap CIs and paired method comparisons."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(v), size=len(v))
        boots.append(v[idx].mean())
    boots = np.sort(np.asarray(boots))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"mean": float(v.mean()), "lo": lo, "hi": hi, "n": int(len(v)),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """CI for mean(a-b); positive => a better if higher-is-better metric."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b)
    d = a - b
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        boots.append(d[idx].mean())
    boots = np.sort(np.asarray(boots))
    return {
        "delta_mean": float(d.mean()),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(len(d)),
        "ci_crosses_zero": bool(np.quantile(boots, alpha / 2)
                                <= 0 <= np.quantile(boots, 1 - alpha / 2)),
    }


def summarize_eval_json(path: str, metric: str = "f1") -> Dict:
    with open(path) as f:
        data = json.load(f)
    vals = [float(s[metric]) for s in data.get("per_sample", []) if metric in s]
    ci = bootstrap_mean_ci(vals)
    return {"path": path, "metric": metric, **ci,
            "point_mean": data.get(f"mean_{metric}", ci["mean"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", action="append", required=True)
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rows = [summarize_eval_json(p, args.metric) for p in args.eval_json]
    out = {"results": rows}
    if len(rows) == 2:
        # paired if same n; load per-sample
        with open(args.eval_json[0]) as f:
            a = json.load(f)
        with open(args.eval_json[1]) as f:
            b = json.load(f)
        amap = {s["file"]: s[args.metric] for s in a["per_sample"]}
        bmap = {s["file"]: s[args.metric] for s in b["per_sample"]}
        keys = sorted(set(amap) & set(bmap))
        out["paired_delta_a_minus_b"] = paired_bootstrap_delta(
            [amap[k] for k in keys], [bmap[k] for k in keys])
    print(json.dumps(out, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
