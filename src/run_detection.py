"""E8: training-free fault detection on baseline-referenced features.

The anomaly score of a window is the largest absolute deviation of any
baseline-referenced physics feature from the installation's own healthy
reference period:  s = max_j |z_j|.  A single threshold is calibrated on the
reference windows of the eight MPPT experiments only and then applied to all
16 experiments, so the IPPT mode is never seen during calibration.
An alarm is raised after K consecutive windows exceed the threshold.
"""
import json
from pathlib import Path

import numpy as np

from features import T_REF, T_SKIP, baseline_reference, build_dataset, feature_cols

OUT = Path(__file__).resolve().parents[1] / "results"
K = 3  # consecutive windows required for an alarm


def run(w=1000, feature_set="physics", q=0.99):
    d = baseline_reference(build_dataset(w, feature_set))
    cols = feature_cols(d)
    d["score"] = d[cols].abs().max(axis=1)
    is_ref = d["t_end"] < T_SKIP + T_REF
    thr = float(d.loc[is_ref & (d["mode"] == "M"), "score"].quantile(q))
    d["exceed"] = d["score"] > thr
    # alarm = K consecutive exceedances within an experiment
    d["alarm"] = False
    for _, g in d.groupby("exp"):
        run_len = g["exceed"].astype(int).groupby((~g["exceed"]).cumsum()).cumsum()
        d.loc[g.index, "alarm"] = run_len.values >= K
    ev = d[(~is_ref) & (d["label"] >= 0)]
    normal = ev["label"] == 0
    res = {"threshold": thr, "quantile": q, "window": w, "K": K,
           "far_window": float(ev.loc[normal, "alarm"].mean()),
           "per_experiment": {}}
    for exp, g in d.groupby("exp"):
        onset = g["onset"].iloc[0]
        post = g[g["t_start"] >= onset]
        pre = g[(g["t_end"] < onset) & (g["t_end"] >= T_SKIP + T_REF)]
        first = post.loc[post["alarm"], "t_end"]
        res["per_experiment"][exp] = {
            "false_alarm_pre": float(pre["alarm"].mean()) if exp[1] != "0" else float(g.loc[~is_ref[g.index], "alarm"].mean()),
            "detection_rate_post": float(post["alarm"].mean()) if exp[1] != "0" else None,
            "delay_s": float(first.iloc[0] - onset) if (exp[1] != "0" and len(first)) else None,
        }
    fault = (ev["label"] > 0)
    res["det_rate_window"] = float(ev.loc[fault, "alarm"].mean())
    return res, d


def main():
    R = {}
    for q in [0.95, 0.99, 1.0]:
        R[f"q{q}"] = run(q=q)[0]
    json.dump(R, open(OUT / "detection.json", "w"), indent=1)
    for k, r in R.items():
        print(k, "thr %.2f FAR %.3f DR %.3f" % (r["threshold"], r["far_window"], r["det_rate_window"]))
        for e, v in r["per_experiment"].items():
            print("   ", e, v)


if __name__ == "__main__":
    main()
