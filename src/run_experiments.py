"""Run all experiments and write results to results/.

E1  Prior-work protocol: whole-file labels, random 70/30 window split.
E2  Sanity check: pre-fault windows only (no fault physically present), file labels.
E3  Onset-aware labels, random 70/30 split.
E4  Onset-aware labels, temporally blocked 5-fold CV with a purge gap.
E5  Onset-aware labels, cross-mode transfer (MPPT -> IPPT and IPPT -> MPPT).
E6  Window-length sensitivity for E4/E5.
E7  Model footprint (size, latency) of the deployable models.
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import T_REF, T_SKIP, baseline_reference, build_dataset, feature_cols

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
CLASSES = list(range(8))
N_FOLDS = 5
GAP = 0.2  # s, purge gap between training and test blocks of the same experiment


def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "DT": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "LGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                               random_state=SEED, verbose=-1, n_jobs=4),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=600, early_stopping=True,
                                                             random_state=SEED)),
    }


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    fault = y != 0
    return {
        "acc": accuracy_score(y, p),
        "bacc": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0),
        "recall": dict(zip(map(str, CLASSES), recall_score(y, p, labels=CLASSES, average=None, zero_division=0))),
        "det_rate": float((p[fault] != 0).mean()) if fault.any() else float("nan"),
        "far": float((p[~fault] != 0).mean()) if (~fault).any() else float("nan"),
    }


def eligible(df):
    """Drop straddling windows and the healthy reference period (used for referencing)."""
    return df[(df["label"] >= 0) & (df["t_start"] >= T_SKIP + T_REF)].reset_index(drop=True)


def blocked_folds(df):
    """Per experiment and per label side (pre/post onset), split time into N_FOLDS
    contiguous blocks. Fold k tests block k of every side of every experiment and
    purges training windows within GAP seconds of a test block."""
    block = np.zeros(len(df), int)
    for _, g in df.groupby(["exp", "label"]):
        r = g["t_start"].rank(method="first").values - 1
        block[g.index] = np.minimum((r * N_FOLDS // len(g)).astype(int), N_FOLDS - 1)
    for k in range(N_FOLDS):
        test = block == k
        train = ~test
        for exp, g in df[test].groupby("exp"):
            for _, gg in g.groupby("label"):
                a, b = gg["t_start"].min() - GAP, gg["t_end"].max() + GAP
                m = (df["exp"] == exp) & (df["t_end"] >= a) & (df["t_start"] <= b) & train
                train &= ~m.values
        yield np.where(train)[0], np.where(test)[0]


def run_random(df, label_col, model_names=("RF",), test_size=0.3):
    X, y = df[feature_cols(df)].values, df[label_col].values
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size, stratify=y, random_state=SEED)
    res = {}
    for name in model_names:
        m = models()[name].fit(X[tr], y[tr])
        res[name] = scores(y[te], m.predict(X[te]))
    return res


def run_blocked(df, model_names):
    X, y = df[feature_cols(df)].values, df["label"].values
    res = {}
    for name in model_names:
        pred = np.full(len(df), -1)
        for tr, te in blocked_folds(df):
            m = models()[name].fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        res[name] = scores(y, pred)
        res[name]["cm"] = confusion_matrix(y, pred, labels=CLASSES).tolist()
    return res


def run_cross_mode(df, model_names):
    X, y = df[feature_cols(df)].values, df["label"].values
    res = {}
    for name in model_names:
        res[name] = {}
        y_all, p_all = [], []
        for src, dst in [("M", "L"), ("L", "M")]:
            tr, te = (df["mode"] == src).values, (df["mode"] == dst).values
            m = models()[name].fit(X[tr], y[tr])
            p = m.predict(X[te])
            res[name][f"{src}->{dst}"] = scores(y[te], p)
            y_all.append(y[te]); p_all.append(p)
        y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
        res[name]["pooled"] = scores(y_all, p_all)
        res[name]["pooled"]["cm"] = confusion_matrix(y_all, p_all, labels=CLASSES).tolist()
    return res


def main():
    t0 = time.time()
    R = {}
    names = list(models())

    # ---- E1 / E2: prior-work protocol and the pre-fault sanity check (W=200, 52 stats)
    d200 = build_dataset(200, "stats")
    R["E1_file_labels_random"] = {
        "both_modes": run_random(d200, "label_file", names),
        "mppt_only": run_random(d200[d200["mode"] == "M"].reset_index(drop=True), "label_file", ["RF"]),
    }
    pre = d200[d200["t_end"] < d200["onset"]].reset_index(drop=True)
    R["E2_prefault_only_file_labels"] = run_random(pre, "label_file", names)
    # same check under the blocked split (temporal extrapolation within the healthy half)
    pre_b = pre.copy(); pre_b["label"] = pre_b["label_file"]
    R["E2b_prefault_only_blocked"] = run_blocked(pre_b, ["RF", "LGBM"])
    print("E1/E2 done", round(time.time() - t0, 1)); json.dump(R, open(OUT / "results.json", "w"), indent=1)

    # ---- E3-E5 at W=1000 for each feature set, raw vs baseline-referenced
    for fs in ["stats", "physics", "both"]:
        raw = build_dataset(1000, fs)
        ref = baseline_reference(raw)
        for tag, d in [("raw", raw), ("ref", ref)]:
            d = eligible(d)
            key = f"{fs}_{tag}"
            if fs == "both":
                R.setdefault("E3_onset_random", {})[key] = run_random(d, "label", names)
            R.setdefault("E4_onset_blocked", {})[key] = run_blocked(d, names)
            R.setdefault("E5_cross_mode", {})[key] = run_cross_mode(d, names)
            print(key, "done", round(time.time() - t0, 1))
            json.dump(R, open(OUT / "results.json", "w"), indent=1)

    # ---- E6: window-length sensitivity (physics features, baseline-referenced, LGBM)
    for w in [200, 500, 1000, 2000, 5000]:
        d = eligible(baseline_reference(build_dataset(w, "physics")))
        R.setdefault("E6_window", {})[str(w)] = {
            "blocked": run_blocked(d, ["LGBM"])["LGBM"],
            "cross": run_cross_mode(d, ["LGBM"])["LGBM"]["pooled"],
            "n_windows": int(len(d)),
        }
        print("W", w, "done", round(time.time() - t0, 1))
    json.dump(R, open(OUT / "results.json", "w"), indent=1)

    # ---- E7: footprint of models trained on physics_ref features (W=1000)
    d = eligible(baseline_reference(build_dataset(1000, "physics")))
    X, y = d[feature_cols(d)].values, d["label"].values
    fp = {}
    for name, m in models().items():
        m.fit(X, y)
        size = len(pickle.dumps(m))
        reps = 50
        t = time.perf_counter()
        for _ in range(reps):
            for i in range(0, 200):
                m.predict(X[i:i + 1])
        lat = (time.perf_counter() - t) / (reps * 200) * 1e3
        fp[name] = {"size_kB": size / 1024, "latency_ms_per_window": lat}
    # feature extraction cost per window
    from features import SIGNALS, physics_features
    win = {s: np.random.randn(1, 1000) for s in SIGNALS}
    t = time.perf_counter()
    for _ in range(2000):
        physics_features(win, 1000)
    fp["feature_extraction_ms_per_window"] = (time.perf_counter() - t) / 2000 * 1e3
    fp["n_features"] = len(feature_cols(d))
    R["E7_footprint"] = fp
    json.dump(R, open(OUT / "results.json", "w"), indent=1)
    print("all done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
