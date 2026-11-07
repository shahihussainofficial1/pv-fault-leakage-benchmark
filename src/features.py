"""Windowing, labelling and feature extraction for GPVS-Faults."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SIGNALS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
FS = 10_000.0          # Hz, median sampling interval is 100 us
T_SKIP = 0.3           # s, start-up transient of the PLL / magnitude estimators
T_REF = 2.0            # s, healthy reference period used for baseline referencing
FAULT_NAMES = {
    0: "Normal", 1: "Inverter (IGBT)", 2: "Feedback sensor", 3: "Grid sag",
    4: "Partial shading", 5: "PV open circuit", 6: "MPPT/IPPT ctrl.", 7: "Boost ctrl.",
}


def experiments():
    """Yield (name, fault_id, mode, DataFrame) for the 16 experiments."""
    for p in sorted(RAW_DIR.glob("F*.csv")):
        name = p.stem
        yield name, int(name[1]), name[2], pd.read_csv(p)


def _windows(x, w):
    n = len(x) // w
    return x[: n * w].reshape(n, w)


def _thd(x, w):
    """Total harmonic distortion (h = 2..13) of each window row."""
    spec = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1))
    res = FS / w
    k1 = int(round(50 / res))
    if w >= 500:  # enough resolution to track +-1 bin of frequency deviation
        lo, hi = max(k1 - 1, 1), k1 + 2
        k_rel = np.argmax(spec[:, lo:hi], axis=1) + lo
    else:
        k_rel = np.full(len(x), k1)
    fund = spec[np.arange(len(x)), k_rel]
    harm = np.zeros(len(x))
    for h in range(2, 14):
        idx = np.clip(k_rel * h, 0, spec.shape[1] - 1)
        harm += spec[np.arange(len(x)), idx] ** 2
    return np.sqrt(harm) / (fund + 1e-9)


def stat_features(win):
    """Statistical signatures (mean, std, min, max) of every signal: 52 features."""
    out = {}
    for s in SIGNALS:
        x = win[s]
        out[f"{s}_mean"] = x.mean(1)
        out[f"{s}_std"] = x.std(1)
        out[f"{s}_min"] = x.min(1)
        out[f"{s}_max"] = x.max(1)
    return out


def physics_features(win, w):
    """Compact, physically interpretable features of the PV-converter-grid chain."""
    ia, ib, ic, va, vb, vc = (win[k] for k in ["ia", "ib", "ic", "va", "vb", "vc"])
    rms = lambda x: np.sqrt((x ** 2).mean(1))
    i_rms = np.stack([rms(ia), rms(ib), rms(ic)], 1)
    v_rms = np.stack([rms(va), rms(vb), rms(vc)], 1)
    p_pv = (win["Ipv"] * win["Vpv"]).mean(1)
    p_ac = (va * ia + vb * ib + vc * ic).mean(1)
    return {
        "P_pv": p_pv,
        "P_ac": p_ac,
        "eta_ac_dc": p_ac / (np.abs(p_pv) + 1e-3),
        "Vpv_mean": win["Vpv"].mean(1),
        "Ipv_mean": win["Ipv"].mean(1),
        "Vdc_mean": win["Vdc"].mean(1),
        "Vdc_ripple": win["Vdc"].std(1),
        "Ipv_ripple": win["Ipv"].std(1),
        "I_rms": i_rms.mean(1),
        "V_rms": v_rms.mean(1),
        "I_unbalance": (i_rms.max(1) - i_rms.min(1)) / (i_rms.mean(1) + 1e-6),
        "V_unbalance": (v_rms.max(1) - v_rms.min(1)) / (v_rms.mean(1) + 1e-6),
        "I_zero_seq": np.abs((ia + ib + ic).mean(1)),
        "I_dc_offset": np.abs(np.stack([ia.mean(1), ib.mean(1), ic.mean(1)], 1)).max(1),
        "THD_i": np.mean([_thd(ia, w), _thd(ib, w), _thd(ic, w)], 0),
        "THD_v": np.mean([_thd(va, w), _thd(vb, w), _thd(vc, w)], 0),
        "Iabc_mean": win["Iabc"].mean(1),
        "Vabc_mean": win["Vabc"].mean(1),
        "f_dev": np.abs(win["Vf"] - 50.0).mean(1),
    }


def build_dataset(w=1000, feature_set="stats"):
    """Return a DataFrame with one row per non-overlapping window.

    Columns: features + meta (exp, fault, mode, t_start, t_end, onset, label_file, label).
    label_file: fault id of the whole experiment (label used in prior work).
    label: onset-aware label (0 before the documented mid-experiment fault
           injection, fault id after it; windows straddling the onset are dropped).
    """
    rows = []
    for name, fault, mode, df in experiments():
        t = df["Time"].values
        win = {s: _windows(df[s].values.astype(float), w) for s in SIGNALS}
        tw = _windows(t, w)
        feats = {}
        if feature_set in ("stats", "both"):
            feats.update(stat_features(win))
        if feature_set in ("physics", "both"):
            feats.update(physics_features(win, w))
        f = pd.DataFrame(feats)
        f["exp"], f["fault"], f["mode"] = name, fault, mode
        f["t_start"], f["t_end"] = tw[:, 0], tw[:, -1]
        onset = t[len(t) // 2]
        f["onset"] = onset
        f["label_file"] = fault
        lab = np.where(f["t_end"] < onset, 0, np.where(f["t_start"] >= onset, fault, -1))
        if fault == 0:
            lab[:] = 0
        f["label"] = lab
        f = f[f["t_start"] >= T_SKIP].reset_index(drop=True)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


META = ["exp", "fault", "mode", "t_start", "t_end", "onset", "label_file", "label"]


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def baseline_reference(df):
    """Express every feature as a deviation from the experiment's own healthy
    reference period (first T_REF s after start-up). This mimics a commissioning
    baseline recorded per installation and operating mode."""
    cols = feature_cols(df)
    out = df.copy()
    for name, g in df.groupby("exp"):
        ref = g[g["t_end"] < T_SKIP + T_REF][cols]
        mu, sd = ref.mean(), ref.std()
        sd = np.maximum(sd, np.maximum(1e-3 * mu.abs(), 1e-6))
        out.loc[g.index, cols] = ((g[cols] - mu) / sd).clip(-50, 50).values
    return out
