"""Generate paper figures and LaTeX tables from results/results.json."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import shap

from features import FAULT_NAMES, baseline_reference, build_dataset, experiments, feature_cols
from run_experiments import eligible, models

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "results" / "results.json"))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.bbox": "tight"})
COL = 3.45  # IEEE column width in inches


def fig_signals():
    data = {n: d for n, f, m, d in experiments() if n in ("F5M", "F2M", "F7M")}
    fig, axes = plt.subplots(3, 1, figsize=(COL, 3.3), sharex=True)
    for ax, (n, sig, ttl) in zip(axes, [("F5M", "Ipv", "F5M (PV open circuit): $I_{pv}$ [A]"),
                                        ("F2M", "Vdc", "F2M (feedback sensor): $V_{dc}$ [V]"),
                                        ("F7M", "Ipv", "F7M (boost controller): $I_{pv}$ [A]")]):
        d = data[n]
        ax.plot(d["Time"][::5], d[sig][::5], lw=0.3, color="tab:blue")
        ax.plot(d["Time"], d[sig].rolling(1000, center=True).mean(), lw=0.9, color="k")
        ax.axvline(d["Time"].iloc[len(d) // 2], color="tab:red", ls="--", lw=0.9)
        ax.set_title(ttl, fontsize=7.5, pad=2)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(FIG / "signals.pdf")


def fig_protocols():
    rows = [
        ("P1: file labels,\nrandom split", R["E1_file_labels_random"]["both_modes"]["RF"]["macro_f1"]),
        ("P2: pre-fault only,\nfile labels", R["E2_prefault_only_file_labels"]["RF"]["macro_f1"]),
        ("P3: onset labels,\nrandom split", R["E3_onset_random"]["both_raw"]["RF"]["macro_f1"]),
        ("P4: onset labels,\nblocked CV", R["E4_onset_blocked"]["stats_raw"]["RF"]["macro_f1"]),
        ("P5: onset labels,\ncross-mode", R["E5_cross_mode"]["stats_raw"]["RF"]["pooled"]["macro_f1"]),
        ("P5 + baseline\nreferencing", R["E5_cross_mode"]["stats_ref"]["RF"]["pooled"]["macro_f1"]),
    ]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.bar(x, vals, color=["#9e9e9e", "#d62728", "#9e9e9e", "#1f77b4", "#1f77b4", "#2ca02c"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["P1", "P2", "P3", "P4", "P5", "P5+ref."], fontsize=7)
    ax.set_ylabel("Macro-F1 (RF)")
    ax.set_ylim(0, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "protocols.pdf")


def fig_confusion(key="stats_ref", model="RF"):
    cm = np.array(R["E5_cross_mode"][key][model]["pooled"]["cm"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.6 else "black")
    lab = [f"F{i}" if i else "N" for i in range(8)]
    ax.set_xticks(range(8), lab); ax.set_yticks(range(8), lab)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_crossmode.pdf")


def fig_window():
    E6 = R["E6_window"]
    w = sorted(map(int, E6))
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    ax.plot([x / 10 for x in w], [E6[str(x)]["blocked"]["macro_f1"] for x in w], "o-", label="Blocked CV")
    ax.plot([x / 10 for x in w], [E6[str(x)]["cross"]["macro_f1"] for x in w], "s-", label="Cross-mode")
    ax.set_xscale("log"); ax.set_xlabel("Window length [ms]"); ax.set_ylabel("Macro-F1")
    ax.set_xticks([x / 10 for x in w], [f"{x / 10:g}" for x in w])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "window.pdf")


def fig_shap():
    d = eligible(baseline_reference(build_dataset(1000, "physics")))
    cols = feature_cols(d)
    m = models()["LGBM"].fit(d[cols].values, d["label"].values)
    sv = shap.TreeExplainer(m).shap_values(d[cols].values)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv  # (n, f, classes)
    imp = np.abs(sv).mean(0)  # (f, classes)
    order = np.argsort(imp.sum(1))[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    left = np.zeros(len(order))
    cmap = plt.get_cmap("tab10")
    for c in range(imp.shape[1]):
        ax.barh(range(len(order)), imp[order, c], left=left, color=cmap(c),
                label=FAULT_NAMES[c] if c else "Normal", height=0.7)
        left += imp[order, c]
    names = [cols[i].replace("_", r"\_") for i in order]
    ax.set_yticks(range(len(order)), [f"${n}$" if False else cols[i] for n, i in zip(names, order)], fontsize=6.5)
    ax.set_xlabel("Mean |SHAP value| (stacked by class)")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "shap.pdf")
    top = pd.DataFrame(imp, index=cols, columns=[FAULT_NAMES[c] for c in range(8)])
    top.to_csv(ROOT / "results" / "shap_importance.csv")


def tables():
    # Table: blocked CV and cross-mode, all models x feature configs
    lines = []
    for key, lab in [("stats_raw", "Stats, raw"), ("stats_ref", "Stats, ref."), ("physics_raw", "Physics, raw"),
                     ("physics_ref", "Physics, ref."), ("both_raw", "Both, raw"), ("both_ref", "Both, ref.")]:
        cells = []
        for m in ["LR", "DT", "RF", "LGBM", "MLP"]:
            b = R["E4_onset_blocked"][key][m]["macro_f1"]
            c = R["E5_cross_mode"][key][m]["pooled"]["macro_f1"]
            cells.append(f"{b:.2f} / {c:.2f}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (TAB / "main.tex").write_text("\n".join(lines) + "\n")

    # Table: per-class recall for best deployable config
    lines = []
    for c in range(8):
        r1 = R["E1_file_labels_random"]["both_modes"]["RF"]["recall"][str(c)]
        r4 = R["E4_onset_blocked"]["stats_raw"]["RF"]["recall"][str(c)]
        r5 = R["E5_cross_mode"]["stats_ref"]["RF"]["pooled"]["recall"][str(c)]
        r5r = R["E5_cross_mode"]["stats_raw"]["RF"]["pooled"]["recall"][str(c)]
        name = FAULT_NAMES[c]
        lines.append(f"{'F' + str(c) if c else 'N'} & {name} & {r1:.2f} & {r4:.2f} & {r5r:.2f} & {r5:.2f} \\\\")
    (TAB / "perclass.tex").write_text("\n".join(lines) + "\n")

    # Table: footprint
    fp = R["E7_footprint"]
    lines = [f"{m} & {fp[m]['size_kB']:.1f} & {fp[m]['latency_ms_per_window']:.3f} \\\\"
             for m in ["LR", "DT", "RF", "LGBM", "MLP"]]
    (TAB / "footprint.tex").write_text("\n".join(lines) + "\n")


def fig_detection():
    from run_detection import run
    res, d = run(q=0.99)
    thr = res["threshold"]
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.4), sharey=True)
    for ax, exp in zip(axes.ravel(), ["F5L", "F2L", "F7M", "F4L"]):
        g = d[d["exp"] == exp]
        ax.semilogy(g["t_end"], g["score"].clip(lower=0.1), lw=0.8, color="tab:blue")
        ax.axhline(thr, color="k", ls=":", lw=0.8)
        ax.axvline(g["onset"].iloc[0], color="tab:red", ls="--", lw=0.8)
        al = g[g["alarm"]]
        ax.plot(al["t_end"], al["score"].clip(lower=0.1), ".", ms=2, color="tab:red")
        ax.set_title(exp, fontsize=7, pad=2)
    for ax in axes[1]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$s=\max_j |z_j|$")
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(FIG / "detection.pdf")
    # table
    D = json.load(open(ROOT / "results" / "detection.json"))["q0.99"]
    lines = []
    for f in range(1, 8):
        cells = []
        for m in ["L", "M"]:
            v = D["per_experiment"][f"F{f}{m}"]
            dl = "--" if v["delay_s"] is None else f"{v['delay_s']:.1f}"
            cells += [f"{v['detection_rate_post']:.2f}", dl]
        lines.append(f"F{f} & {FAULT_NAMES[f]} & " + " & ".join(cells) + r" \\")
    (TAB / "detection.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_signals(); fig_protocols(); fig_confusion(); fig_window(); fig_shap(); tables(); fig_detection()
    print("figures and tables written")
