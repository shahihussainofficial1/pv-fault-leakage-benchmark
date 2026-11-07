# Fault Diagnosis or Experiment Recognition?

Leakage-aware evaluation of machine learning for fault diagnosis in grid-connected photovoltaic (PV) systems, using the public [GPVS-Faults](https://doi.org/10.17632/n76t439f65.1) dataset.

## Key findings

| Protocol | What it tests | RF macro-F1 |
|---|---|---|
| P1: file labels, random split (prior work) | protocol used in earlier studies | **1.00** |
| P2: same, but **pre-fault data only** | nothing to diagnose; should be chance (0.125) | **1.00** |
| P3: onset-aware labels, random split | within-recording | 0.81 |
| P4: onset-aware labels, blocked CV + purge | within-recording, no neighbor leakage | 0.79 (FAR 44%) |
| P5: cross-mode (MPPT ↔ IPPT) | unseen recording | 0.25 |
| P5 + baseline referencing (ours) | unseen recording | **0.49** |

- Near-perfect accuracy on GPVS-Faults comes from recognizing **which recording** a window belongs to, not which fault is present.
- **Baseline referencing** (features expressed relative to a 2 s healthy reference of the same installation) roughly doubles cross-mode macro-F1.
- A **training-free detector** (`max |z|` over 19 physics-informed features), calibrated on MPPT data only, transfers to IPPT with a **0.7% false-alarm rate** and detects PV open-circuit (97–99%), sensor (77–79%), inverter (67–77%) and boost-controller (35–46%) faults. Partial shading and MPPT-controller faults remain undetected.

The preprint is available in [`paper/main.pdf`](paper/main.pdf).

## Repository structure

```
scripts/download_data.py   download and extract GPVS-Faults (CC BY 4.0)
src/features.py            windowing, onset-aware labels, statistical & physics features, baseline referencing
src/run_experiments.py     protocols P1-P5, window-length study, model footprint  -> results/results.json
src/run_detection.py       training-free detector                                -> results/detection.json
src/make_figures.py        all figures and LaTeX tables                          -> paper/figures, paper/tables
paper/                     preprint in IEEE conference format (main.pdf, main.tex, references.bib)
results/                   numeric results used in the paper
```

## Reproducing the results

Requires Python 3.10+ and runs on a CPU (about 15 minutes on a laptop).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # ~120 MB
cd src
python run_experiments.py
python run_detection.py
python make_figures.py
```

Compile the paper with any LaTeX distribution (or upload the `paper/` folder to Overleaf):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Dataset

A. Bakdi, A. Guichi, S. Mekhilef, W. Bounoua, *GPVS-Faults: Experimental data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*, Mendeley Data, V1, 2020. doi:10.17632/n76t439f65.1, licensed CC BY 4.0. The data are not redistributed in this repository; use the download script.

## Citation

```bibtex
@misc{shahi2026leakage,
  title  = {Fault Diagnosis or Experiment Recognition? A Leakage-Aware Evaluation of Machine Learning for Grid-Connected PV Fault Diagnosis},
  author = {Haider Md. Faiyaz Hussain Shahi},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/shahihussainofficial1/pv-fault-leakage-benchmark}
}
```

## License

Code: MIT License (see `LICENSE`).
