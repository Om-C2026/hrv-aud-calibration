# ECG-Derived HRV for Stress Classification in AUD Relapse Monitoring

**Paper:** ECG-derived heart rate variability achieves robust stress-state classification under leave-one-subject-out validation: a proof-of-concept for scalable alcohol use disorder relapse monitoring with cross-device heart rate agreement from consumer wearables

**Authors:** Om Chhaya, Viswanath Ananth

**Status:** Under peer review at PLOS ONE (PONE-D-26-48878)

---

## Overview

This repository contains the complete analysis for subject-independent HRV-based stress classification using the WESAD dataset, validated with leave-one-subject-out (LOSO) cross-validation to prevent subject leakage.

## Key Results (LOSO Cross-Validation)

| Model | Accuracy | ROC AUC | AUC 95% CI | F1 |
|-------|----------|---------|------------|-----|
| Random Forest | 0.867 | 0.946 | 0.871-0.998 | 0.786 |
| Logistic Regression | 0.800 | 0.873 | 0.720-0.979 | 0.640 |
| SVM (RBF) | 0.756 | 0.893 | 0.788-0.969 | 0.522 |

**Note on subject leakage:** An earlier version of this analysis used a conventional 70/30 train-test split that allowed data from the same subjects in both training and test sets (RF AUC = 0.848). The LOSO results above are the corrected, unbiased estimates.

**Cross-device heart rate comparison (n=1, 7 days):** Empatica EmbracePlus vs Fitbit showed mean HR difference of +3.1 bpm (SD 23.1 bpm). No HRV metrics were derived from Fitbit (1 Hz heart rate only).

---

## Quick Start

### Step 1: Clone and set up

```bash
git clone https://github.com/Om-C2026/hrv-aud-calibration.git
cd hrv-aud-calibration
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install scikit-learn neurokit2 heartpy pandas numpy matplotlib scipy torch jupyter
```

### Step 2: Download the WESAD dataset

1. Go to https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
2. Download and extract into data/raw/wesad/

### Step 3: Run the primary analysis (LOSO)

```bash
python hrv_loso_analysis.py
```

This runs leave-one-subject-out cross-validation for all three classifiers (Random Forest, Logistic Regression, SVM) on both 3-class and binary stress tasks, and reports accuracy, precision, recall, F1, ROC AUC with bootstrap 95% confidence intervals.

**Expected output:**
```
RandomForest (Binary LOSO):
  Accuracy:  0.867 (mean per-fold: 0.867 +/- 0.163)
  Precision: 0.846
  Recall:    0.733
  F1:        0.786
  ROC AUC:   0.946 (95% CI: 0.871 - 0.998)
```

**Runtime:** Under 2 minutes. No GPU required.

### Step 4: Run the original baseline (for comparison)

```bash
jupyter notebook 01_hrv_ml_baseline.ipynb
```

This shows the original train-test split results with subject leakage, for comparison with the corrected LOSO analysis.

### Step 5: Run sequence models

```bash
jupyter notebook 03_hrv_sequence_models.ipynb
```

LSTM and CNN results (both failed to improve over classical ML due to small sample size).

### Step 6: Generate figures

```bash
jupyter notebook 05_reporting.ipynb
```

---

## Repository Structure

```
hrv-aud-calibration/
    hrv_loso_analysis.py                        Primary analysis (LOSO CV)
    01_hrv_ml_baseline.ipynb                    Original baseline (with leakage)
    02_hrv_calibration_transfer*.ipynb          Calibration exploration
    03_hrv_sequence_models.ipynb                LSTM/CNN evaluation
    04_hrv_results_summary.ipynb                Results aggregation
    05_reporting.ipynb                          Figure generation
    S1_Table.csv                                Classifier predictions
    S2_Table.csv                                WESAD HRV features
    README.md
```

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| WESAD (ECG) | Schmidt et al. 2018 | https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ |
| Ambulatory HR (Empatica + Fitbit) | This study, n=1 | Zenodo (DOI pending) |

## Important Notes

- **Fitbit provides 1 Hz heart rate only.** No beat-to-beat intervals are available, so no HRV metrics can be derived from Fitbit data. The cross-device comparison is heart rate agreement only.
- **Empatica EmbracePlus is research-grade**, not FDA-approved for clinical HRV. It provides beat-to-beat intervals suitable for HRV analysis.
- **The ambulatory comparison is n=1** (the first author). Population-level concordance studies are needed.
- **WESAD stress induction is a proxy** for AUD relapse-relevant autonomic arousal. Validation in longitudinal AUD cohorts is needed.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 or later (tested on 3.13) |
| OS | Windows, macOS, Linux |
| RAM | 8 GB minimum |
| GPU | Not required |
| Runtime | Under 10 minutes total |

## Methodology

Initial Python code was written by the lead author and refined using ChatGPT (GPT-4). LOSO cross-validation correction and manuscript preparation used Claude (Anthropic, claude-opus-4-6). The methodology is described in a companion paper under review at Bioinformatics Advances (BIOADV-2026-626).

## Citation

```
Chhaya, O., & Ananth, V. (2026). ECG-derived heart rate variability achieves robust
stress-state classification under leave-one-subject-out validation: a proof-of-concept
for scalable alcohol use disorder relapse monitoring. Under review at PLOS ONE.
PONE-D-26-48878.
```

## License

MIT
