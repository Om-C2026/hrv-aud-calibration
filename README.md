# Calibrated PPG Heart Rate Variability for AUD Relapse Monitoring

**Paper:** Calibrated photoplethysmography-derived heart rate variability approximates electrocardiographic gold-standard performance for stress-state classification: a proof-of-concept for scalable alcohol use disorder relapse monitoring

**Authors:** Om Chhaya, Viswanath Ananth

**Status:** Under peer review at PLOS ONE (PONE-S-26-64505)

---

## Overview

This repository contains everything needed to reproduce the complete analysis: ECG baseline classification, PPG-to-ECG calibration, sequence model evaluation, and cross-device ambulatory comparison. From a clean machine to final figures in under 30 minutes.

## Key Results

| Model | Accuracy | ROC AUC | F1 |
|-------|----------|---------|-----|
| Random Forest (ECG) | 0.80 | 0.85 | 0.77 |
| PPG uncalibrated | 0.80 | 0.79 | 0.77 |
| PPG calibrated | 0.80 | 0.85 | 0.77 |
| LSTM | 0.60 | N/A | 0.75 |
| CNN | 0.00 | N/A | 0.00 |

---

## Quick Start (Complete Recipe)

### Step 1: Clone this repository

```bash
git clone https://github.com/Om-C2026/hrv-aud-calibration.git
cd hrv-aud-calibration
```

### Step 2: Set up Python environment

Requires Python 3.11 or later. No GPU needed.

**Windows:**
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If no requirements.txt is present, install manually:
```bash
pip install scikit-learn neurokit2 heartpy pandas numpy matplotlib scipy torch jupyter
```

### Step 3: Download the WESAD dataset

1. Go to https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
2. Download the WESAD dataset (requires academic registration)
3. Extract into data/raw/wesad/ so the structure looks like:

```
data/
  raw/
    wesad/
      S2/
        S2.pkl
      S3/
        S3.pkl
      ...
```

### Step 4: Download ambulatory wearable data (optional)

The 7-day Empatica EmbracePlus + Fitbit concurrent recordings are available from Zenodo (DOI pending). Download and place in:

```
data/
  raw/
    ambulatory/
      Empatica_Fitbit_1Hz.csv
      Empatica_HRV_Daily.csv
      Fitbit_HRV_Daily.csv
```

This step is optional. The WESAD analysis (Steps 5-8) runs without it.

### Step 5: Run the baseline classifier

```bash
jupyter notebook 01_hrv_ml_baseline.ipynb
```

**What it does:**
- Loads WESAD ECG and PPG recordings for all 15 subjects
- Extracts HRV features: RMSSD, SDNN, pNN50 (time-domain), LF, HF, LF/HF (frequency-domain), sample entropy, DFA (nonlinear)
- Splits into 70/30 stratified train/test (35 train, 15 test)
- Trains Logistic Regression, Random Forest, and SVM (RBF kernel)
- Produces: accuracy, precision, recall, F1, ROC AUC for each model
- Saves baseline_with_labels.csv (predictions + probabilities for ROC)

**Expected output:**
```
Random Forest: Accuracy=0.800, AUC=0.848, F1=0.769
Logistic Reg:  Accuracy=0.867, AUC=1.000, F1=0.833
SVM (RBF):     Accuracy=0.800, AUC=0.848, F1=0.769
```

**Runtime:** about 2 minutes

### Step 6: Run the calibration analysis

```bash
jupyter notebook 02_hrv_calibration_transfer_with_debug_and_ROC.ipynb
```

**What it does:**
- Extracts paired ECG and PPG HRV features from WESAD
- Fits per-feature linear regression (PPG to ECG mapping)
- Computes calibration quality: R-squared, MAE, RMSE per feature
- Runs classification using uncalibrated PPG features (AUC = 0.79)
- Runs classification using calibrated PPG features (AUC = 0.85)
- Produces calibration scatter plots and comparison bar chart

**Expected output:**
```
Uncalibrated PPG AUC: 0.790
Calibrated PPG AUC:   0.848  (matches ECG gold standard)
```

**Runtime:** about 2 minutes

### Step 7: Run sequence models

```bash
jupyter notebook 03_hrv_sequence_models.ipynb
```

**What it does:**
- Trains LSTM (64 hidden units) on HRV time series
- Trains 1D CNN (32 filters) on HRV time series
- Evaluates both against the Random Forest baseline
- Documents the failure (LSTM: 60% accuracy, CNN: 0%) and explains why (n=35 training samples insufficient for deep architectures)

**Expected output:**
```
LSTM: Accuracy=0.600, F1=0.750, AUC=N/A
CNN:  Accuracy=0.000, F1=0.000, AUC=N/A
```

**Runtime:** about 3 minutes

### Step 8: Generate results summary and figures

```bash
jupyter notebook 04_hrv_results_summary.ipynb
jupyter notebook 05_reporting.ipynb
```

**What it does:**
- Aggregates results across all models into summary tables
- Generates publication-quality figures: ROC curves, confusion matrices, calibration comparison bar chart, cross-device 24-hour and 7-day heart rate traces (if ambulatory data downloaded)
- Saves all figures to results/

**Runtime:** about 2 minutes

---

## Repository Structure

```
hrv-aud-calibration/
    01_hrv_ml_baseline.ipynb                    Step 5: ECG baseline
    02_hrv_calibration_transfer*.ipynb          Step 6: PPG calibration
    03_hrv_sequence_models.ipynb                Step 7: LSTM/CNN
    04_hrv_results_summary.ipynb                Step 8a: Results
    05_reporting.ipynb                          Step 8b: Figures
    baseline_with_labels.csv                    Classifier predictions (S1 Table)
    WESAD_HRV_features.csv                      Extracted features (S2 Table)
    requirements.txt                            Python dependencies
    data/
        raw/
            wesad/                              WESAD dataset (download separately)
            ambulatory/                         Empatica/Fitbit data (from Zenodo)
    results/
        figures/                                Generated publication figures
        tables/                                 Generated result tables
    README.md
```

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| WESAD (ECG + PPG) | Schmidt et al. 2018 | https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ |
| Ambulatory (Empatica + Fitbit) | This study | Zenodo (DOI pending) |

## HRV Features Extracted

| Domain | Feature | Description |
|--------|---------|-------------|
| Time | RMSSD | Root mean square of successive RR differences (parasympathetic) |
| Time | SDNN | Standard deviation of NN intervals (overall variability) |
| Time | pNN50 | Percentage of successive intervals differing by more than 50ms |
| Frequency | LF | Low-frequency power, 0.04-0.15 Hz |
| Frequency | HF | High-frequency power, 0.15-0.40 Hz (vagal tone) |
| Frequency | LF/HF | Sympathovagal balance ratio |
| Nonlinear | SampEn | Sample entropy (signal complexity) |
| Nonlinear | DFA | Detrended fluctuation analysis (short-term correlations) |

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 or later (tested on 3.13) |
| OS | Windows 10/11, macOS, Linux |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | Not required |
| Disk | About 2 GB (including WESAD dataset) |
| Runtime | Under 30 minutes total (all notebooks) |

## Methodology

This study uses the Prompt-as-a-Protocol (PaaP) methodology with Generative Research Automation (GRA) for pipeline generation and quality assurance. The initial Python code was written by the lead author and refined using ChatGPT (GPT-4) for code optimization and pipeline implementation. Protocol synthesis and quality-assurance design used Claude (Anthropic, claude-opus-4-6). The PaaP+GRA framework is described in a companion methodology paper (in preparation).

## Troubleshooting

**ModuleNotFoundError: No module named 'neurokit2'**
Run pip install neurokit2 inside your activated virtual environment.

**WESAD .pkl files won't load**
Ensure you are using Python 3.11 or later. Older pickle protocols may cause issues. Try: pip install --upgrade pickle5

**LSTM/CNN showing different results than reported**
Deep learning results vary across runs due to random initialization. The reported results reflect the specific seed used. Set torch.manual_seed(42) for consistency.

**Calibration R-squared values are very low**
This is expected (R-squared = 0.03-0.21). Per-feature calibration is modest, but joint feature calibration recovers classification performance. See Discussion in the paper.

## Citation

```
Chhaya, O., & Ananth, V. (2026). Calibrated photoplethysmography-derived heart rate
variability approximates electrocardiographic gold-standard performance for stress-state
classification: a proof-of-concept for scalable alcohol use disorder relapse monitoring.
Under review at PLOS ONE. PONE-S-26-64505.
```

## License

CC-BY 4.0
