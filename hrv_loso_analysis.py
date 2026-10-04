"""
CORRECTED HRV Baseline Analysis — Leave-One-Subject-Out Cross-Validation
=========================================================================
Fixes subject leakage in the original train_test_split approach.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/mnt/user-data/uploads/WESAD_HRV_features.csv")
print("Loaded:", df.shape)
print("Subjects:", sorted(df["Subject"].unique()))
print("Labels:", df["Label"].value_counts().to_dict())
print("Label names:", df["LabelName"].unique().tolist())

feature_cols = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_LF", "HRV_HF", "HRV_LFHF"]

# Drop NaN rows
mask = df[feature_cols].notna().all(axis=1)
df = df[mask].copy()
print("After NaN drop:", df.shape)

X = df[feature_cols].values
y = df["Label"].values
groups = df["Subject"].values
subjects = df["Subject"].values

print("\n" + "=" * 60)
print("ANALYSIS 1: 3-CLASS (Rest vs Stress vs Amusement)")
print("Leave-One-Subject-Out Cross-Validation")
print("=" * 60)

logo = LeaveOneGroupOut()
models_spec = {
    "LogReg": lambda: LogisticRegression(max_iter=1000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": lambda: SVC(probability=True, kernel="rbf", random_state=42),
}

for name, model_fn in models_spec.items():
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_accs = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        fold_accs.append(accuracy_score(y_test, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)

    try:
        classes = sorted(np.unique(y))
        y_true_bin = label_binarize(all_y_true, classes=classes)
        roc = roc_auc_score(y_true_bin, all_y_prob, multi_class="ovr", average="weighted")
    except Exception as e:
        roc = float("nan")

    print(f"\n{name} (3-class LOSO):")
    print(f"  Accuracy:  {acc:.3f} (mean per-fold: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f})")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  ROC AUC:   {roc:.3f}")
    print(f"  Confusion matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: BINARY (Stress vs Non-stress) — matches the paper
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: BINARY (Stress vs Non-Stress)")
print("Leave-One-Subject-Out Cross-Validation")
print("=" * 60)

# Recode: Label 2 = Stress -> 1, Labels 1,3 = Non-stress -> 0
df2 = df.copy()
df2["BinLabel"] = (df2["Label"] == 2).astype(int)
print("Binary label dist:", df2["BinLabel"].value_counts().to_dict())
print("  0 = Non-stress (Rest + Amusement), 1 = Stress")

X2 = df2[feature_cols].values
y2 = df2["BinLabel"].values
groups2 = df2["Subject"].values

for name, model_fn in models_spec.items():
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_accs = []

    for train_idx, test_idx in logo.split(X2, y2, groups2):
        X_train, X_test = X2[train_idx], X2[test_idx]
        y_train, y_test = y2[train_idx], y2[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        fold_accs.append(accuracy_score(y_test, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    try:
        roc = roc_auc_score(all_y_true, all_y_prob)
    except:
        roc = float("nan")

    # Bootstrap CI for AUC
    n_boot = 1000
    boot_aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(all_y_true), len(all_y_true), replace=True)
        if len(np.unique(all_y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(all_y_true[idx], all_y_prob[idx]))
        except:
            pass
    ci_lo = np.percentile(boot_aucs, 2.5) if boot_aucs else float("nan")
    ci_hi = np.percentile(boot_aucs, 97.5) if boot_aucs else float("nan")

    print(f"\n{name} (Binary LOSO):")
    print(f"  Accuracy:  {acc:.3f} (mean per-fold: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f})")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  ROC AUC:   {roc:.3f} (95% CI: {ci_lo:.3f} - {ci_hi:.3f})")
    print(f"  Confusion matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)
    # TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")

print("\n" + "=" * 60)
print("COMPARISON: Original (leaked) vs Corrected (LOSO)")
print("=" * 60)
print("Original (train_test_split, subject leakage):")
print("  RF: Accuracy=0.800, AUC=0.848")
print("  LogReg: Accuracy=0.867, AUC=1.000")
print("")
print("Corrected results shown above.")
print("The difference quantifies the inflation from subject leakage.")
