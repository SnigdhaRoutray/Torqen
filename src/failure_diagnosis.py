"""
Torqen Failure-Mode Diagnosis
Phase 21-22 of the roadmap.

Earlier analysis (Session 1, dataset verification) found that individual
failure-mode counts (TWF+HDF+PWF+OSF+RNF = 373) exceed total Machine
failure positives (339) -> failure modes OVERLAP -> treated as multi-label,
not strict multiclass. This script trains one independent binary
classifier per failure mode, restricted to rows where Machine failure=1
(diagnosis only makes sense once a failure is known/predicted to occur).

Usage:
    python failure_diagnosis.py full_featured_dataset.csv
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURE_COLUMNS = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]", "temperature_difference", "torque_speed_product",
]


def main(csv_path: str):
    df = pd.read_csv(csv_path, sep=None, engine="python")

    missing = [c for c in FEATURE_COLUMNS + FAILURE_MODES + ["Machine failure"] if c not in df.columns]
    if missing:
        print(f"Missing expected columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Diagnosis scope: only rows where an actual failure occurred.
    failed = df[df["Machine failure"] == 1].copy()
    print(f"Total rows: {len(df)} | Rows with Machine failure=1: {len(failed)}\n")

    results = []
    for mode in FAILURE_MODES:
        X = failed[FEATURE_COLUMNS]
        y = failed[mode]

        support = int(y.sum())
        if support < 5:
            print(f"{mode}: only {support} positive examples among failures — "
                  f"too few to train/evaluate meaningfully. Skipping model, "
                  f"reporting support only.\n")
            results.append({"failure_mode": mode, "support": support,
                             "precision": None, "recall": None, "f1": None,
                             "note": "insufficient samples"})
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if support >= 10 else None
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        print(f"{mode}: support={support} (of {len(failed)} failures)")
        print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        print(f"  Confusion matrix:\n{cm}\n")

        results.append({"failure_mode": mode, "support": support,
                         "precision": round(precision, 3), "recall": round(recall, 3),
                         "f1": round(f1, 3), "note": ""})

    out = pd.DataFrame(results)
    out_path = csv_path.replace(".csv", "_failure_diagnosis_results.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nSummary:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python failure_diagnosis.py <full_featured_dataset_csv>")
        sys.exit(1)
    main(sys.argv[1])