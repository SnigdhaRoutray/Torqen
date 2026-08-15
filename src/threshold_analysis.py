"""
Torqen Threshold Sweep
Phase 20 of the roadmap: instead of assuming a 0.5 cutoff, test a range of
probability thresholds and see how expected cost, precision, and recall
change. Produces sensitivity_results-style output and identifies t*.

Usage:
    python threshold_analysis.py predictions_for_cost_model.csv
"""

import sys
import pandas as pd

# Same cost assumptions as cost_model.py - keep these in sync
COST_CORRECT_MONITOR = 0
COST_MISSED_FAILURE = 18000
COST_UNNECESSARY_FLAG = 450   # blended SCHEDULE/STOP cost for "flag as risk"
COST_CAUGHT_FAILURE = 600     # blended cost when correctly flagged

THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def evaluate_threshold(df: pd.DataFrame, threshold: float, prob_col: str, label_col: str):
    flagged = df[prob_col] >= threshold
    actual_fail = df[label_col] == 1

    tp = (flagged & actual_fail).sum()          # correctly flagged failures
    fp = (flagged & ~actual_fail).sum()          # unnecessary flags
    fn = (~flagged & actual_fail).sum()          # missed failures
    tn = (~flagged & ~actual_fail).sum()         # correctly left alone

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    total_cost = (
        tp * COST_CAUGHT_FAILURE
        + fp * COST_UNNECESSARY_FLAG
        + fn * COST_MISSED_FAILURE
        + tn * COST_CORRECT_MONITOR
    )

    return {
        "threshold": threshold,
        "flagged_count": int(flagged.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "total_cost": total_cost,
    }


def main(csv_path: str, prob_col: str = "confidence(1)", label_col: str = "Machine failure"):
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if prob_col not in df.columns or label_col not in df.columns:
        print("Column not found. Available columns:")
        print(list(df.columns))
        sys.exit(1)

    results = [evaluate_threshold(df, t, prob_col, label_col) for t in THRESHOLDS]
    out = pd.DataFrame(results)

    out_path = csv_path.replace(".csv", "_threshold_sweep.csv")
    out.to_csv(out_path, index=False)

    best_row = out.loc[out["total_cost"].idxmin()]

    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")
    print(f"\nOptimal threshold (t*): {best_row['threshold']}")
    print(f"  -> Total cost: {best_row['total_cost']:,.0f}")
    print(f"  -> Recall: {best_row['recall']:.1%}  |  Precision: {best_row['precision']:.1%}")
    print(f"  -> Missed failures: {int(best_row['false_negatives'])}  |  Unnecessary flags: {int(best_row['false_positives'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python threshold_analysis.py <predictions_csv>")
        sys.exit(1)
    main(sys.argv[1])