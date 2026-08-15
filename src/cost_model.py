"""
Torqen Cost Model
Phase 17-19 of the roadmap: turn a failure probability into a cost-optimal
maintenance action (MONITOR / SCHEDULE / STOP).

Usage:
    python cost_model.py predictions_for_cost_model.csv

Expects a CSV with at least a column named 'confidence(1)' containing the
predicted probability of failure per row (this is what RapidMiner's Apply
Model operator outputs). Adjust COLUMN NAMES below if yours differ.
"""

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# 1. COST ASSUMPTIONS
# ---------------------------------------------------------------------------
# These are ASSUMPTIONS, not measured factory costs — document this clearly
# in your report (Phase 17 explicitly requires this disclosure).
#
# Rough anchoring logic (cite in your report):
#   - Industry benchmarks put average unplanned downtime around $260k/hr
#     at large multi-line facilities (Aberdeen/Siemens-Senseye surveys).
#   - Academic sources (e.g. arXiv work on maintenance optimization) cite
#     per-machine/per-line downtime costs up to ~$70-80k/hr for high-value
#     equipment, and ~$50B/year industry-wide unplanned downtime losses.
#   - AI4I is single-machine granularity, not a whole facility, so costs
#     below are scaled DOWN from those enterprise figures to a plausible
#     single-machine order of magnitude. Treat these as illustrative.
#
# All figures in an arbitrary currency unit (call it INR or USD, your choice
# — just be consistent and state the unit in your report).

COST = {
    # (action, actual_state): cost
    ("MONITOR", "no_failure"): 0,        # correct: no action needed, no cost
    ("MONITOR", "failure"):    18000,    # missed failure -> emergency/unplanned downtime (raised: a missed failure should be far costlier than an unnecessary stop, given downtime research anchors)
    ("SCHEDULE", "no_failure"): 300,     # unnecessary but low-cost planned check
    ("SCHEDULE", "failure"):   800,      # caught early, planned repair, some downtime
    ("STOP", "no_failure"):    600,      # unnecessary shutdown, lost production time
    ("STOP", "failure"):       400,      # correctly caught, controlled shutdown, cheapest fix
}

ACTIONS = ["MONITOR", "SCHEDULE", "STOP"]


def expected_cost(p_failure: float, action: str) -> float:
    """Expected cost of taking `action` given predicted P(failure) = p_failure."""
    p_no_failure = 1 - p_failure
    return (
        p_no_failure * COST[(action, "no_failure")]
        + p_failure * COST[(action, "failure")]
    )


def recommend_action(p_failure: float) -> tuple[str, float, dict]:
    """Return the cost-minimizing action, its expected cost, and all three costs."""
    costs = {a: expected_cost(p_failure, a) for a in ACTIONS}
    best_action = min(costs, key=costs.get)
    return best_action, costs[best_action], costs


def main(csv_path: str, prob_column: str = "confidence(1)"):
    # RapidMiner's Write CSV sometimes uses ';' instead of ',' depending on
    # locale/settings — auto-detect the delimiter instead of assuming comma.
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if prob_column not in df.columns:
        print(f"Column '{prob_column}' not found. Available columns:")
        print(list(df.columns))
        sys.exit(1)

    results = []
    for _, row in df.iterrows():
        p_fail = row[prob_column]
        action, exp_cost, all_costs = recommend_action(p_fail)
        results.append({
            "failure_probability": p_fail,
            "recommended_action": action,
            "expected_cost": exp_cost,
            "cost_if_monitor": all_costs["MONITOR"],
            "cost_if_schedule": all_costs["SCHEDULE"],
            "cost_if_stop": all_costs["STOP"],
        })

    out = pd.DataFrame(results)
    # keep original columns too (e.g. true label, machine id if present)
    out = pd.concat([df.reset_index(drop=True), out], axis=1)

    out_path = csv_path.replace(".csv", "_with_decisions.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}\n")
    print("Decision breakdown:")
    print(out["recommended_action"].value_counts())
    print(f"\nTotal expected cost across all machines: {out['expected_cost'].sum():,.2f}")

    # ---------------------------------------------------------------------
    # PHASE 26: Maintenance capacity constraint / priority queue
    # ---------------------------------------------------------------------
    # Only K machines can actually be serviced this window. Rank candidates
    # (anything not already MONITOR) by expected saving = cost if left as
    # MONITOR minus the cost of the recommended action. Take the top K.
    K_VALUES = [20, 50, 100, 200]  # test multiple capacity assumptions

    candidates_base = out[out["recommended_action"] != "MONITOR"].copy()
    candidates_base["expected_saving"] = candidates_base["cost_if_monitor"] - candidates_base["expected_cost"]
    candidates_base = candidates_base.sort_values("expected_saving", ascending=False)

    print(f"\n--- Maintenance Priority Queue (capacity sensitivity) ---")
    print(f"Candidates needing action: {len(candidates_base)}")

    summary_rows = []
    for K in K_VALUES:
        pq = candidates_base.head(K).reset_index(drop=True)
        pq.insert(0, "priority_rank", range(1, len(pq) + 1))
        saving_captured = pq["expected_saving"].sum()
        deferred = candidates_base.iloc[K:]
        saving_missed = deferred["expected_saving"].sum()

        pq_path = csv_path.replace(".csv", f"_priority_queue_K{K}.csv")
        pq.to_csv(pq_path, index=False)

        summary_rows.append({
            "capacity_K": K,
            "serviced": min(K, len(candidates_base)),
            "deferred": max(0, len(candidates_base) - K),
            "saving_captured": round(saving_captured, 2),
            "saving_missed": round(saving_missed, 2),
        })
        print(f"K={K:>4} | serviced={min(K, len(candidates_base)):>4} | "
              f"saving captured={saving_captured:,.0f} | saving missed={saving_missed:,.0f} "
              f"| saved to {pq_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = csv_path.replace(".csv", "_capacity_sensitivity.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCapacity sensitivity summary saved: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cost_model.py <predictions_csv>")
        sys.exit(1)
    main(sys.argv[1])