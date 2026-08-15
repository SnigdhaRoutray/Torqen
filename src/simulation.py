"""
Torqen Maintenance Policy Simulation
Phase 28-29 of the roadmap: compare Reactive vs Fixed Schedule vs Torqen
maintenance policies over a simulated fleet and time horizon.

METHODOLOGY / ASSUMPTIONS (document these explicitly in the report):
- Fleet of N_MACHINES simulated over T_PERIODS weekly periods (1 year).
- AI4I is a cross-sectional (not time-series) dataset, so each period, each
  machine's state is drawn (with replacement) from the real predictions
  CSV — reusing genuine (predicted probability, actual outcome) pairs from
  the trained Random Forest rather than inventing synthetic randomness.
  This preserves the real relationship between predicted risk and actual
  failure while extending a static dataset into a repeated-period
  simulation. This is a documented simplification, not real longitudinal
  machine data.
- Reactive: no monitoring; failures cost the full emergency cost.
- Fixed Schedule: every FIXED_INTERVAL weeks, service all machines
  regardless of risk; other weeks behave like Reactive.
- Torqen: rank machines by predicted P(failure) each period, flag
  those above THRESHOLD, subject to a per-period capacity cap; flagged
  machines that would have failed are "caught" at lower cost, flagged
  machines that wouldn't have failed cost an unnecessary-action fee,
  unflagged machines that fail cost the full emergency cost.

Usage:
    python simulation.py predictions_for_cost_model.csv
"""

import sys
import numpy as np
import pandas as pd

N_MACHINES = 200
T_PERIODS = 52
FIXED_INTERVAL = 4          # weeks between scheduled maintenance
CAPACITY_PER_PERIOD = 20    # Torqen capacity constraint per period
THRESHOLD = 0.04            # cost-optimal threshold from Phase 20

COST_EMERGENCY_FAILURE = 18000   # reactive / missed failure
COST_FIXED_MAINTENANCE = 300     # scheduled maintenance, per machine
COST_CAUGHT_FAILURE = 600        # Torqen correctly flagged, failure occurred
COST_UNNECESSARY_FLAG = 450      # Torqen flagged, no failure occurred

# Sensitivity analysis scenarios (Phase 30) - vary the emergency failure
# cost, since that's the single assumption with the most influence on
# every policy comparison. Low/Baseline/High reflect uncertainty in how
# expensive an unplanned failure really is.
SCENARIOS = {
    "Low downtime cost":      {"COST_EMERGENCY_FAILURE": 9000},
    "Baseline":                {"COST_EMERGENCY_FAILURE": 18000},
    "High downtime cost":      {"COST_EMERGENCY_FAILURE": 30000},
}

RANDOM_SEED = 42


def simulate(df: pd.DataFrame, prob_col: str, label_col: str, cost_emergency_failure: float = COST_EMERGENCY_FAILURE):
    rng = np.random.default_rng(RANDOM_SEED)

    reactive_cost = 0
    fixed_cost = 0
    Torqen_cost = 0

    reactive_failures = 0
    fixed_failures = 0
    Torqen_failures = 0
    Torqen_caught = 0
    Torqen_flags_total = 0

    for t in range(T_PERIODS):
        sample = df.sample(n=N_MACHINES, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        probs = sample[prob_col].values
        actual = sample[label_col].values

        # --- Reactive ---
        reactive_cost += (actual == 1).sum() * cost_emergency_failure
        reactive_failures += (actual == 1).sum()

        # --- Fixed Schedule ---
        if t % FIXED_INTERVAL == 0:
            fixed_cost += N_MACHINES * COST_FIXED_MAINTENANCE
        else:
            fixed_cost += (actual == 1).sum() * cost_emergency_failure
            fixed_failures += (actual == 1).sum()

        # --- Torqen ---
        order = np.argsort(-probs)  # descending risk
        flagged = np.zeros(N_MACHINES, dtype=bool)
        flagged_count = 0
        for idx in order:
            if flagged_count >= CAPACITY_PER_PERIOD:
                break
            if probs[idx] >= THRESHOLD:
                flagged[idx] = True
                flagged_count += 1
        Torqen_flags_total += flagged_count

        caught = flagged & (actual == 1)
        unnecessary = flagged & (actual == 0)
        missed = (~flagged) & (actual == 1)

        Torqen_cost += (
            caught.sum() * COST_CAUGHT_FAILURE
            + unnecessary.sum() * COST_UNNECESSARY_FLAG
            + missed.sum() * cost_emergency_failure
        )
        Torqen_caught += caught.sum()
        Torqen_failures += missed.sum()  # failures NOT caught under Torqen

    return {
        "Reactive": {
            "total_cost": reactive_cost,
            "uncaught_failures": int(reactive_failures),
        },
        "Fixed Schedule": {
            "total_cost": fixed_cost,
            "uncaught_failures": int(fixed_failures),
        },
        "Torqen": {
            "total_cost": Torqen_cost,
            "uncaught_failures": int(Torqen_failures),
            "failures_caught": int(Torqen_caught),
            "avg_flags_per_period": round(Torqen_flags_total / T_PERIODS, 1),
        },
    }


def main(csv_path: str, prob_col: str = "confidence(1)", label_col: str = "Machine failure"):
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if prob_col not in df.columns or label_col not in df.columns:
        print("Column not found. Available columns:")
        print(list(df.columns))
        sys.exit(1)

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    all_rows = []
    for scenario_name, overrides in SCENARIOS.items():
        cost_ef = overrides.get("COST_EMERGENCY_FAILURE", COST_EMERGENCY_FAILURE)
        results = simulate(df, prob_col, label_col, cost_emergency_failure=cost_ef)

        print(f"=== Scenario: {scenario_name} (emergency failure cost = {cost_ef:,}) ===")
        for policy, stats in results.items():
            print(f"{policy}: total_cost={stats['total_cost']:,}  uncaught_failures={stats['uncaught_failures']}")
        reactive_total = results["Reactive"]["total_cost"]
        Torqen_total = results["Torqen"]["total_cost"]
        savings_pct = (1 - Torqen_total / reactive_total) * 100
        print(f"Torqen savings vs Reactive: {savings_pct:.1f}%\n")

        for policy, stats in results.items():
            row = {"scenario": scenario_name, "emergency_failure_cost": cost_ef, "policy": policy}
            row.update(stats)
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out_path = csv_path.replace(".csv", "_sensitivity_results.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulation.py <predictions_csv>")
        sys.exit(1)
    main(sys.argv[1])