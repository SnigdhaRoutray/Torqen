"""
Torqen — Report Figure Generator
Produces Figures 3, 4, 6, 7, 8, 9, 10 for the report (Figures 1, 2, 5 are
RapidMiner screenshots and are NOT produced by this script).

All charts are single-color (black/gray only), matching the report's
IEEE-style formatting requirement.

RUN THIS FROM THE PROJECT ROOT FOLDER (e.g. Torqen/), not from
inside src/. It reads:
    data/raw/ai4i2020.csv                                         (for Fig. 3, 4)
    outputs/predictions_for_cost_model_threshold_sweep.csv       (for Fig. 7)
    outputs/predictions_for_cost_model_capacity_sensitivity.csv  (for Fig. 8)
    outputs/predictions_for_cost_model_policy_simulation.csv     (for Fig. 9)
    outputs/predictions_for_cost_model_sensitivity_results.csv   (for Fig. 10)
Figure 6 (feature importance) uses hardcoded values taken directly from
the validated RapidMiner AttributeWeights result (Table III), since that
output was not separately exported to CSV.

USAGE (from the project root):
    python src/generate_figures.py

Each figure is saved as a separate PNG directly into the figures/ folder
at the project root. Run once, all 7 images appear there.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "figures"
GRAY = "#4d4d4d"
DARK = "#1a1a1a"
LIGHT = "#a6a6a6"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})


def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_class_distribution(ai4i_csv):
    df = pd.read_csv(ai4i_csv, sep=None, engine="python")
    counts = df["Machine failure"].value_counts().sort_index()
    labels = ["No Failure (0)", "Failure (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=[LIGHT, DARK], edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Machine Failure Class Distribution (AI4I 2020, n=10,000)")
    ax.set_ylim(0, 11000)
    savefig(fig, "fig3_class_distribution.png")


def fig4_failure_mode_distribution(ai4i_csv):
    df = pd.read_csv(ai4i_csv, sep=None, engine="python")
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    values = [int(df[m].sum()) for m in modes]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(modes, values, color=DARK, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Count of positive occurrences")
    ax.set_title("Failure Mode Distribution")
    ax.set_ylim(0, 130)
    savefig(fig, "fig4_failure_mode_distribution.png")


def fig6_feature_importance():
    # Hardcoded from validated RapidMiner AttributeWeights output (Table III)
    features = ["torque_speed_product", "Torque [Nm]", "Rotational speed [rpm]",
                "temperature_difference", "Tool wear [min]", "Air temperature [K]",
                "Process temperature [K]", "Type"]
    weights = [0.012, 0.008, 0.005, 0.003, 0.003, 0.001, 0.000, 0.000]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = range(len(features))
    ax.barh(y_pos, weights, color=DARK, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # highest weight on top
    ax.set_xlabel("Gini Index Weight")
    ax.set_title("Feature Importance (Gini Index)")
    ax.set_xlim(0, 0.014)
    savefig(fig, "fig6_feature_importance.png")


def fig7_threshold_cost(threshold_csv):
    df = pd.read_csv(threshold_csv, sep=None, engine="python")
    df = df.sort_values("threshold")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["threshold"], df["total_cost"], marker="o", color=DARK, linewidth=1.5)

    best_row = df.loc[df["total_cost"].idxmin()]
    ax.axvline(best_row["threshold"], color=GRAY, linestyle="--", linewidth=1)
    ax.text(best_row["threshold"] + 0.01, ax.get_ylim()[1] * 0.9,
            f"t* = {best_row['threshold']}", fontsize=10)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Operational Cost vs. Decision Threshold")
    savefig(fig, "fig7_threshold_cost.png")


def fig8_capacity_sensitivity(capacity_csv):
    df = pd.read_csv(capacity_csv, sep=None, engine="python")
    df = df.sort_values("capacity_K")

    # Add the unlimited-capacity ceiling point (K=678) if not already present
    ceiling_K, ceiling_saving = 678, 935005
    K_vals = list(df["capacity_K"]) + [ceiling_K]
    saving_vals = list(df["saving_captured"]) + [ceiling_saving]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_vals, saving_vals, marker="o", color=DARK, linewidth=1.5)
    ax.set_xlabel("Maintenance Capacity (K, machines/window)")
    ax.set_ylabel("Expected Cost Saving Captured")
    ax.set_title("Expected Saving Captured vs. Maintenance Capacity")
    savefig(fig, "fig8_capacity_sensitivity.png")


def fig9_policy_simulation(policy_csv):
    df = pd.read_csv(policy_csv, sep=None, engine="python")
    order = ["Reactive", "Fixed Schedule", "Torqen"]
    df["policy"] = pd.Categorical(df["policy"], categories=order, ordered=True)
    df = df.sort_values("policy")

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(df["policy"].astype(str), df["total_cost"], color=[LIGHT, "#808080", DARK], edgecolor="black")
    for b, v in zip(bars, df["total_cost"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 50000, f"{int(v):,}", ha="center", fontsize=9)
    ax.set_ylabel("Total 52-Week Cost")
    ax.set_title("Total Simulated Maintenance Cost by Policy\n(52 Weeks, 200 Machines)")
    savefig(fig, "fig9_policy_simulation.png")


def fig10_sensitivity_analysis(sensitivity_csv):
    df = pd.read_csv(sensitivity_csv, sep=None, engine="python")
    scenario_order = ["Low downtime cost", "Baseline", "High downtime cost"]

    rows = []
    for scenario in scenario_order:
        sub = df[df["scenario"] == scenario]
        reactive = sub[sub["policy"] == "Reactive"]["total_cost"].values[0]
        torqen = sub[sub["policy"] == "Torqen"]["total_cost"].values[0]
        cost = sub[sub["policy"] == "Reactive"]["emergency_failure_cost"].values[0] \
            if "emergency_failure_cost" in sub.columns else None
        savings_pct = (1 - torqen / reactive) * 100
        rows.append((scenario, cost, savings_pct))

    labels = [f"{s.split(' ')[0]}\n({int(c):,})" if c is not None else s for s, c, _ in rows]
    values = [v for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=[LIGHT, "#808080", DARK], edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Torqen Cost Savings vs. Reactive (%)")
    ax.set_title("Robustness of Torqen Cost Advantage\nAcross Downtime-Cost Scenarios")
    ax.set_ylim(0, 90)
    savefig(fig, "fig10_sensitivity_analysis.png")


def main():
    ai4i_csv = os.path.join("data", "raw", "ai4i2020.csv")
    threshold_csv = os.path.join("outputs", "predictions_for_cost_model_threshold_sweep.csv")
    capacity_csv = os.path.join("outputs", "predictions_for_cost_model_capacity_sensitivity.csv")
    policy_csv = os.path.join("outputs", "predictions_for_cost_model_policy_simulation.csv")
    sensitivity_csv = os.path.join("outputs", "predictions_for_cost_model_sensitivity_results.csv")

    missing = [p for p in [ai4i_csv, threshold_csv, capacity_csv, policy_csv, sensitivity_csv] if not os.path.exists(p)]
    if missing:
        print("The following required files were not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nThis script expects to be run from the project ROOT folder (e.g. Torqen/),")
        print("with ai4i2020.csv under data/raw/ and the other CSVs under outputs/.")
        print("Either move the files to match this structure, or edit the paths at the top of main() in this script.")
        sys.exit(1)

    fig3_class_distribution(ai4i_csv)
    fig4_failure_mode_distribution(ai4i_csv)
    fig6_feature_importance()
    fig7_threshold_cost(threshold_csv)
    fig8_capacity_sensitivity(capacity_csv)
    fig9_policy_simulation(policy_csv)
    fig10_sensitivity_analysis(sensitivity_csv)

    print(f"\nAll figures saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()