# Torqen: Cost-Aware Maintenance Decision Intelligence

Torqen is a cost-aware predictive maintenance framework that goes beyond failure prediction. It diagnoses likely failure modes, converts risk into a MONITOR/SCHEDULE/STOP decision using an expected-cost model, prioritizes interventions under a maintenance-capacity constraint, and compares the resulting policy against Reactive and Fixed-Schedule maintenance via simulation — using the AI4I 2020 predictive maintenance dataset.

## Folder Structure

```
Torqen-SynTech2026/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│       └── full_featured_dataset.csv
│
├── rapidminer/
│   ├── LogisticRegression.rmp
│   ├── DecisionTree.rmp
│   ├── RandomForest.rmp
│   ├── RandomForest_FeatureEngineering.rmp
│   └── RF_ProbabilityExport.rmp
│
├── src/
│   ├── cost_model.py
│   ├── threshold_analysis.py
│   ├── failure_diagnosis.py
│   ├── simulation.py
│   └── generate_figures.py
│
├── outputs/
│   ├── predictions_for_cost_model.csv
│   ├── predictions_for_cost_model_with_decisions.csv
│   ├── predictions_for_cost_model_threshold_sweep.csv
│   ├── predictions_for_cost_model_capacity_sensitivity.csv
│   ├── predictions_for_cost_model_priority_queue_K20.csv
│   ├── predictions_for_cost_model_priority_queue_K50.csv
│   ├── predictions_for_cost_model_priority_queue_K100.csv
│   ├── predictions_for_cost_model_priority_queue_K200.csv
│   ├── predictions_for_cost_model_policy_simulation.csv
│   ├── predictions_for_cost_model_sensitivity_results.csv
│   └── full_featured_dataset_failure_diagnosis_results.csv
│
├── figures/
│   ├── fig1_rapidminer_process_overview.png
│   ├── fig2_decision_rule_screenshot.png
│   ├── fig3_class_distribution.png
│   ├── fig4_failure_mode_distribution.png
│   ├── fig5_roc_curves.png
│   ├── fig6_feature_importance.png
│   ├── fig7_threshold_cost.png
│   ├── fig8_capacity_sensitivity.png
│   ├── fig9_policy_simulation.png
│   └── fig10_sensitivity_analysis.png
│
└── report/
    └── Torqen_Report.pdf
```

## Dataset

AI4I 2020 Predictive Maintenance Dataset (S. Matzka, UCI Machine Learning Repository, 2020): 10,000 machine observations, 14 attributes, 3.39% failure rate. Download and place at `data/raw/ai4i2020.csv`:
- UCI: https://doi.org/10.24432/C5HS5C
- Kaggle mirror: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

## Reproduction Steps

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **RapidMiner pipeline** (Altair AI Studio)
   Open each `.rmp` file in `rapidminer/` and run it in this order:
   1. `LogisticRegression.rmp`, `DecisionTree.rmp`, `RandomForest.rmp` — model comparison (10-fold CV)
   2. `RandomForest_FeatureEngineering.rmp` — adds engineered features + feature importance
   3. `RF_ProbabilityExport.rmp` — trains Random Forest, exports per-machine failure probabilities to `outputs/predictions_for_cost_model.csv` (also contains the native MONITOR/SCHEDULE/STOP decision rule)

3. **Python analysis** (run from the project root, in order)
   ```
   python src/cost_model.py outputs/predictions_for_cost_model.csv
   python src/threshold_analysis.py outputs/predictions_for_cost_model.csv
   python src/failure_diagnosis.py data/processed/full_featured_dataset.csv
   python src/simulation.py outputs/predictions_for_cost_model.csv
   python src/generate_figures.py
   ```

4. **Report**: `report/Torqen_Report.docx`

## Key Result

Torqen reduces total simulated maintenance cost by **77.0%** vs. Reactive maintenance and **73.4%** vs. Fixed-Schedule maintenance (52-week, 200-machine simulation), a result that holds and strengthens across three downtime-cost scenarios.
