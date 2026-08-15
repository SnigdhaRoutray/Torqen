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
    └── Torqen_Report.docx