========================================================================
RADIOGRAPHIC EXPOSURE PARAMETERS OPTIMIZATION & MACHINE LEARNING SYSTEM
========================================================================

This delivery archive contains all datasets, tables, figures, serialized estimators,
and clinical documentation developed to optimize exposure parameters (kVp & mAs).

-----------------------------
DIRECTORY STRUCTURE & MANIFEST
-----------------------------
/Radiography_ML/
│
├── data/
│   ├── raw_data_original.xlsx   <- Original clinical xlsx backup copy.
│   ├── cleaned_data.xlsx        <- Preprocessed, audited, and cleaned Excel sheet.
│   ├── cleaned_data.csv         <- Preprocessed data in CSV format.
│   ├── train_ids.csv            <- Patient Record IDs assigned to the training split (80%).
│   └── test_ids.csv             <- Patient Record IDs assigned to the unseen test split (20%).
│
├── tables/
│   ├── All_Tables.xlsx          <- Master consolidated workbook containing Tables 4.1 to 4.11.
│   ├── Data_Dictionary.xlsx     <- Variable definitions, ranges, units, and predictor roles.
│   ├── Table_Data_Audit.xlsx    <- Clean clinical audit constraint checking matrix summary.
│   ├── Table_Cleaning_Log.xlsx  <- Log tracking the 164 field repairs and standardizations.
│   ├── Table_Descriptive.xlsx   <- Raw descriptive statistics and subgroup stratifications.
│   ├── Table_Inferential.xlsx   <- Spearman matrix, normality parameters, OLS, and group test results.
│   ├── Table_Model_Comparison.xlsx  <- Hyperparameter tuning and model cross-validation results.
│   └── Table_Test_Performance.xlsx  <- Continuous evaluation metrics with bootstrap 95% CIs on test set.
│
├── figures/                     <- 300 DPI high-resolution figures (PNG format)
│   ├── Figure_4.1.png           <- Outliers detection boxplot panel.
│   ├── Figure_4.2.png           <- Density histograms for biometrics & exposure parameters.
│   ├── Figure_4.3.png           <- Categorical demographic frequency distributions.
│   ├── Figure_4.4.png           <- kVp and mAs distributions stratified by BMI categories.
│   ├── Figure_4.5.png           <- Bivariate scatter plots overlaid with regression lines.
│   ├── Figure_4.6.png           <- Adjusted Spearman Rank Correlation Heatmap.
│   ├── Figure_4.7.png           <- Regression diagnostic residual plots.
│   ├── Figure_4.8.png           <- Cross-validation MAE boxplots (15 fits per architecture).
│   ├── Figure_4.9.png           <- Overfitting check learning curves for selected Ridge estimators.
│   ├── Figure_4.10.png          <- Actual vs. Predicted values against the identity line.
│   ├── Figure_4.11.png          <- Prediction error residual scatterplots.
│   ├── Figure_4.12.png          <- Distribution profiles of residual errors.
│   ├── Figure_4.13.png          <- Bland-Altman agreement panels and Limits of Agreement.
│   ├── Figure_4.14.png          <- Permutation feature importances with confidence boundaries.
│   ├── Figure_4.15.png          <- Partial Dependence Plots (PDP) of dominant biometrics.
│   └── Figure_4.16.png          <- Scatter plots of mAs recommendations mapped by clinical label.
│
├── models/
│   ├── scaler.joblib            <- Serialized StandardScaler fitted on training biometrics.
│   ├── kvp_model.joblib         <- Serialized Ridge model predicting tube voltage (kVp).
│   ├── mas_model.joblib         <- Serialized Ridge model predicting exposure charge (mAs).
│   ├── Model_Card.md            <- Metadata, intended use, and performance limits.
│   └── requirements.txt         <- Verified Python libraries and environmental versions.
│
└── reports/
    ├── Descriptive_Tables.docx  <- Chapter 4 academic write-ups and descriptive visualizations.
    ├── Inferential_Tables.docx  <- Normality tests, correlation profiles, and hypotheses decisions.
    ├── Model_Comparison_Report.docx  <- Modeling architecture comparisons and learning diagnostics.
    ├── Model_Evaluation.docx    <- Continuous test metrics, interpretability, and clinical validation.
    ├── Methods_Summary.docx     <- Technical methodology summary for Chapter 3 insertion.
    └── Results_Tables_and_Figures.docx  <- Complete consolidated list of all tables and figures in order.