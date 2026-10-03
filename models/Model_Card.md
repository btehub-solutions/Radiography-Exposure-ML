# Model Card: Radiographic Exposure Parameter Predictor (Ridge Regression)

## 1. Model Details
- **Developer:** Clinical Data Science Research Team
- **Model Type:** L2-Regularized Linear Regression (Ridge Regression, Alpha = 10.0)
- **Target Variables:** 
  - Peak Kilovoltage (`kVp`): Controls image contrast and beam penetrability.
  - Milliampere-Seconds (`mAs`): Dictates tube current-time product / total radiation quantity.
- **Date:** October 2026
- **Version:** 1.0.0
- **License:** Research Only

## 2. Intended Use
- **Primary Purpose:** Diagnostic decision support prototype for thoracic radiography technical configurations.
- **Intended Users:** Radiologic technologists, medical physicists, and imaging researchers.
- **Strict Constraint:** **RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.** All configurations must be independently verified by a licensed clinical radiographer before patient exposure.

## 3. Training & Validation Data
- **Dataset Cohort:** 300 adult subjects (86.0% optimal exposure rate utilized for primary training; n = 258 optimal subset partitioned 80/20 train/test).
- **Population Demographics:** 
  - Age: 18–60 years (Mean: 38.8 ± 12.2 years)
  - Sex: 51.3% Female, 48.7% Male
  - Weight: 45.0–110.9 kg (Mean: 65.8 ± 11.1 kg)
  - BMI: 17.4–34.3 kg/m² (WHO Classes: Underweight, Normal, Overweight, Obese)
  - Chest Thickness: 20.0–26.9 cm (Mean: 22.8 ± 1.6 cm)
- **Gating Factor:** Target exposures restricted strictly to a Source-to-Image Distance (SID) of exactly 180 cm.

## 4. Input Variables & Units
1. Biological Sex (`sex_encoded`): Binary [0 = Female, 1 = Male]
2. Age (`age`): Continuous, Years
3. Patient Weight (`weight_kg`): Continuous, kilograms (kg)
4. Patient Height (`height_m`): Continuous, meters (m)
5. Body Mass Index (`bmi`): Continuous, kg/m² (computed from weight and height)
6. Anteroposterior Chest Thickness (`chest_cm`): Continuous, centimeters (cm)

## 5. Performance Metrics on Unseen Test Split (N = 52)
| Target Metric | Test MAE | 95% Confidence Interval | Test RMSE | 95% Confidence Interval | Test R² | Test MAPE (%) |
|---|---|---|---|---|---|---|
| **kVp** | 2.0186 | [1.65, 2.42] | 2.4748 | [2.00, 3.01] | 0.2054 | 2.77% |
| **mAs** | 0.9246 | [0.75, 1.11] | 1.1438 | [0.92, 1.35] | 0.0618 | 8.42% |

## 6. Limits & Practical Constraints
- **Floor and Ceiling Scaling:** The training data exhibits strong boundary clustering (such as the minimum voltage limit at 70 kVp). The model predictions near boundaries may present minor physical skew due to clinical preference bounds.
- **Demographics:** Model restricts predictions strictly to adult populations (18 to 60 years). Do not apply to pediatric cohorts.
- **Fixed Geometrics:** Constrained entirely to 180 cm SID chest acquisitions; not applicable to alternative distances or exams.
- **Ethical Notes:** Prevents dose optimization bias across biometric extremes through robust regularized parameters.