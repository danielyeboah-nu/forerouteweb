# ForeRoute Model Monitoring Report

**Assignment:** Configure a model-monitoring framework to track data skew, drift, observability, anomalies, and model outcome decay.

**Use case:** ForeRoute, a weather-aware safe-routing application. The monitored model is `ForeRoute-BostonRisk@production`, a Boston crash-history classifier served through MLflow and used by the Next.js route planner.

**Artifacts**

- Model card: [`MODEL_CARD.md`](MODEL_CARD.md)
- Monitoring notebook: [`ml/monitoring_assignment.ipynb`](ml/monitoring_assignment.ipynb)
- Evidently reports: [`ml/reports/`](ml/reports/)
- Report generator: [`ml/regenerate_evidently_reports.py`](ml/regenerate_evidently_reports.py)

**Guidance used**

- Chip Huyen, ["Data Distribution Shifts and Monitoring"](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)
- Evidently community notebook, [`bicycle_demand_monitoring_setup.ipynb`](https://github.com/evidentlyai/community-examples/blob/main/tutorials/bicycle_demand_monitoring_setup.ipynb)

---

## Executive Summary

ForeRoute provides route-level safety advice using two signals:

1. **Weather right now:** a deterministic rule-based score for current weather hazards.
2. **Crash history:** the ML model `ForeRoute-BostonRisk`, which estimates whether a road segment has a higher crash-history risk than typical Boston segments.

Because this is a safety-adjacent advisory, silent model decay is a serious risk. A model can become unreliable if production routes differ from training data, if crash patterns change, or if the feature pipeline breaks. The monitoring framework therefore tracks:

- Input distribution drift across all model features.
- Data quality and anomalies such as nulls, invalid ranges, and schema mismatches.
- Model output drift in the prediction-probability distribution.
- Classification performance drift when delayed crash labels are available.

The implementation uses Evidently AI reports generated against a fixed reference window and a current comparison window, following the same reference/current pattern used in Evidently's bicycle-demand monitoring example.

---

## Part 1: Monitoring Design

### 1. Model Inputs, Outputs, and Ground Truth

The monitored ML input is the 15-feature payload sent by the web route planner to the MLflow model server:

| Feature group | Columns |
|---|---|
| Weather | `temperature`, `precipitation_type`, `precipitation_intensity`, `wind_speed`, `visibility`, `humidity`, `dew_point` |
| Road and segment | `road_type`, `segment_distance_m`, `lat`, `lon` |
| Time | `hour_of_day`, `day_of_week`, `month` |
| Historical risk | `prior_year_crash_count` |

The model output is a positive-class probability for each route segment. The web application maps the probability into a plain-language crash-history verdict rather than displaying the raw probability as a literal chance of crashing.

Ground truth is delayed. Updated crash labels come from Boston Vision Zero crash records after new records are published. In a production setting, ground truth could also include driver feedback, incident reports, or telematics, but the academic implementation uses the held-out labelled test split as the current window.

### 2. Data Distribution Shift Framework

Following Chip Huyen's taxonomy, ForeRoute must monitor three related but distinct shift types.

| Shift type | Definition | ForeRoute example | Monitoring signal |
|---|---|---|---|
| Covariate shift | `P(X)` changes while `P(Y | X)` is stable | Winter produces more snow, colder temperatures, and lower visibility than the training baseline. | Input feature drift with Evidently `DataDriftPreset` |
| Label shift | `P(Y)` changes while `P(X | Y)` is stable | Boston changes reporting practices and more minor crashes enter the dataset, changing the hazardous-label rate. | Prediction-distribution drift and backfilled label-rate drift |
| Concept drift | `P(Y | X)` changes | Road redesigns, vehicle safety technology, or changed traffic patterns make the same features imply different crash risk. | Degraded PR-AUC, recall, precision, and F1 after labels arrive |

Covariate shift is the earliest warning because it does not require labels. Concept drift is more dangerous, but it can only be confirmed once delayed labels are available.

### 3. What We Monitor

| Surface | What is checked | Evidently / tooling |
|---|---|---|
| Numerical inputs | Distribution shift on weather, location, time, distance, and prior crash count | `DataDriftPreset` |
| Categorical inputs | Drift in `precipitation_type` and `road_type` | `DataDriftPreset` |
| Data quality | Nulls, dtypes, basic stats, unexpected values, invalid ranges | `DataSummaryPreset` plus custom checks in notebook |
| Model outputs | Drift in `prediction_proba` | `DataDriftPreset` |
| Model performance | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix | `ClassificationPreset` |
| Operational thresholds | PSI above 0.25 or performance drop above 5 percentage points | Alerting policy |

### 4. Reference and Current Windows

The monitoring design uses two datasets:

- **Reference window:** the training split for `ForeRoute-BostonRisk` v3.
- **Current window for this assignment:** the held-out labelled test split, used as a proxy for recent production traffic.

In production, the current window should be replaced with the most recent 7 days of logged route-segment scoring requests.

### 5. Alert and Retraining Policy

| Trigger | Response |
|---|---|
| PSI > 0.25 on any input feature for three consecutive days | Investigate feature pipeline and queue retraining |
| PSI > 0.25 on prediction probability | Check whether the shift is driven by seasonality, feature bugs, or changed route mix |
| PR-AUC or recall drops by at least 5 percentage points on backfilled labels | Consider rollback and retraining |
| New quarterly Vision Zero data release | Rebuild dataset and retrain |
| No drift trigger | Retrain quarterly by default |

Rollback is straightforward because MLflow keeps previous model versions. The production alias can be moved back to a prior version if the new model underperforms.

---

## Part 2: Evidently Implementation

### 1. Monitoring Pipeline

The implemented workflow is:

1. Load `ForeRoute-BostonRisk@production` from MLflow.
2. Load `ml/data/boston_crash_dataset.csv`.
3. Split data into reference and current windows.
4. Score both windows with the production model.
5. Build typed Evidently `Dataset` objects for feature drift and classification performance.
6. Generate HTML reports under `ml/reports/`.

Run command:

```bash
cd ml
make evidently
```

The latest verified run used `ForeRoute-BostonRisk` v3 from MLflow run `94cc207cd66949ba8468c174a99e0ced`.

### 2. Generated Reports

| Report | Purpose |
|---|---|
| `01_data_drift.html` | Detects drift across all 15 input features. |
| `02_data_quality.html` | Summarizes data quality, column statistics, nulls, and schema health. |
| `03_output_drift.html` | Detects drift in model prediction probabilities. |
| `04_classification_performance.html` | Compares classification quality across reference and current windows. |

The monitored model is classifier-only, so no regression-performance report is included.

### 3. Verification Results

The report generator completed successfully and produced four Evidently HTML reports:

| File | Size |
|---|---:|
| `01_data_drift.html` | 4.4 MB |
| `02_data_quality.html` | 3.7 MB |
| `03_output_drift.html` | 3.6 MB |
| `04_classification_performance.html` | 3.7 MB |

The production model used the correct 15-feature schema. A stale `traffic_volume` feature was removed from the training script because it was not present in the Boston dataset or the web API payload.

Model performance for the registered production model:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6506 |
| Accuracy | 0.7146 |
| Precision | 0.3887 |
| Recall | 0.2471 |
| F1 | 0.3021 |

The model's ROC-AUC is modest but expected for this task: crash risk is influenced by many omitted factors such as real-time traffic volume, road geometry, driver behavior, and signal timing. ForeRoute therefore treats the ML result as a crash-history advisory and pairs it with the rule-based current-weather score.

### 4. Observability Gaps and Production Next Steps

The current assignment implementation is an offline monitoring snapshot. To make it production-grade, ForeRoute should add:

1. **Inference logging:** write one JSONL row per scored segment from `/api/routes`, including timestamp, features, prediction probability, verdict, route id, and rule-based score.
2. **Nightly monitoring job:** read the latest 7-day inference window, run the Evidently reports, and store the HTML outputs.
3. **Alerting:** send Slack or email alerts when drift or performance thresholds are breached.
4. **Backfill evaluation:** join new Vision Zero records to historical predictions and recompute classification performance.
5. **Model governance:** review the model card, monitoring report, and generated Evidently reports before promoting a retrained model.

---

## Conclusion

This monitoring framework gives ForeRoute a practical way to detect model decay before users are misled by stale predictions. Evidently provides the core monitoring reports for input drift, output drift, data quality, and classification performance. The design follows Chip Huyen's guidance by separating covariate shift, label shift, and concept drift, and by treating observability as part of the system design rather than a one-time notebook exercise.

The current implementation is sufficient for the assignment: it defines the monitoring approach, applies it to the ForeRoute prediction application, generates reproducible Evidently reports, and documents the remaining steps needed for live production monitoring.
