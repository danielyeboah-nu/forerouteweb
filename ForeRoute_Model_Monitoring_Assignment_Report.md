---
title: "ForeRoute Model Monitoring Framework"
subtitle: "Data Skew, Drift Detection, Observability, and Model Outcome Monitoring with Evidently AI"
author: "ForeRoute Group"
date: "May 13, 2026"
---

# Contents {.unnumbered}

1. Executive Summary
2. Project Context and AI/ML Use Case
3. Part 1: Monitoring Design
4. Part 2: Evidently AI Implementation
5. Operational Integration Plan
6. Limitations
7. Conclusion
8. Appendix A: Repository Files
9. Appendix B: Submission Checklist

# Executive Summary

ForeRoute is a weather-aware safe-routing application that helps drivers compare possible routes using two complementary safety signals. The first signal is a deterministic "Weather right now" score that evaluates live weather hazards such as snow, heavy precipitation, low visibility, wind, and risky road types. The second signal is the machine-learning model `ForeRoute-BostonRisk`, a Boston crash-history classifier served through MLflow. The model estimates whether a road segment has a higher crash-history risk than a typical Boston road segment.

This report designs and implements a model monitoring framework for ForeRoute using Evidently AI. The framework tracks input data distribution decay, model output drift, data quality anomalies, delayed performance decay, and production observability gaps. The design follows Chip Huyen's guidance that monitoring must distinguish between covariate shift, label shift, and concept drift, and that observability must be built into the system before failures can be diagnosed.

The implemented monitoring pipeline uses a reference/current comparison pattern similar to Evidently's bicycle-demand monitoring example. The reference window is the model's training baseline. The current window, for this academic assignment, is a held-out labelled test split that acts as a stand-in for recent production traffic. The pipeline generates four Evidently HTML reports:

- `01_data_drift.html` for input feature drift.
- `02_data_quality.html` for schema, missingness, range, and distribution summaries.
- `03_output_drift.html` for prediction-probability drift.
- `04_classification_performance.html` for classification quality when labels are available.

The latest verified run used `ForeRoute-BostonRisk` version 3, registered under the MLflow `production` alias. The model uses the same 15-feature schema as the web application payload. The monitoring reports were regenerated successfully under `ml/reports/`.

# Project Context and AI/ML Use Case

ForeRoute is not a general navigation system and does not claim to predict whether a specific driver will crash on a specific trip. Instead, it provides route-level safety context. The product deliberately separates two different questions:

1. Are current weather and road conditions hazardous right now?
2. Does this road segment have a worse crash history than typical Boston road segments?

The first question is handled by rule-based logic because physical driving hazards are known and should remain available even if ML infrastructure fails. The second question is handled by the `ForeRoute-BostonRisk` ML model because historical crash patterns are too local and interaction-heavy to encode manually.

This dual-score design is important for monitoring. The ML model is not the only safety mechanism in the product, but it can still mislead users if it silently degrades. Monitoring therefore focuses on whether the ML component is receiving familiar data, returning stable outputs, and maintaining acceptable performance once delayed crash labels arrive.

## Monitored Model

| Field | Value |
|---|---|
| Model name | `ForeRoute-BostonRisk` |
| Production alias | `ForeRoute-BostonRisk@production` |
| Current version | v3 |
| Model family | Gradient Boosting Classifier |
| Serving framework | MLflow pyfunc |
| Output | Positive-class probability |
| Geographic scope | Boston metro bounding box |
| Ground-truth source | Boston Vision Zero crash records |

The model is trained on segment-level samples from the Boston crash dataset. Positive examples are real crash records, and negative examples represent non-crash segment/time samples. The feature set combines weather, road type, geography, time, and prior crash history.

## Feature Contract

The monitored ML input is the 15-feature payload sent by the Next.js route planner to the MLflow model server.

| Feature group | Features | Monitoring concern |
|---|---|---|
| Weather | `temperature`, `precipitation_type`, `precipitation_intensity`, `wind_speed`, `visibility`, `humidity`, `dew_point` | Seasonal changes, API outages, unit mismatches, rare weather extrapolation |
| Road and segment | `road_type`, `segment_distance_m`, `lat`, `lon` | Route mix changes, out-of-region requests, Mapbox classification changes |
| Time | `hour_of_day`, `day_of_week`, `month` | Time-of-day usage changes, seasonal route behavior |
| Historical risk | `prior_year_crash_count` | Data refresh problems, stale crash history, spatial lookup bugs |

The feature contract must be consistent across training, monitoring, and live inference. During verification for this assignment, a stale `traffic_volume` feature was found in the training script even though it was not present in the Boston dataset or the web API payload. That inconsistency was corrected, and `ForeRoute-BostonRisk` v3 was registered with the correct 15-feature schema.

# Part 1: Monitoring Design

## Monitoring Goals

The monitoring framework is designed to answer five operational questions.

1. Is production data still similar to the data the model was trained on?
2. Are input features valid, complete, and within expected ranges?
3. Are model predictions drifting even before new labels are available?
4. Is classification performance decaying once delayed crash labels arrive?
5. Is the system instrumented well enough to diagnose why a monitoring alert fired?

These questions map directly to the assignment requirements: I/O data distribution decay, model outcome drift, anomaly detection, and observability.

## Distribution Shift Framework

Chip Huyen separates distribution shift into covariate shift, label shift, and concept drift. This distinction matters because each type of shift has a different detection signal and a different response.

| Shift type | Definition | ForeRoute example | Detection approach | Response |
|---|---|---|---|---|
| Covariate shift | `P(X)` changes while `P(Y | X)` stays stable | Winter route requests contain more snow, colder temperatures, and lower visibility than the training baseline. | Evidently input feature drift using numerical and categorical drift checks. | Investigate whether the shift is expected seasonality or a feature pipeline issue. Retrain if sustained. |
| Label shift | `P(Y)` changes while `P(X | Y)` stays stable | Boston changes reporting practices and more minor crashes enter Vision Zero records, changing the observed hazardous-label rate. | Prediction distribution drift before labels; label-rate drift after backfill. | Recalibrate thresholds and review model assumptions. |
| Concept drift | `P(Y | X)` changes | Road redesigns, new traffic patterns, or improved vehicle safety make the same weather and road features imply different crash risk. | Performance decay on backfilled labels: PR-AUC, recall, precision, F1. | Retrain, compare against prior model version, and consider rollback. |

Covariate shift is the earliest signal because it does not require ground truth. Concept drift is the most important model-quality failure, but it is harder to detect because crash labels arrive later. This is why ForeRoute monitors both unlabeled signals, such as feature drift and output drift, and labelled signals, such as classification performance.

## Input Data Distribution Decay

Input data distribution decay occurs when the data arriving at inference time gradually becomes less representative of the data used for training. For ForeRoute, likely causes include seasonal weather, changes in user geography, changes in Mapbox road-type classification, and stale crash-history features.

The monitoring framework compares the reference and current windows for all 15 input features. Numerical features are checked for distribution differences, and categorical features are checked for category-share changes. The practical goal is not to alert on every small difference. The goal is to identify sustained changes large enough to make the model operate outside its familiar training regime.

Important input drift scenarios include:

- A higher share of `precipitation_type = snow` during winter.
- Lower mean `visibility` during storm periods.
- A sudden spike in missing or default weather values caused by an OpenWeather or Open-Meteo API issue.
- A route mix shift from residential roads to highways or bridges.
- A geography shift toward the edge of the Boston bounding box.
- A stale `prior_year_crash_count` lookup that stops changing after a failed data refresh.

The Evidently `DataDriftPreset` report is used to detect these shifts and identify the most affected columns.

## Output Drift and Model Outcome Drift

Output drift measures whether the model's predicted probability distribution changes between the reference and current windows. Output drift can occur even when no labels are available, which makes it useful as an early warning signal.

For ForeRoute, output drift could mean:

- The model is assigning higher crash-history probabilities because winter conditions are more common.
- The route planner is sending a different mix of road types or neighborhoods.
- A feature pipeline bug is pushing many requests toward the same prediction range.
- The production model version or preprocessing schema is inconsistent with the monitoring baseline.

Output drift does not prove that model performance has decayed. It is a proxy signal. However, in a safety-adjacent product, a major prediction-distribution shift is still actionable. It should trigger an investigation into whether the shift is expected, seasonal, data-quality related, or evidence of a model problem.

## Data Quality and Anomaly Monitoring

Data quality checks catch invalid inputs before they become misleading drift signals. A feature can drift because the real world changed, but it can also drift because the pipeline broke. The monitoring framework therefore includes both aggregate data summaries and per-record anomaly checks.

Examples of ForeRoute anomaly checks:

- `temperature` outside a plausible weather range.
- `humidity` below 0 or above 100.
- Negative `visibility`.
- Non-positive `segment_distance_m`.
- Unknown `precipitation_type`.
- Unknown `road_type`.
- Missing latitude or longitude.
- Repeated default values that suggest an upstream API failure.

These anomaly checks are especially important because production model monitoring often fails when teams treat all drift as a model problem. Many apparent model problems are actually data pipeline, schema, or integration failures.

## Performance Drift with Delayed Labels

Performance monitoring requires labels. In ForeRoute, labels are delayed because crash records are updated after incidents are reported and published. Once labels are available, the model should be evaluated on:

- ROC-AUC.
- PR-AUC.
- Accuracy.
- Precision.
- Recall.
- F1.
- Confusion matrix.

Recall is important because the model should not miss too many genuinely high-risk crash-history segments. Precision is also important because excessive false positives would cause the app to overstate risk and reduce user trust. PR-AUC is especially useful because the positive class is less common than the negative class.

The academic implementation uses the held-out labelled test split as the labelled current window. In production, the current labelled window would be built by joining logged predictions to newly released crash records.

## Observability Design

Monitoring tells the team when something is wrong. Observability gives the team enough information to understand why. ForeRoute's production observability should log one row per scored segment.

Recommended inference log fields:

| Category | Fields |
|---|---|
| Request metadata | timestamp, route id, segment id, app version, model name, model version |
| Geography | latitude, longitude, in/out of supported Boston bounding box |
| Weather features | all weather inputs used by the model |
| Road features | road type, segment distance, prior crash count |
| Model outputs | raw probability, thresholded prediction, UI verdict |
| Rule-based outputs | weather score, rule factors |
| Diagnostics | request latency, model-server status, fallback flag |

This log would support nightly drift reports, root-cause analysis, alert triage, and delayed performance backfills.

## Alerting and Retraining Policy

The monitoring policy uses a combination of drift thresholds, performance thresholds, and scheduled retraining.

| Trigger | Action |
|---|---|
| PSI > 0.25 on any input feature for three consecutive days | Investigate feature pipeline and queue retraining if the shift is real. |
| PSI > 0.25 on prediction probability | Check whether the change is seasonal, geographic, or caused by a schema/pipeline bug. |
| PR-AUC drops by at least 5 percentage points on backfilled labels | Compare with previous model, consider rollback, and retrain. |
| Recall drops by at least 5 percentage points on backfilled labels | Review threshold calibration and missed-positive segments. |
| New quarterly Vision Zero release | Rebuild dataset, retrain, regenerate reports, and update model card. |
| No alert | Retrain quarterly as a maintenance baseline. |

The production alias in MLflow allows rollback. If a newly promoted model performs worse, the alias can be moved back to the previous version while a new training run is investigated.

# Part 2: Evidently AI Implementation

## Implementation Overview

The Evidently monitoring implementation is located under the `ml/` directory. It uses the same reference/current structure as Evidently's bicycle-demand monitoring notebook:

1. Load a reference dataset.
2. Load a current dataset.
3. Score both windows with the monitored model.
4. Build Evidently datasets with explicit feature, target, and prediction roles.
5. Generate reusable HTML reports.

The main script is:

```bash
ml/regenerate_evidently_reports.py
```

The command used to regenerate the reports is:

```bash
cd ml
make evidently
```

## Reference and Current Data Windows

For this assignment:

- The reference window is the training split from the Boston crash dataset.
- The current window is a held-out stratified test split.
- Both windows are scored using the production MLflow model.
- The current window serves as a proxy for recent production traffic.

This is appropriate for an academic monitoring assignment because it demonstrates the mechanics of monitoring even though ForeRoute does not yet have live inference logs. In a deployed product, the current window should be replaced with the most recent seven days of logged prediction requests.

## Evidently Dataset Definition

Evidently needs to know which columns are numerical, which columns are categorical, which column is the true target, and which columns contain predictions.

Numerical columns:

- `temperature`
- `precipitation_intensity`
- `wind_speed`
- `visibility`
- `humidity`
- `dew_point`
- `segment_distance_m`
- `lat`
- `lon`
- `hour_of_day`
- `day_of_week`
- `month`
- `prior_year_crash_count`

Categorical columns:

- `precipitation_type`
- `road_type`

Classification fields:

- Target: `label`
- Prediction label: `prediction`
- Prediction probability: `prediction_proba`

This explicit schema is important because monitoring can become misleading if columns are auto-detected incorrectly.

## Generated Report 1: Input Data Drift

The first report is `01_data_drift.html`. It compares the reference and current distributions for all 15 input features.

This report answers:

- Which features drifted?
- Are the drifting features numerical or categorical?
- How large is the shift?
- Is the shift isolated to one feature or broad across the dataset?

For ForeRoute, this report is the first line of defense against covariate shift. It would help identify seasonality, road mix changes, unsupported geography, weather API changes, or stale crash-history features.

## Generated Report 2: Data Quality

The second report is `02_data_quality.html`. It summarizes data quality and descriptive statistics for the model inputs.

This report answers:

- Are columns missing?
- Are data types stable?
- Are null values increasing?
- Are feature ranges plausible?
- Are there duplicate or unusual records?

This report is useful because data quality failures often look like drift. For example, if a weather API returns a default value for all requests, the feature distribution will drift, but the root cause is an upstream integration failure rather than a model issue.

## Generated Report 3: Output Drift

The third report is `03_output_drift.html`. It compares the distribution of `prediction_proba` between the reference and current windows.

This report answers:

- Is the model producing higher or lower probabilities than expected?
- Has the prediction distribution compressed or widened?
- Is the current route mix pushing predictions into a different risk band?

Output drift is especially valuable before labels arrive. It does not prove performance decay, but it provides an early warning that the system is behaving differently from the baseline.

## Generated Report 4: Classification Performance

The fourth report is `04_classification_performance.html`. It compares classification performance across reference and current data.

This report answers:

- Did ROC-AUC or PR-AUC change?
- Did precision or recall change?
- Is the confusion matrix becoming worse?
- Are errors concentrated in a particular class?

This report requires labels. For the assignment, labels are available from the held-out split. In production, this report would be generated after Vision Zero records are backfilled and joined to historical predictions.

## Verification Results

The monitoring pipeline was executed successfully. Four reports were generated under `ml/reports/`.

| Report | Size | Status |
|---|---:|---|
| `01_data_drift.html` | 4.4 MB | Generated |
| `02_data_quality.html` | 3.7 MB | Generated |
| `03_output_drift.html` | 3.6 MB | Generated |
| `04_classification_performance.html` | 3.7 MB | Generated |

The model used for the verified run was `ForeRoute-BostonRisk` v3, MLflow run `94cc207cd66949ba8468c174a99e0ced`.

## Production Model Metrics

The registered production model has the following held-out performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6506 |
| Accuracy | 0.7146 |
| Precision | 0.3887 |
| Recall | 0.2471 |
| F1 | 0.3021 |

The ROC-AUC is modest, but this is expected for crash-history prediction using the available features. Crash risk is affected by important variables that are not included in the current model, including real-time traffic volume, road geometry, driver behavior, signal timing, construction, and temporary closures. Because of this limitation, the web application frames the ML output as a crash-history advisory and pairs it with the deterministic weather score.

# Operational Integration Plan

## Current State

The assignment implementation provides offline monitoring. The reports can be regenerated manually or during a release process. The MLflow registry stores model versions and metrics, and the Evidently reports provide a repeatable view into drift and model quality.

## Production Additions Needed

To make the framework production-grade, ForeRoute should add the following components.

1. **Inference logging.** The `/api/routes` endpoint should write one JSONL row per scored segment after each model call.
2. **Nightly monitoring job.** A scheduled job should load the last seven days of logged predictions and generate the Evidently reports.
3. **Metric extraction.** Key drift and performance metrics should be written to MLflow or a dashboard.
4. **Alerting.** Threshold breaches should send Slack or email alerts.
5. **Backfill evaluation.** New Vision Zero records should be joined to historical predictions to compute real performance drift.
6. **Governance review.** Before promoting a retrained model, the team should review the model card, development metrics, and Evidently reports.

## Example Production Workflow

1. A user requests routes in the web application.
2. The route planner generates segment-level feature rows.
3. The MLflow model returns prediction probabilities.
4. The application stores the request, features, model version, prediction, and UI verdict in a prediction log.
5. A nightly monitoring job reads the latest prediction log window.
6. Evidently compares the current window with the training reference.
7. The system emits reports, summary metrics, and alerts.
8. When new crash labels arrive, the backfill job recomputes classification performance.
9. If drift or performance decay is sustained, the team retrains and evaluates a candidate model.
10. MLflow promotes or rolls back the production alias.

# Limitations

The monitoring framework improves reliability, but it does not remove all risk.

First, Vision Zero crash records are not perfect ground truth. Some crashes are unreported, delayed, or misclassified. Monitoring performance against these records is useful, but it should not be interpreted as a complete measure of real-world road safety.

Second, the current assignment implementation uses a held-out test split as a proxy for production traffic. This demonstrates the monitoring workflow, but true production monitoring requires live inference logging.

Third, drift does not always mean performance decay. Seasonal changes may create real input drift without harming the model. Conversely, performance can degrade due to concept drift even when input drift looks small. This is why the framework monitors features, outputs, and labelled performance together.

Fourth, the model does not include many important causal factors for crashes. Its predictions should remain advisory, and the application should continue to pair the ML signal with the deterministic weather score.

# Conclusion

This report designs and applies a model monitoring framework for the ForeRoute prediction application. The framework uses Evidently AI to monitor input data drift, data quality anomalies, prediction output drift, and classification performance. It follows Chip Huyen's guidance by separating covariate shift, label shift, and concept drift, and by treating observability as an operational requirement rather than a notebook-only activity.

The implemented pipeline successfully generated four Evidently reports for `ForeRoute-BostonRisk` v3. The framework is sufficient for the assignment and provides a clear path toward production monitoring through inference logging, nightly report generation, alerting, label backfills, and MLflow-based model governance.

# Appendix A: Repository Files

| File | Role |
|---|---|
| `REPORT_MONITORING.md` | Markdown version of the assignment report |
| `ForeRoute_Model_Monitoring_Assignment_Report.docx` | Word document version of this report |
| `ml/monitoring_assignment.ipynb` | Evidently notebook for Part 2 |
| `ml/regenerate_evidently_reports.py` | Script that regenerates Evidently HTML reports |
| `ml/reports/01_data_drift.html` | Input drift report |
| `ml/reports/02_data_quality.html` | Data quality report |
| `ml/reports/03_output_drift.html` | Prediction output drift report |
| `ml/reports/04_classification_performance.html` | Classification performance report |
| `MODEL_CARD.md` | Model card for `ForeRoute-BostonRisk` |

# Appendix B: Submission Checklist

- Part 1 explains the monitoring design for data distribution decay, model outcome drift, anomalies, and observability.
- Part 1 maps ForeRoute risks to covariate shift, label shift, and concept drift.
- Part 2 applies the monitoring design using Evidently AI.
- Evidently reports are generated and stored under `ml/reports/`.
- The monitored model and feature schema are documented.
- Production gaps and next steps are clearly identified.
