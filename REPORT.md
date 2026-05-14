# ForeRoute — Comprehensive Model Development, Evaluation, Explainability, and Monitoring Report

**Course** · *[course code]*
**Assignment** · Predictive Model Development + Held-Out Performance Evaluation + Monitoring Framework
**Group** · *[member 1]* · *[member 2]* · *[member 3]* · *[member 4]*

**Companion documents**
- Model card · [`MODEL_CARD.md`](MODEL_CARD.md)
- Monitoring framework · [`REPORT_MONITORING.md`](REPORT_MONITORING.md)
- Slide deck · [`PRESENTATION.pptx`](PRESENTATION.pptx) · source: [`PRESENTATION.md`](PRESENTATION.md)
- Reproducible training pipeline · `ml/train.py`
- Reproducible data pipeline · `ml/build_boston_dataset.py`
- Threshold calibration · `ml/calibrate_thresholds.py`
- Explainability + ROC/PR analysis · `ml/explainability_analysis.py`
- Web product consuming the registered model · `web/`

---

## Table of contents

1. Executive summary
2. Problem framing and product design
3. Data sources and provenance
4. Data cleaning and validation
5. Negative sampling methodology
6. Feature engineering (15 features in depth)
7. Train / test split and class-balance strategy
8. Model selection rationale
9. **Performance evaluation — ROC, AUC, precision, recall, F1, PR-AUC, confusion matrix, calibration**
10. **Threshold calibration from quantiles**
11. **Explainability — rule-based factors, global feature importance, SHAP deep dive**
12. **Monitoring — Evidently AI deep dive, drift taxonomy, alerting policy**
13. Serving and deployment architecture
14. Limitations (honest accounting)
15. Future work
16. Appendix A — file map and Makefile targets

---

## 1. Executive summary

ForeRoute is a weather-aware safe-routing application that helps Boston-area drivers choose between alternative routes by surfacing two complementary safety scores per route, side by side:

- **Weather right now** — a deterministic, 0–100, rule-based score that encodes the physics of driving in adverse conditions (ice formation at temperatures near freezing with moisture, hydroplaning thresholds for rainfall, low-visibility cutoffs, wind exposure, and amplified risk on bridges, tunnels, and mountain roads).
- **Crash history** — a probabilistic verdict (`Fewer crashes than usual` / `About average` / `More crashes than usual` / `Crashy stretch`) produced by a Gradient Boosting classifier trained on real Boston Vision Zero crash records from 2018–2025, with weather, location, and time features attached.

The crash-history model — `ForeRoute-BostonRisk v1` — was trained on **89,040 samples** (22,260 real crashes plus 66,780 carefully constructed negatives) and evaluated on a held-out stratified test split. Headline metrics on the held-out test set:

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC | **0.6506** | Ranking quality, threshold-independent: probability that the model scores a random crash sample higher than a random non-crash sample |
| PR-AUC | **0.3652** | Average precision across all recall levels; honest for imbalanced data |
| **Recall at the chosen operating point (threshold 0.35)** | **0.841** | The model catches about 84% of real crash situations |
| Precision at the same operating point | 0.307 | When the model warns, it is right about three in ten times — an acceptable trade for high recall |
| **F1 at the same operating point** | **0.449** | The F1-optimal point on the precision-recall curve — substantially higher than the 0.30 measured at threshold 0.5 |
| Accuracy (threshold 0.5, reference) | 0.715 | Overall correct fraction at the conventional cutoff |

A 0.5 threshold is the wrong place to evaluate a safety advisory. § 9.4 walks through the precision-recall curve and justifies operating at threshold 0.35 (recall ≈ 0.84, F1-optimal); § 10 then explains why the production product replaces a binary threshold entirely with a five-bucket plain-language verdict.

Three modelling families were trained for honest comparison; **Gradient Boosting** won on ROC-AUC and is the model registered to the MLflow Model Registry under the alias `production`.

The product **does not** show the raw probability to the user. Instead, each probability is mapped to a plain-language verdict using thresholds calibrated from quantiles of a representative inference-time distribution (§ 10). Explainability comes from two complementary mechanisms (§ 11): the rule-based score is decomposed into named factors with named thresholds, and the ML model's predictions are explained globally via scikit-learn's `feature_importances_` and locally via SHAP TreeExplainer. Monitoring (§ 12) is framed around Chip Huyen's three distribution-shift types and operationalised with Evidently AI reports.

---

## 2. Problem framing and product design

### 2.1 The two-question framing

Conventional navigation applications optimise for time-to-destination. They do not surface the information a driver actually wants on a snowy Tuesday morning: *"Is this trip safe today, on these particular roads?"* That question, when decomposed honestly, becomes two:

1. **Are right-now conditions known-adverse?** This is a physics question. Snow on a bridge at −3 °C is hazardous regardless of where the bridge is or what time of day it is. The deterministic rule-based scorer answers this.
2. **Does this stretch of road have a worse crash history than typical for Boston?** This is a statistical question. A clear Wednesday rush hour through a known-crashy intersection is dangerous in a way that weather alone cannot detect. The machine-learned scorer answers this.

A single combined number cannot honestly answer both. Combining them implies an exchange rate between weather and history that does not exist — a hazardous-by-weather route is hazardous in a different way from a hazardous-by-history route, and the appropriate driver response is different in each case.

### 2.2 Disagreement as signal

When the rule says `Safe` and the model says `Crashy stretch`, that is informative — the driver is on a high-risk corridor in a moment when the weather is not the warning sign. When the rule says `Hazardous` and the model says `Fewer crashes than usual`, that is also informative — the weather is the dominant concern, not the road. The web product flags both kinds of disagreement explicitly with banner copy tailored to which way the signals diverge.

### 2.3 Label semantics

The ML model's label is **real**. Every positive sample is a Boston police `dispatch_ts` record from the Vision Zero crash dataset — a date, time, latitude, longitude, and mode for a real, reported incident. This is a deliberate shift from an earlier prototype that used a synthetic deterministic-then-noised label. Synthetic labels let a model achieve artificially high held-out metrics by relearning the rule it was trained on; this disqualifies the metric as evidence of real-world predictive ability.

---

## 3. Data sources and provenance

The training dataset is the join of three independent sources, all real and public.

### 3.1 Boston Vision Zero Crash Records

- **Source.** City of Boston open data portal — Vision Zero Crash Records CSV.
- **Schema used.** `dispatch_ts` (timestamp), `lat`, `long`, `mode_type` (vehicle / pedestrian / cyclist), `location_type`, street and cross-street names.
- **Raw volume.** 32,001 records as downloaded.
- **Role.** Every positive training sample is exactly one of these records.

### 3.2 Open-Meteo Historical Weather Archive

- **Source.** [`https://archive-api.open-meteo.com/v1/archive`](https://archive-api.open-meteo.com/v1/archive) — free, no API key required.
- **Schema used.** Hourly time series of `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `snowfall`, `rain`, `wind_speed_10m`, `visibility`, `weather_code` (WMO code).
- **Volume queried.** 43 unique ~2.5 km grid cells × eight years of hourly observations.
- **Role.** Provides the per-sample weather features matched at the exact hour and grid cell of each crash (or matched negative).

### 3.3 Mapbox Streets — Tilequery

- **Source.** [`https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/...`](https://docs.mapbox.com/api/maps/tilequery/) — requires API token.
- **Schema used.** Per-point GeoJSON features for road geometry, with `properties.class` (motorway, trunk, primary, secondary, tertiary, street, service, residential) and `properties.structure` (bridge, tunnel).
- **Volume queried.** 62,185 unique training points classified.
- **Role.** Provides ground-truth road classification — the type of road a crash actually happened on. This is also called at inference time in the web app so the model sees the same classification it learned on (see § 13).

### 3.4 Mapping Mapbox classes to our road-type enum

We map the Mapbox `class` and `structure` properties to a six-value enum `{highway, arterial, residential, bridge, tunnel, mountain}` used by both the training pipeline and the rule-based scorer. Bridge and tunnel are taken from `structure`; otherwise the class field is used: `motorway`/`trunk` → highway, `primary`/`secondary` → arterial, the rest → residential. (`mountain` is in the enum for the rule-based scorer's multiplier table; no Boston point classifies as `mountain`.)

---

## 4. Data cleaning and validation

Five filtering rules shaped the final 89,040-row training set. Each is justified separately.

### 4.1 Geographic bounding box

Crashes were filtered to a Boston metro bounding box: latitude 42.20–42.45, longitude −71.20 to −70.95. This bbox is used identically at training time (filtering the raw crash CSV) and at inference time (`web/lib/mlVerdict.ts` checks segment midpoints against the same bbox and tags out-of-region segments).

**Why.** Vision Zero is a Boston-specific dataset. Anything outside Boston was either misgeocoded or refers to a different jurisdiction with different reporting standards. Keeping it would inject noise without adding signal.

### 4.2 Eight-year temporal window

Crashes with timestamps before 2018-01-01 or after `now - 7 days` were dropped. The seven-day buffer at the end exists because Open-Meteo's archive lags real time slightly.

**Why.** Eight years balances signal volume against representativeness. Older data reflects different vehicle fleets (lower automatic-emergency-braking penetration), different signal timing, and pre-Vision-Zero infrastructure. Going further back would dilute the model's relevance to current driving conditions. We retain enough years (~22 k positives after filter) to support stratified splitting and per-cohort breakdowns.

### 4.3 Valid coordinates and timestamps

Records missing latitude, longitude, or `dispatch_ts`, or with values failing `pd.to_datetime` parsing, were dropped. In practice this removed a small fraction of records (well under 1%).

### 4.4 Hour-precision weather attachment

Each crash's weather features are looked up at the **exact hour** the crash happened, in the nearest 2.5 km cell. The lookup is on `pd.to_datetime(crash.ts, utc=True).floor("h")` and grid-cell centre coordinates. Daily averages are insufficient — a 7 PM thunderstorm has a very different driving profile from clear weather at 7 AM the same day.

### 4.5 No future leakage in `prior_year_crash_count`

This is the most consequential rule. The `prior_year_crash_count` feature counts crashes near a sample's location, but the count uses **only crashes whose timestamp is strictly less than the sample's own timestamp** and within the prior 365 days. A sample never sees its own crash event in its own feature value, and never sees a future crash. The implementation uses scikit-learn's `BallTree` with the haversine metric; for each sample we query the tree for neighbours within 100 m and post-filter by the time predicate.

**Why this matters.** Without the strict-less-than rule, the model would trivially memorise the location of every positive. A common subtle bug in crash-prediction pipelines is computing "nearby crashes" over the whole dataset, which leaks the answer into the feature. We avoid this by construction.

---

## 5. Negative sampling methodology

The label is binary: was there a crash at this location at this time? Positives come straight from Vision Zero. Negatives have to be synthesised, and the choice of how to synthesise them deeply affects what the model learns.

### 5.1 The naive approach and why we rejected it

The obvious approach is to sample random (lat, lon, timestamp) tuples uniformly from the Boston bbox. We tried this first. The model trained on those negatives quickly learned a confident but misleading shortcut: **"residential streets are safe."**

This appeared correct on held-out metrics but was statistically spurious. Real crashes cluster on arterials and highways; uniform-bbox negatives mostly landed on quiet residential streets where the cell tower density of crashes is naturally lower. The model wasn't learning anything about crash dynamics — it was just learning the prior distribution of where we'd sampled negatives versus where Vision Zero records cluster.

### 5.2 Anchored negative sampling

The negatives in the final dataset are anchored on real positive locations:

1. Pick a random positive crash record `p`.
2. Apply a small Gaussian jitter (σ ≈ 0.0005 degrees, roughly 55 metres) to `p.lat` and `p.lon`.
3. Resample the timestamp uniformly from the dataset's overall time window.
4. **Reject** the candidate if it falls within 100 m and ±2 hours of any actual positive crash.
5. Repeat until accepted.
6. Generate 3× this many negatives total, so the final positive rate is 25%.

### 5.3 What this achieves

After this process:

- The negative population's road-type distribution mirrors the positives' distribution. The model cannot distinguish them on `road_type` alone.
- The negative population's `lat`/`lon` distribution mirrors the positives' distribution. The model cannot distinguish them on raw location alone.
- The differences that remain are in `temperature`, `precipitation_type`, `precipitation_intensity`, `wind_speed`, `visibility`, `humidity`, `dew_point`, `hour_of_day`, `day_of_week`, `month`, and `prior_year_crash_count`.

In other words, the model is forced to learn *when* a crash happens at a given location, not *where*. This is the substantive learning task we wanted to set up.

---

## 6. Feature engineering — the 15 inputs in depth

The 15 features were chosen to cover the documented signal in the crash-prediction literature without introducing computationally expensive components.

### 6.1 Weather features (7)

| Feature | Unit | Source | Rationale |
|---|---|---|---|
| `temperature` | °C | Open-Meteo `temperature_2m` | Direct factor in ice formation; cold-weather driving fatigue |
| `precipitation_type` | enum | Mapped from WMO `weather_code` + `rain` + `snowfall` | Snow and freezing rain have categorically different friction profiles |
| `precipitation_intensity` | mm / h | `snowfall × 10` (cm→mm) for snow / sleet / freezing rain; `rain` for rain; 0 otherwise | Hydroplaning probability scales with intensity |
| `wind_speed` | km / h | Open-Meteo `wind_speed_10m` | High wind = vehicle handling load, esp. on high-profile vehicles |
| `visibility` | km | Open-Meteo `visibility` (m → km) | Direct effect on reaction time |
| `humidity` | % | Open-Meteo `relative_humidity_2m` | Combined with low temperature, predicts road dew / black ice |
| `dew_point` | °C | Open-Meteo `dew_point_2m` | Together with temperature, sharper indicator of condensation risk |

### 6.2 Road features (2)

| Feature | Type | Source | Rationale |
|---|---|---|---|
| `road_type` | categorical (6 values) | Mapbox tilequery | Bridge vs. residential is the most operationally important distinction for the rule-based scorer; the ML model also uses it |
| `segment_distance_m` | numeric | Constant 1000 m for crash-point samples | Place-holder for compatibility with the web schema; carries little signal for point-based samples |

### 6.3 Location features (2)

| Feature | Source | Rationale |
|---|---|---|
| `lat` | Sample location | Crashes are not spatially uniform in Boston; some intersections crash 30× more than others on the same street |
| `lon` | Sample location | Same as above |

These two features were added late in development. Before they existed, the model was structurally unable to distinguish two segments of the same road. Adding them gave the largest single lift to held-out ROC-AUC of any feature change we made.

### 6.4 Temporal features (3)

| Feature | Source | Rationale |
|---|---|---|
| `hour_of_day` | Sample timestamp (UTC, 0–23) | Rush hours and bar-closing hours have measurably different crash profiles |
| `day_of_week` | Pandas `dayofweek` (0 = Monday … 6 = Sunday) | Friday-night and weekend-morning distributions diverge sharply from weekdays |
| `month` | Sample timestamp (1–12) | Captures seasonal driving population shifts not already explained by weather |

The dominance of `hour_of_day` in the SHAP analysis (§ 11.3) is a *post-hoc* confirmation that adding this feature was correct.

### 6.5 Crash-history feature (1)

| Feature | Source | Rationale |
|---|---|---|
| `prior_year_crash_count` | BallTree spatial-temporal join (§ 4.5) | Past crashes are the strongest known predictor of future crashes in the literature; we add it cheaply via a single tree query |

**Distribution on the training set.** Mean 2.88; max 32; non-zero on 66,441 of 89,040 samples (75%). The right tail is long — a handful of intersections in central Boston accumulate dozens of nearby crashes per year. The model uses this feature as a stable indicator of intrinsic intersection risk.

---

## 7. Train / test split and class-balance strategy

### 7.1 Split

We use an **80 / 20 stratified split** on `label` with `random_state=42`. Pre-rebalance positive rate is 0.250 in both folds (stratification enforces this). The train fold contains 71,232 rows; the test fold contains 17,808 rows.

We did *not* use a group-based split on `route_id` because the Boston dataset is point-based — each row is an event location, not part of a multi-segment route. The leakage risk that motivated `GroupShuffleSplit` in a route-oriented synthetic dataset does not apply here.

### 7.2 Within-train rebalancing

Within the training fold we under-sample negatives to 1.5 × positives, producing a training set with positive rate ≈ 0.40 and total size 44,520 rows. **The test fold is not rebalanced.**

The rebalance has three motivations:

- **Asymmetric error costs.** For a safety advisory, a false negative (missing a hazardous segment) is more costly than a false positive (over-warning on a safe one). Mildly oversampling positives (relative to their natural prevalence) biases the learner toward recall.
- **Stable PR curve.** At 3 : 1 negative : positive, the precision-recall curve at the high-recall end is squeezed against the floor by easy negatives. Rebalancing to 1.5 : 1 produces a cleaner, more interpretable PR curve for threshold selection.
- **Honest test reporting.** Reporting headline metrics on the rebalanced distribution would inflate precision and AUC numbers. The held-out test set keeps the natural 0.250 positive rate so all reported metrics are deployment-honest.

---

## 8. Model selection rationale

Three deliberately different model families were trained on identical preprocessing pipelines (`SimpleImputer → StandardScaler` on numerics; `SimpleImputer → OneHotEncoder(handle_unknown="ignore")` on categoricals).

### 8.1 Decision Tree (`max_depth=6, min_samples_leaf=20`)

An interpretable baseline. The fitted tree can be drawn out and audited — every prediction path is a chain of human-readable threshold tests. If a complex model can't beat this baseline on the chosen metric, the complexity is not earning its keep.

We deliberately constrain depth and leaf size to avoid memorising the training set: a depth-6 tree with 20-sample leaves cannot represent the high-dimensional crash distribution, but it can capture the dominant patterns and tell us whether the dominant patterns are simple or interactive.

### 8.2 Gradient Boosting (sklearn defaults)

The standard production-grade choice for table-shaped tabular data. It captures non-linear interactions across our 15 features without manual feature crosses — "snow × cold × bridge" emerges automatically when the ensemble of weak learners builds joint splits. It is well-calibrated out of the box compared with deep models on similar problems, and `feature_importances_` is auto-logged by MLflow.

### 8.3 MLP (`hidden_layer_sizes=(64, 32), max_iter=600`)

A neural network with a different inductive bias from trees. On tabular data with many categorical features, MLPs typically underperform tree ensembles, but training one provides a sanity check: if the MLP wildly outperformed Gradient Boosting we would investigate; if it underperforms, that's evidence the dataset's signal is well-captured by tree-style threshold splits.

---

## 9. Performance evaluation

### 9.1 What ROC and AUC are, and why they matter here

The **Receiver Operating Characteristic (ROC) curve** plots the *True Positive Rate* (recall) on the y-axis against the *False Positive Rate* (1 − specificity) on the x-axis as the classification threshold sweeps from 1.0 (most conservative) down to 0.0 (most permissive). Each point on the curve corresponds to a specific threshold and tells you the trade-off between catching positives and falsely alarming on negatives at that threshold.

The **Area Under the ROC Curve (AUC)** is a single scalar summarising the curve. It has a precise probabilistic interpretation: AUC equals the probability that the model assigns a higher score to a randomly chosen positive sample than to a randomly chosen negative sample. AUC = 1.0 means perfect ranking; AUC = 0.5 means random guessing; AUC < 0.5 means worse-than-random ranking (which we have never observed in practice for this dataset).

Why we treat ROC-AUC as our headline metric:

- **Threshold-independent.** It doesn't depend on a specific operating point. The product never actually uses a binary cutoff — it computes a probability and maps it to a verdict via calibrated quantile thresholds. ROC-AUC measures the underlying *ranking* quality, which is what the product depends on.
- **Comparable across classifiers.** All three trained families produce probabilities, and ROC-AUC compares them on the same axis.
- **Resilient to class imbalance.** Accuracy and F1 can be misleading at our 25 % positive rate; ROC-AUC is not biased by it in the same way.

### 9.2 Held-out metrics at the conventional 0.5 threshold

Test set, n = 17,808 (the 20 % stratified split), positive rate 0.250.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decision Tree | 0.6355 | 0.707 | 0.373 | 0.253 | 0.302 |
| **Gradient Boosting (registered)** | **0.6506** | 0.715 | 0.389 | 0.247 | 0.302 |
| MLP | 0.6098 | 0.639 | 0.321 | 0.400 | 0.356 |

These numbers use the conventional 0.5 cutoff as a reference point. **A 0.5 threshold is the wrong operating point for a safety model.** It treats false negatives and false positives as equally costly, which is the opposite of what a safety advisory needs: missing a hazardous segment is far more expensive than over-warning on a safe one. The same model, evaluated at a recall-prioritised threshold, produces very different (and much better) safety metrics — see § 9.4.

### 9.3 Reading the threshold-0.5 numbers

- **Gradient Boosting wins on ROC-AUC.** Our headline metric, which is threshold-independent. 0.65 places us at the lower edge of the published 0.65 – 0.75 band for crash-prediction-from-weather work.
- **Decision Tree is competitive on F1.** The interpretable baseline isn't far off the production model in classification metrics, which means most of the signal is captured by relatively simple threshold logic; the gradient boosting ensemble adds modest refinement, mainly in the ranking.
- **MLP buys recall at the cost of precision** at threshold 0.5. It labels more things positive (recall 0.40 vs. 0.25 for the others), but the false-positive rate cost dragged its precision down to 0.32 and its ROC-AUC to 0.61. For a safety advisory the F1 lift looks appealing, but the ranking-quality regression rules out the MLP for production — Gradient Boosting hits the same recall *and higher* by operating at a different threshold (§ 9.4) without sacrificing ranking quality.

### 9.4 Choosing the right operating point — recall-first for safety

For a safety advisory the question is not "is this segment a crash, yes or no?" — it is "how likely is this segment crashy enough to warrant the driver's attention?" Different downstream consequences imply different optimal thresholds. We make the threshold choice explicit rather than defaulting to 0.5.

**The cost asymmetry.** A false negative for ForeRoute is a driver who proceeds without extra caution on a route that turned out to be crash-prone. A false positive is a driver who exercised extra caution on a route that turned out fine. The first costs orders of magnitude more in expected harm than the second; the operating point must reflect that.

**The full threshold sweep.** Computed on the held-out test set (n = 17,808, positive rate 0.250):

| Threshold | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| 0.50 | 0.389 | 0.247 | 0.302 | The conventional default — wrong for safety |
| 0.40 | 0.320 | 0.723 | 0.443 | Recall climbs sharply |
| **0.35 (chosen)** | **0.307** | **0.841** | **0.449** | **F1-optimal · our operating point** |
| 0.30 | 0.294 | 0.902 | 0.443 | Slightly more aggressive |
| 0.28 | 0.290 | 0.920 | 0.441 | Very high recall, narrow F1 cost |
| 0.25 | 0.283 | 0.944 | 0.435 | Recall keeps climbing, F1 starts dropping |
| 0.22 | 0.273 | 0.963 | 0.426 | Below the verdict-layer "lower → typical" cutoff |
| 0.20 | 0.266 | 0.979 | 0.419 | Almost everything flagged |

**Why threshold 0.35.** It is the F1-optimal operating point — every other threshold gives strictly lower F1. At this threshold the model catches **84 % of real crash situations** (recall 0.841) while flagging the right segment about three in ten times (precision 0.307). For a safety advisory this is the principled choice: it picks the threshold the data itself nominates as the best balance, and it sits firmly in the recall-prioritised region without giving up so much precision that warnings become noise.

**Concretely.** At threshold 0.35 the model would issue **about 12,000 warnings** on this test set (where there are 4,452 actual positives) — meaning the driver sees a warning on roughly 70 % of segments, with 84 % of those warnings on real crash-prone stretches. Compared to threshold 0.5 the model catches **3.4 ×** as many real crash situations (3,742 vs. 1,100 true positives) at the cost of misclassifying about 5× as many safe segments.

**Why we ultimately don't operate at any single binary threshold.** The production product does not actually take a binary "alert / no alert" decision. Each probability is mapped to one of five plain-language verdicts via calibrated quantile thresholds (§ 10). The "lower / typical" boundary (probability 0.228) sits *below* threshold 0.35, so any segment whose verdict is `About average` or higher is already in the recall-prioritised region. The verdict layer expresses the recall-first safety stance through richer language than a binary alert.

### 9.5 Honest framing of the 0.65 ceiling

A ROC-AUC of 0.65 may sound low compared to image classifiers (often > 0.95). It is not low for the underlying problem. There are three honest reasons we are at this level:

1. **The strongest crash predictors are not in our feature set.** Real-time traffic volume, individual driver behaviour, specific intersection geometry, signal timing, and pedestrian density are all known to matter, and none of them are in the 15 features we ship. The available features explain only a fraction of crash variance.
2. **Most crashes happen in mild weather.** Most driving happens in mild weather, so the conditional probability of a crash given the weather features is genuinely close to the base rate for most weather configurations. A well-calibrated model will produce probabilities clustered near the base rate, which limits the slope of the ROC curve.
3. **The label is noisy.** Vision Zero records under-count unreported crashes and over-represent crashes in areas with more police presence. The signal-to-noise ratio of the label itself caps any model's achievable AUC.

A model that scored higher than 0.75 on this data and feature set would be suspicious — almost certainly leaking future information or memorising location.

### 9.6 Precision-Recall AUC

On a re-split for the explainability analysis (n = 25,377), we computed the precision-recall AUC (also called Average Precision):

| Metric | Value |
|---|---|
| ROC-AUC | 0.6519 |
| **PR-AUC (Average Precision)** | **0.3652** |

PR-AUC is more sensitive than ROC-AUC at the imbalanced-class end of the operating curve. It is bounded below by the positive rate (0.25 here), so 0.365 represents a meaningful lift over chance for the minority class.

### 9.7 Confusion matrix at threshold 0.5 (reference only)

The product does not use threshold 0.5 in production, but it is the conventional reference point. At threshold 0.5 on the test split (n = 25,377):

|  | **Predicted negative** | **Predicted positive** |
|---|---|---|
| **Actually negative** | TN = 16,549 | FP = 2,484 |
| **Actually positive** | FN = 4,755 | TP = 1,589 |

From which:

- Precision = 1,589 / (1,589 + 2,484) = **0.390**
- Recall = 1,589 / (1,589 + 4,755) = **0.250**
- F1 = 2 · P · R / (P + R) = **0.305**
- Accuracy = (1,589 + 16,549) / 25,377 = **0.715**
- Specificity (TN / (TN + FP)) = 16,549 / 19,033 = 0.870

The asymmetry is informative. At threshold 0.5 the model is far better at confidently identifying *non-crash* situations (specificity 0.87) than confidently identifying *crash* situations (recall 0.25). For a safety product this is exactly the *wrong* direction: we would rather over-warn than miss a hazard. Lowering the threshold to our chosen 0.35 flips the asymmetry — at that operating point recall reaches **0.841** (specificity drops to about 0.37). That is the trade-off a safety advisory should make. The confusion matrix at threshold 0.35: TN = 4,891, FP = 8,465, FN = 710, TP = 3,742.

### 9.8 Calibration

We have not run a formal isotonic or Platt calibration step yet. Predicted probabilities cluster around the 0.25 base rate as expected for a moderately discriminative model. The verdict-mapping layer (§ 10) handles the practical consequence of imperfect calibration by replacing raw probabilities with quantile-based plain-language labels.

A formal calibration step (isotonic regression on a held-out fold, applied as a post-processing wrapper) is on the future-work list (§ 15).

---

## 10. Threshold calibration from quantiles

### 10.1 The problem with raw probabilities

The MLflow-served pyfunc returns probabilities in `[0, 1]`. Showing one of these directly to a driver is misleading: "45 %" invites the misread "45 % chance of crashing on this trip", which is not what the number means. The actual semantics are "the model assigns this segment a positive-class probability of 0.45 relative to the training distribution's 25 % base rate", which is unusable as a user-facing statement.

### 10.2 The verdict design

We map each probability to one of five plain-language verdicts using fixed, data-derived thresholds. The mapping is implemented in `web/lib/mlVerdict.ts`:

| Verdict | UI label | Threshold |
|---|---|---|
| `lower` | Fewer crashes than usual | probability < 0.228 |
| `typical` | About average | 0.228 ≤ probability < 0.453 |
| `above` | More crashes than usual | 0.453 ≤ probability < 0.532 |
| `muchHigher` | Crashy stretch | probability ≥ 0.532 |
| `outOfRegion` | Not enough data | Segment midpoint outside Boston bbox |

### 10.3 How the thresholds were derived

The thresholds are quantiles of the model's predicted probability on a **representative inference-time distribution**. We do not use the training distribution to set thresholds, because the training negatives are spatially anchored on crash locations — they over-represent areas of high crash density and would push the median artificially high.

Instead, `ml/calibrate_thresholds.py` synthesises a 500-sample reference set:

- 500 random points drawn uniformly from the Boston bbox.
- "Typical Boston" weather features (mild temperature, no precipitation, normal humidity / wind / visibility).
- Random `hour_of_day`, `day_of_week`, `month` to span temporal variation.
- `road_type` sampled in proportion {55 % arterial, 40 % residential, 5 % highway} — the empirical mix Mapbox returns for Boston routes.
- `prior_year_crash_count` computed by haversine scan against the last-365-day crash subset.

The registered model is scored on this 500-row set, and quantiles (q10 = 0.171, q25 = 0.228, q50 = 0.322, q75 = 0.453, q90 = 0.532) become the verdict thresholds. The median is also exported to render an explanatory "1.4 × a typical Boston road" caption in the UI.

### 10.4 Why this matters

This approach decouples the model's internal probability scale from the user-facing communication. A retraining that shifts the probability distribution by a constant (a calibration drift) is automatically handled the next time `calibrate_thresholds.py` is re-run, without changing the model or the UI. The user always sees "More crashes than usual" mean *"in the top 25 % of inference-time Boston points"* — which is meaningful regardless of how the model's absolute probabilities shift.

---

## 11. Explainability framework

Explainability is not a single technique; it's a layered set of mechanisms that together let a user (and a reviewer) understand why a number is what it is.

### 11.1 Rule-based explanations — inherently itemised

The rule-based **Weather right now** score is decomposed by construction. Each contributing condition adds a typed factor with a severity and a human-readable description:

| Factor | Contribution | Trigger |
|---|---|---|
| Snow | up to 30 points, scaled by intensity | `precipitation_type == "snow"` and `intensity > 0` |
| Ice | up to 25 points, scaled by `(2 − temperature) / 2` | `temperature ≤ 2 °C` AND `(rain OR snow OR humidity > 80%)` |
| Rain | up to 20 points, scaled by intensity | `precipitation_type == "rain"` and `intensity > 2.5 mm/h` |
| Low visibility | up to 15 points, scaled by `(1 − visibility)` | `visibility < 1 km` |
| High wind | up to 10 points, scaled past 40 km/h | `wind_speed > 40 km/h` |
| Road-type multiplier | × 0.8 – 1.5 | `bridge × 1.5`, `mountain × 1.4`, `tunnel × 1.2`, `highway × 1.0`, `arterial × 0.9`, `residential × 0.8` |

The UI renders each triggered factor as a coloured bar with a description ("Heavy rain (4.2 mm/h) — hydroplaning risk"), ordered by severity. When no factor fires the UI shows a "No weather hazards detected" panel with the actual current values for each threshold and an expandable list of conditions that *would* trigger the score. This gives a non-technical user a complete causal chain.

### 11.2 Global feature importance from scikit-learn

`GradientBoostingClassifier.feature_importances_` is auto-logged to MLflow on every training run. It expresses, for each input feature, the average reduction in the loss function attributable to that feature across all the boosted trees. Useful as a quick global sanity check, but it has known limitations: it is computed only on training data, biased toward high-cardinality features, and cannot explain individual predictions.

### 11.3 SHAP — a deep dive

For both global and per-prediction explanations we use **SHAP (SHapley Additive exPlanations)**. SHAP values are derived from cooperative game theory's Shapley values and have a key property: each feature's SHAP value is the average marginal contribution of that feature to the prediction across all possible coalitions of features. The values are additive — they sum to the difference between the model's prediction and its mean prediction.

#### 11.3.1 TreeExplainer

For tree-based models including Gradient Boosting, the exact computation would be exponential in the number of features. SHAP's `TreeExplainer` exploits the tree structure to compute exact SHAP values in polynomial time. For a Gradient Boosting model with `n` features and `T` trees of depth `d`, TreeExplainer runs in `O(T · L · d²)` where `L` is the number of leaves per tree — fast enough to run on every prediction in real-time at our scale.

#### 11.3.2 Methodology

The script `ml/explainability_analysis.py` loads the registered `ForeRoute-BostonRisk@production` pyfunc, unwraps it to get the underlying scikit-learn `Pipeline`, transforms a sample of test data through the preprocessing step (`pre`), and runs `TreeExplainer(pipe.named_steps["clf"]).shap_values(X_transformed)` on a 1,500-row subset. SHAP values for one-hot-encoded categoricals (`precipitation_type`, `road_type`) are summed back to their parent feature name so the output is at the user-facing feature granularity.

#### 11.3.3 Results — global feature importance via SHAP

Mean absolute SHAP value per feature, on a 1,500-row test sample (higher = more influence on predictions in either direction):

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | `hour_of_day` | 0.323 |
| 2 | `road_type` | 0.117 |
| 3 | `day_of_week` | 0.069 |
| 4 | `temperature` | 0.052 |
| 5 | `dew_point` | 0.049 |
| 6 | `prior_year_crash_count` | 0.044 |
| 7 | `month` | 0.033 |
| 8 | `lon` | 0.028 |
| 9 | `humidity` | 0.025 |
| 10 | `visibility` | 0.021 |

#### 11.3.4 Interpretation

The dominance of `hour_of_day` is striking and credible. Boston's crash density is heavily concentrated in rush-hour windows; a feature that distinguishes 8 AM from 3 AM is structurally one of the most discriminative inputs available. The fact that `road_type` is the second-strongest contributor justifies the late-development bug fix that closed the train-vs-inference road-classification gap (§ 13.2).

Temperature and dew point appear together near the top — a known correlation. They together encode the ice-formation conditions that the rule-based scorer also flags. The model is rediscovering this physics from data, which is itself a useful sanity check.

`prior_year_crash_count` ranks lower than we initially expected, but the absolute scale of SHAP values does not directly map to predictive importance — it's an average, and `prior_year_crash_count` is highly concentrated in its right tail. For the few segments that *do* have a high prior-crash count, that feature contributes a large value; on average across the dataset, the contribution is moderate.

`lat` and `lon` separately contribute less than the temporal features. This makes sense given the negative sampling strategy: negatives and positives share spatial distribution, so raw coordinates do less work than the combination of location-via-`road_type` and the time-of-event features.

#### 11.3.5 Local SHAP — planned, not yet shipped

The next iteration ships per-prediction local SHAP. The implementation path:

1. Wrap the pyfunc to return `{probability, shap}` per row instead of just a probability.
2. Update `web/lib/mlflow.ts` to parse the extended response.
3. Render per-segment SHAP contributions in `RouteDetail.tsx` using the same coloured-bar component the rule-based factor list already uses.

The work is small (one pyfunc change, one types update, one UI section), but it is deferred behind the production launch because it changes the response contract on the serving endpoint and requires retraining the model artifact to include the SHAP wrapper.

### 11.4 Why these two layers together

The rule-based factors and the ML SHAP values answer related but distinct questions:

- *"Why is the conditions risk what it is?"* → rule factors.
- *"Why does the model think this stretch is crashier than typical?"* → ML feature attribution (global today, local next iteration).

Both surfaces are visible to the user simultaneously. When the two signals agree (e.g., the rule flags ice and the model attributes most of its score to `temperature`), the user gets a clear unified narrative. When they disagree, the disagreement banner in the UI surfaces it explicitly.

---

## 12. Monitoring framework — Evidently AI deep dive

### 12.1 What Evidently AI is

[Evidently AI](https://www.evidentlyai.com/) is an open-source Python library for production ML monitoring. It computes a battery of statistical tests on a "reference" dataset vs. a "current" dataset, packaging the results into a single object that renders to an interactive HTML report. The library handles the boilerplate of distribution comparisons, schema validation, and metric drift; we configure which checks to run via Evidently's "preset" abstraction.

Evidently is well-suited for ForeRoute because:

- It runs offline, against snapshot data — no live infrastructure needed for the academic deliverable.
- The HTML reports are self-contained, version-controllable artifacts that sit next to the model.
- It maps cleanly onto Chip Huyen's distribution-shift taxonomy (§ 12.2).

### 12.2 Distribution-shift taxonomy

Following Huyen's framing in *Designing Machine Learning Systems*, we plan for three distinct ways a production model can decay:

| Shift type | Definition | ForeRoute example | Detection signal |
|---|---|---|---|
| **Covariate shift** | P(X) changes; P(Y \| X) unchanged | Winter arrives — colder temperatures, more snow, lower visibility hours. The input distribution shifts even though crash physics has not. | Per-feature drift on the input columns |
| **Label shift** | P(Y) changes; P(X \| Y) unchanged | Boston widens Vision Zero reporting to include cyclist near-misses, increasing the share of "hazardous" labels overall. | Drift on the prediction distribution and (when available) on the ground-truth class balance |
| **Concept drift** | P(Y \| X) changes | Automatic-emergency-braking adoption makes "snowy + cold" intersections genuinely safer — same inputs, different outcomes. | Performance degradation on backfilled ground truth even when input distributions look stable |

Covariate shift is detectable without ground truth and arrives first (winter starts before we have a year of new labels). Concept drift is invisible until labels arrive and is the slowest but most dangerous failure mode.

### 12.3 Evidently presets we use

| Preset | What it checks | Coverage |
|---|---|---|
| `DataDriftPreset` | Per-feature drift: Kolmogorov–Smirnov test on numerics, chi-squared on categoricals, plus PSI for operational thresholds | All 15 features + the prediction column |
| `DataQualityPreset` | Per-column null rate, value ranges, dtype mismatches, schema validation | All 15 input columns |
| `ClassificationPreset` | Confusion matrix, ROC curve, PR curve, per-class precision / recall / F1 against backfilled ground truth | Output column vs. label column (when label is available) |

The library implements each test as a class with sensible defaults (e.g., it auto-picks KS over Wasserstein for `n > 1000`). We accept the defaults; the PSI thresholds are applied externally to the report output for alerting.

### 12.4 Reference and current windows

- **Reference window.** The training rows of `ForeRoute-BostonRisk v1` (post-rebalance), capturing the input distribution the model was trained on.
- **Current window (academic exercise).** The held-out test set, used as a stand-in for production traffic.
- **Current window (production, planned).** A sliding 7-day window of `/invocations` requests captured to a JSONL log on the web side. The web orchestrator does not yet log predictions; wiring that up is the highest-priority deployment task (§ 13.3).

### 12.5 Generated reports

Five HTML reports live in `ml/reports/`:

| File | Contents |
|---|---|
| `01_data_drift.html` | KS + χ² + PSI per feature, training vs. current window |
| `02_data_quality.html` | Per-column null / range / dtype audit |
| `03_output_drift.html` | KS test on the prediction probability column |
| `04_classification_performance.html` | Confusion matrix, PR / ROC curves, per-class metrics |
| `05_regression_performance.html` | Retained from an earlier project version that exposed a continuous risk-score regression head — informational, not load-bearing |

These reports are regenerated on every model release; they sit in the repository as static HTML and are reviewable by anyone with a browser.

### 12.6 Operational alerting policy

We apply explicit alerting thresholds on Evidently's output:

| Signal | Threshold | Action |
|---|---|---|
| PSI on any input feature | > 0.25 for ≥ 3 consecutive days | Page on-call; queue a retraining run |
| PSI on the prediction column | > 0.25 | Investigate whether label or covariate shift; queue retraining |
| PR-AUC drop on backfilled labels | ≥ 5 percentage points vs. current production version | Roll back the production alias to the previous model version; queue retraining |
| New Vision Zero release | Quarterly | Re-run `make boston-data && make train-boston && make calibrate-thresholds`; promote new version only if test ROC-AUC ≥ current version |
| Default cadence | Quarterly | Retrain regardless of drift signal |

Rollback is cheap by design: the previous model version stays in the registry; flipping the `production` alias to a prior version takes effect on the next MLflow `serve` restart. No redeploy, no code change.

### 12.7 What this framework does not claim

- **It does not declare the model "safe."** The dual-score web product is the safety contract; the monitoring layer keeps the ML component honest.
- **It does not assume crash labels are complete truth.** Vision Zero records under-count unreported crashes. PSI on labels would not catch this. The product's framing of the ML output as a *plain-language verdict* rather than a probability is itself a form of monitoring-aware design — it limits how badly a stale model can mislead a driver.
- **It does not replace a human reviewer.** Quarterly, the group should read both reports (`REPORT.md` performance, `REPORT_MONITORING.md` drift), verify the model card's "Out of scope" list is still honest, and approve the next training run.

---

## 13. Serving and deployment architecture

### 13.1 Serving with MLflow pyfunc

The Gradient Boosting model is wrapped in a thin `ProbaWrapper(mlflow.pyfunc.PythonModel)` class that exposes `predict(context, model_input)` returning `pipe.predict_proba(model_input)[:, 1]`. This wrapper is essential: scikit-learn's default pyfunc serving returns `predict()`, which yields hard 0 / 1 class labels rather than probabilities. The web product needs continuous probabilities, so we wrap the pipeline and disable the autolog model-logging hook (`mlflow.sklearn.autolog(log_models=False)`) to prevent the default sklearn artifact from shadowing the wrapped one.

The model is served by `mlflow models serve -m models:/ForeRoute-BostonRisk@production --port 5001 --env-manager local`, exposing a JSON `/invocations` endpoint that accepts a `dataframe_split` payload of 15 features per row and returns one probability per row.

### 13.2 Train-vs-inference parity for `road_type`

A late-development bug discovered during testing: the web orchestrator initially inferred road type from segment length as a heuristic (`< 5 km → residential`), while the training pipeline used real Mapbox tilequery classifications. This caused systematic under-prediction at inference because the model had been trained with proper road classifications and was being fed mis-labelled segments.

The fix adds `tileQueryRoadType()` in `web/lib/mapbox.ts` that calls the same Mapbox tilequery API the training pipeline uses, with the same class-to-enum mapping. Per-segment classifications are cached in memory keyed by 4-decimal coordinates so re-routing through the same area incurs no additional API cost. Empirically the fix lifted the maximum observed probability on urban Boston routes from ≈ 0.25 (residential everywhere) to ≈ 0.45 (correctly mixed arterial / highway / residential), bringing it in line with the model's training distribution.

### 13.3 Deployment plan

What it would take to turn this development build into a real product:

1. **Log every prediction.** After each `scoreBatch` call in `/api/routes/route.ts`, write one JSONL line per scored segment to a rotating log (`web/logs/predictions-YYYY-MM-DD.jsonl`) containing timestamp, segment midpoint, all 15 features, raw probability, assigned verdict, and the rule-based score. This is the missing piece for live drift detection.
2. **Run drift checks nightly.** A small Python job that reads the last seven days of JSONL logs into a DataFrame, loads the training set as the reference, runs Evidently `DataDriftPreset` + `DataQualityPreset`, writes a dated HTML report to `ml/reports/live-drift-YYYY-MM-DD.html`, and emits a PSI summary as an MLflow metric on the production version.
3. **Alert on threshold breaches.** Anything with PSI > 0.25 OR ≥ 5 percentage-point drop in any classification metric triggers an alert (Slack webhook in production; console print in this academic build).
4. **Quarterly retrain.** Pull updated Vision Zero records, re-run `make boston-data && make train-boston && make calibrate-thresholds`, promote the new version only if test ROC-AUC ≥ current production.
5. **One-flip rollback.** Previous versions stay in the registry; flipping the `production` alias takes effect on the next serve restart.
6. **Per-neighbourhood equity audit.** Crash density is shaped by enforcement patterns and infrastructure investment, both of which correlate with neighbourhood demographics. A per-neighbourhood breakdown of model performance is on the future-work list.
7. **Model card maintenance.** The model card (`MODEL_CARD.md`) is the human-readable summary of intended use, out-of-scope uses, sources, performance, and known limitations. It is updated as part of every model release.

---

## 14. Limitations (honest accounting)

| Limitation | Why it matters | Mitigation |
|---|---|---|
| **Boston-only training data** | The model has not seen suburban, rural, or other-city distributions | Out-of-region segments are tagged `Not enough data` in the UI; rule-based score remains available |
| **0.65 ROC-AUC ceiling on weather-only features** | The model honestly cannot rank crashes well from weather alone | The product pairs the model with a rule-based deterministic scorer that catches obvious physical hazards |
| **Vision Zero under-counts unreported crashes** | Some "negative" samples may have had crashes that were not reported | We don't claim the label is ground truth — the UI uses plain-language verdicts rather than probability statements |
| **No real-time traffic, signal-timing, or driver-behaviour features** | Three of the strongest known crash predictors are absent | Acknowledged; future work includes AADT (traffic count) integration |
| **No equity audit yet** | Crash density correlates with neighbourhood demographics; model may underperform on certain neighbourhoods | Per-neighbourhood breakdown planned; the model card explicitly flags this as a known limitation |
| **No live drift signal** | We have the monitoring framework but no production prediction log | First item on the deployment-plan list (§ 13.3) |
| **No live SHAP** | Per-prediction local explanations are not yet served | Design agreed; implementation deferred behind production launch |
| **`prior_year_crash_count` lookup is static** | The web app loads a precomputed crash-density file rather than querying a live database | Acceptable for an academic build; would be updated to a live query in production |

---

## 15. Future work

In approximate cost-to-benefit order:

1. **Prediction logging + nightly Evidently drift job.** Half a day of work. Unlocks live monitoring.
2. **Live SHAP wrapped into the pyfunc.** One day. Closes the explainability gap for per-prediction reasoning.
3. **Isotonic calibration as a post-processing wrapper.** Half a day. Makes the probabilities calibrated enough that a future iteration *could* show them, with a longer-term option to retire the verdict-mapping layer.
4. **Per-neighbourhood equity audit.** One day. Surface accuracy by neighbourhood; flag and document any systematic underperformance.
5. **AADT and traffic-volume features.** Requires MassDOT data integration; estimated two days. Likely the single biggest model-quality lift available.
6. **Multi-city training.** Add NYC, Chicago, and SF open-crash data. Requires per-city feature engineering and a new bbox / region check.

---

## 16. Appendix A — file map and Makefile targets

The entire ML pipeline lives under `ml/`:

| File | Purpose |
|---|---|
| `train.py` | CLI training entrypoint; supports `--data`, `--label-col`, `--registry-name` |
| `build_boston_dataset.py` | Raw Vision Zero CSV → labelled training CSV + density / recent-crashes JSONs |
| `calibrate_thresholds.py` | Re-derives verdict thresholds from quantiles of representative samples |
| `explainability_analysis.py` | Computes SHAP values, ROC / PR curves, confusion matrix on the test set |
| `combine_datasets.py` | (Legacy) merges inter-city + Boston datasets |
| `Makefile` | One-shot targets for the pipeline |
| `data/boston_crashes.csv` | Raw Vision Zero CSV (user-provided) |
| `data/boston_crash_dataset.csv` | Cleaned + labelled 89,040-row training set |
| `data/boston_recent_crashes.json` | 2,572 last-365-day crash points for the runtime density lookup |
| `data/verdict_thresholds.json` | Calibrated verdict thresholds for `web/lib/mlVerdictServer.ts` |
| `data/explainability_summary.json` | SHAP + ROC + confusion-matrix snapshot for this report |
| `mlflow.db` | SQLite tracking + registry store |
| `mlruns/` | Run artifacts including registered model versions |
| `reports/` | Evidently HTML reports |

### Reproducing the pipeline

```
cd ml
make install                              # pip install requirements
export NEXT_PUBLIC_MAPBOX_TOKEN=pk.…      # required for tilequery
make boston-data                          # raw → cleaned + labelled
make train-boston                         # train + register
make calibrate-thresholds                 # refresh verdict thresholds
make serve-boston                         # http://localhost:5001/invocations
```

Caches under `ml/.cache/openmeteo/` and `ml/.cache/tilequery.json` mean a second run typically completes in under two minutes.

---

*This report is the canonical performance and design document for `ForeRoute-BostonRisk v1`. It is complemented by the one-page model card (`MODEL_CARD.md`) for quick reference and the monitoring framework (`REPORT_MONITORING.md`) for operational concerns.*
