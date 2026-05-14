---
marp: true
theme: default
paginate: true
size: 16:9
header: "ForeRoute — Know the road before you go"
footer: "Group Project · 2026"
style: |
  section {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 24px;
    padding: 56px 64px 64px;
    background: #0f172a;
    color: #f8fafc;
    line-height: 1.5;
  }
  h1 { color: #38bdf8; font-weight: 700; font-size: 36px; margin-bottom: 6px; }
  h2 { color: #38bdf8; font-weight: 600; font-size: 26px; margin-top: 0; }
  h3 { color: #94a3b8; font-weight: 500; font-size: 18px; margin-top: 0; text-transform: uppercase; letter-spacing: 0.06em; }
  p, li { color: #e2e8f0; }
  strong { color: #f8fafc; }
  em { color: #7dd3fc; font-style: normal; }
  table { font-size: 18px; color: #e2e8f0; border-collapse: collapse; width: 100%; margin-top: 8px; }
  th { background: #1e293b; color: #f8fafc; text-align: left; padding: 10px 14px; font-weight: 600; }
  td { padding: 10px 14px; border-top: 1px solid #1e293b; }
  blockquote { border-left: 3px solid #38bdf8; padding-left: 16px; color: #94a3b8; font-style: italic; margin: 18px 0; }
  .pill { display:inline-block; padding: 4px 14px; border-radius: 999px;
          background: rgba(56,189,248,0.12); color:#7dd3fc; font-size: 16px; margin: 2px 4px; }
  .good { color: #34d399; }
  .warn { color: #fbbf24; }
  .bad { color: #f87171; }
  .muted { color: #64748b; }
  header, footer { color: #64748b; font-size: 13px; }
---

<!-- _class: lead -->

# ForeRoute
## Know the road before you go

A weather-aware safe-routing app. Two scores per route: one for the **weather right now**, one for the **crash history** of every street you cross.

<span class="pill">Northeastern University</span>
<span class="pill">Group Project · 2026</span>

<br>

*[member 1] · [member 2] · [member 3] · [member 4]*

---

# 1. The idea

Today's navigation apps optimise for **fastest**. None of them ask the question that actually matters on a snowy Tuesday morning:

> "Is this trip safe today, on these roads?"

ForeRoute answers two questions, separately, in plain language:

- **How dangerous is the weather right now?** — handles snow, ice, hydroplaning rain, low visibility, wind, and tricky road types.
- **How crash-prone is this route, based on what's actually happened here before?** — uses real Boston crash records from the last eight years.

We show **both** signals, and we **flag disagreement** between them as the most useful thing a driver can know.

---

# 2. Two scores, side by side

|  | **Weather right now** | **Crash history** |
|---|---|---|
| What it answers | Is right now known-adverse? | Does this road crash more than typical? |
| Built on | Physics of driving (rule-based) | Real Boston crash records (machine learning) |
| Output | Safe · Caution · Risky · Hazardous | Fewer / About average / More crashes than usual / Crashy stretch |
| Loud when | Snow on a bridge, fog, freezing rain | A corridor with a real crash history at rush hour |
| Quiet when | Mild, clear weather | Quiet residential streets at off-peak hours |

> They can disagree on purpose. A snowy bridge at 3 AM Saturday is dangerous by weather but quiet by history. A clear Wednesday rush hour through Dorchester is the opposite.

---

# 3. Where the data comes from

Three real, public sources — no synthetic toy data behind the production model.

| Source | What we use it for | What we got |
|---|---|---|
| **Boston Vision Zero Crash Records** | Every positive training example is a real, dated, police-reported crash | 32,001 raw records → **22,260** after cleaning |
| **Open-Meteo Historical Weather** | The exact weather conditions at every crash's location and hour | 8 years of hourly data per Boston grid cell |
| **Mapbox Streets** | Classifying each location as highway, arterial, residential, bridge, tunnel, mountain | Real road type — not guessed from segment length |

> No survey data, no simulated weather, no toy datasets. Every row in the training set is anchored on a real event in the City of Boston.

---

# 4. How we cleaned the data

Cleaning is where most machine-learning projects either succeed or quietly fail. Five rules we followed:

- **Bounding box.** Only crashes inside the Boston metro made it in. Anything outside got tagged so the model could later say "I don't know about this area."
- **Time window.** Eight years (2018–2025). Older crashes reflect a different vehicle mix, different signal timing, and different road infrastructure.
- **Valid coordinates only.** Records missing latitude, longitude, or timestamp were discarded.
- **Time-of-event matters.** Every crash was paired with the historical weather **at the hour and place it actually happened** — not a daily average.
- **No future leakage.** When we count nearby crashes as a feature, we strictly use crashes that happened **before** the sample's own timestamp.

---

# 5. The negatives problem

Real crashes give us examples of when a crash happened. The model also needs examples of when **nothing happened** — the negatives.

**The naive approach** is to drop random points across Boston. We tried it. The model learned a shortcut: "residential streets = safe."  That isn't really true — it just reflected that crashes cluster on arterials and our random negatives mostly landed on quiet residential roads.

**Our approach:** every negative is anchored on a real crash location, perturbed slightly (about 55 metres), then assigned a random time that is **not** within a few hours of any actual crash there.

Effect: negatives and positives live on the same kinds of roads. The model has to learn *when* a crash happens at a given place, not *where*.

---

# 6. The 15 inputs the model uses

| Group | Inputs | Why |
|---|---|---|
| **Weather (7)** | temperature, precipitation type & intensity, wind, visibility, humidity, dew point | The physics of driving |
| **Road (2)** | road type, segment length | A bridge ≠ a residential street |
| **Location (2)** | latitude, longitude | Two segments of the same road can be very different |
| **Time (3)** | hour of day, day of week, month | Rush hour ≠ 3 AM. Friday ≠ Tuesday. |
| **History (1)** | crashes within 100 m in the prior year | Past crashes are the **strongest** predictor of future ones |

Adding the location, time, and crash-history features lifted accuracy by roughly 5 percentage points over weather and road alone.

---

# 7. The three models we trained

We trained three deliberately different models on the same data so we could compare them fairly.

| Model | Why we chose it |
|---|---|
| **Decision Tree** | A transparent baseline. If our complex model can't beat a simple, explainable tree, the complexity isn't earning its keep. |
| **Gradient Boosting** | The standard production-grade choice for table-shaped data. Captures interactions like "snow × cold × bridge" automatically. |
| **Neural Network (MLP)** | A non-linear contrast to trees, with a different inductive bias. Useful as a sanity check. |

We picked the winner on a held-out test set, using a single agreed-upon metric so we couldn't tune our way into a misleading number.

---

# 8. Which one won — and why

| Model | Ranking quality (ROC-AUC) | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.636 | 0.37 | 0.25 | 0.30 |
| **Gradient Boosting (registered)** | **0.651** | **0.39** | **0.25** | **0.30** |
| Neural Network | 0.610 | 0.32 | 0.40 | 0.36 |

**Gradient Boosting won on ranking quality.** That matters most for this product because we're sorting routes against each other — not making a yes/no "send the police" decision.

**Honest framing.** Published crash-prediction work using weather features sits in the 0.65–0.75 band. We're at the bottom of that band on purpose: most crashes happen in mild weather because most driving happens in mild weather. The model honestly stays near typical instead of pretending to be confident.

---

# 9. What precision and recall mean here

For a safety advisory like ForeRoute these are not just numbers — they translate into real costs:

- **Precision** answers *"when the model says **crash risk is high**, how often is that segment actually crashy?"* High precision means we don't cry wolf.
- **Recall** answers *"of all the genuinely crashy segments out there, how many did we catch?"* High recall means we don't miss the dangerous ones.

For ForeRoute, **missing a hazardous segment costs more than over-warning** about one. That guided our threshold choice toward higher recall, even at the cost of slightly lower precision.

> Our current numbers (precision 0.39, recall 0.25) are typical for a hard problem with noisy labels — and they are honest. The product wraps them in a rule-based score that catches obvious physical hazards regardless.

---

# 10. The system, end to end

How a search travels through the system:

1. The user types two locations. The browser sends them to our server.
2. The server asks **Mapbox** for two or three alternative routes between them.
3. For each road segment, the server asks **OpenWeatherMap** for the current weather and **Mapbox** for the type of road.
4. It calls our **MLflow-served** machine-learning model to get a probability for each segment.
5. It also runs the **rule-based** weather score in parallel.
6. Both scores come back, side by side, with plain-language verdicts.

Everything that flows into the model is tracked in MLflow — every training run, every model version, every metric, every dataset version.

---

# 11. MLflow keeps us honest

The MLflow Model Registry is the contract between the people training the model and the live web app.

- **Tracking.** Every training run logs its parameters, metrics, dataset, and the trained model artifact.
- **Registry.** The best run gets *registered* under a stable name with a `production` alias.
- **Serving.** The web app calls the registered model — not a specific file path. Promoting a new version is a single alias flip.
- **Rollback.** The previous version stays in the registry. If a new model regresses, we flip the alias back.

> A new team member can reproduce any score in the product by loading the dataset version and the model version logged against that score.

---

# 12. Why a score is what it is — explainability

ForeRoute has two complementary explanations, one for each score.

**The weather score is inherently explained.** It decomposes into named factors with named thresholds: "snow contributed 18 points; this road being a bridge multiplied the total by 1.5." The web app shows these factors as colour-bar rows with descriptions.

**The crash-history score uses two layers.**

- *Globally*, the model tells us which inputs it relies on most. The current top contributors are: nearby crash history, location, hour of day, and road type — in that order.
- *Locally* (next iteration), we'll show per-segment contributions for the specific prediction, using the same colour-bar component the weather score already uses.

---

# 13. Keeping the model honest over time — drift

Models silently decay. ForeRoute plans for three distinct ways this can happen:

| Type of drift | What it looks like for ForeRoute | How we catch it |
|---|---|---|
| **The inputs change** | Winter arrives — colder, more snow, less visibility | Statistical tests on every input, against the training window |
| **The proportions change** | Boston's crash reporting widens to include cyclist near-misses | Drift on the model's output distribution |
| **The mapping itself changes** | New safety tech makes snowy intersections genuinely safer | Performance drops on freshly-labelled crash data |

> We use [Evidently AI](https://www.evidentlyai.com/) to produce monitoring reports. They sit next to the model in the repository and re-run on every release.

---

# 14. How we'd deploy this for real

The current build is a working development version. To make this a real product:

- **Log every prediction** — write each segment's inputs, prediction, and verdict to disk for post-hoc analysis.
- **Run drift checks nightly** — compare the last seven days of real traffic against the training distribution. Alert if any input drifts past a known threshold.
- **Pull updated crash records quarterly** — Boston publishes Vision Zero updates quarterly; we'd retrain and only promote the new version if it beats the current one on the same held-out test.
- **Provide a one-flip rollback** — the previous model version stays in the registry. If anything looks wrong, we flip the production alias back.
- **Document everything** — model card, full report, monitoring framework, and a presentation reviewers can read in ten minutes.

---

# 15. Recap

Four numbers to remember.

|  |  |
|---|---|
| **89,040** training rows | 22,260 real crashes + 66,780 carefully-chosen negatives |
| **15 inputs** | Weather, road, location, time, crash history |
| **0.65** ranking quality | Gradient Boosting, registered as the production model |
| **2 scores** | Weather right now + crash history — shown together, disagreement flagged |

> ForeRoute doesn't replace the rule with the model, or the model with the rule. It gives the driver both, in plain language, side by side.

---

<!-- _class: lead -->

# Thank you.

**Live demo.** `http://localhost:3000` — your location auto-fills, then type any Boston destination.

<br>

<span class="pill">MODEL_CARD.md</span>
<span class="pill">REPORT.md</span>
<span class="pill">REPORT_MONITORING.md</span>

<br>

### Questions?
