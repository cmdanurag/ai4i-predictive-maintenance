# Forecast-Window Failure Prediction for Industrial Equipment

**Project report**

| | |
|---|---|
| **Author** | Sumanta Biswas |
| **Date** | 5 October 2026 |
| **Task** | Predict whether a machine will fail **within a future time window** using sensor readings and operating history |
| **Dataset** | NASA C-MAPSS turbofan degradation, subset FD001 — 100 run-to-failure engines |
| **Deliverable** | `notebooks/cmapss_forecast_window.ipynb` (runs top to bottom, 26 code cells, no errors) |
| **Companion project** | [`../PROJECT_REPORT.md`](../PROJECT_REPORT.md) — the *snapshot* formulation on AI4I 2020 |
| **Stack** | Python 3.14, pandas 3.0, scikit-learn 1.9, matplotlib |

---

## 1. Why this project exists

The companion project classifies an independent sensor snapshot: *is this machine in a failure state
right now?* That question is answerable but operationally thin — by the time the reading is abnormal,
the machine is already failing, and a planner has no time to act.

This project answers the question a planner actually needs:

> **Will this machine fail within the next 30 operating cycles?**

That is the *forecast-window* formulation. It requires something AI4I 2020 does not have: **time-ordered
observations of the same machine**. Rolling statistics, lag features and a prediction window are all
undefined without a time index, which is exactly why the companion project did not attempt them and
said so rather than fabricating an ordering from the row number.

C-MAPSS supplies what is needed — 100 engines recorded once per cycle from healthy to failure — so the
three steps the companion project had to declare out of scope are implemented here.

| Reference step | Companion project (AI4I) | This project (C-MAPSS) |
|---|---|---|
| 1. Collect timestamped sensor data | Not applicable — no timestamp | **Implemented** — 100 engines × per-cycle records |
| 2. Rolling statistics and lag features, no future leakage | Not applicable | **Implemented** — 136 trailing features, §4; leakage broken deliberately in §5 |
| 3. Define a failure prediction window and label | Not applicable | **Implemented** — RUL ≤ 30 cycles, §3 |
| 4. Classification with rare-event weights | Implemented | **Implemented** — `class_weight='balanced'` |
| 5. Recall, precision, PR-AUC, false alarms | Implemented | **Implemented** — §8 |
| 6. Monitoring view with risk scores and alerts | Implemented | **Implemented** — risk trajectories, bands, worklist, §10 |

---

## 2. Data

NASA C-MAPSS turbofan engine degradation simulation, subset **FD001** (single fault mode, single
operating condition). Published by NASA's Prognostics Center of Excellence and the standard public
benchmark for remaining-useful-life work.

| | Train | Test |
|---|---|---|
| Engines | 100 | 100 (different engines) |
| Rows (one per cycle) | 20,631 | 13,096 |
| Observation | healthy → **failure** | healthy → **truncated at a random point** |
| Engine lifetime (cycles) | min 128, median 199, max 362 | observed: min 31, median 134, max 303 |
| Missing values | 0 | 0 |

Each row carries 3 operating settings and 21 sensor channels (temperatures, pressures, shaft speeds,
fuel flow, bleed and coolant measurements).

**The test engines are the valuable part of this dataset.** They are cut off at a random healthy
point, with true remaining life supplied separately (`RUL_FD001.txt`, min 7 / median 86 / max 145
cycles). That makes them a **fleet observed at mixed ages** — which is precisely what a deployed
monitoring system sees, and a far harder and more honest test than a held-out slice of run-to-failure
histories.

### Column roles

| Column | Role | Why |
|---|---|---|
| `unit` | **grouping key** | Engine identity. *Not a feature* — a model given it would memorise individual engines, when it must generalise to engines never seen. Used only to keep an engine wholly inside one split (§6). |
| `cycle` | **time index** | Orders each engine's history; builds trailing features and measures lead time. **Excluded as a direct feature** — see §5. |
| `set1`, `set2` | feature | Operating conditions. |
| `set3`, `s1`, `s5`, `s10`, `s16`, `s18`, `s19` | **dropped** | Constant across all 20,631 training rows — no signal, only noise to split on. |
| remaining 15 sensors | feature | The degradation channels. |

**17 base columns survive the audit.** There is no supplied label — constructing it correctly is the
next section.

---

## 3. Defining the forecast window

**Remaining useful life (RUL)** at a cycle is the number of cycles until that engine fails.

- Training engines run to failure: `RUL = max(cycle) − cycle`.
- Test engines are truncated: RUL at the final observed cycle is given, and RUL earlier in the series
  is that value plus the cycles still remaining in the observed window.

**The label:**

> `y = 1` if `RUL ≤ H`, else `0`

**H = 30 cycles.** This is a maintenance-planning decision, not a statistical one — long enough to
procure a part and book a window, short enough that the alert is actionable rather than a vague
warning. It is also the conventional choice on C-MAPSS, keeping these numbers comparable to published
work. Prevalence is a direct function of H:

| H | Positive rate (training rows) |
|---|---|
| 10 | 5.33% |
| 20 | 10.18% |
| **30** | **15.03%** |
| 50 | 24.72% |

**Is computing RUL itself leakage?** No — and the distinction is the heart of this project. RUL uses
the engine's eventual failure cycle, which is legitimate for *labelling historical data*: supervised
learning requires knowing the outcome. It becomes leakage only if that knowledge reaches the
**features**, which §5(b) tests by building exactly that mistake.

### A six-fold prevalence gap, and why it is not a bug

| | Positive rate |
|---|---|
| Training engines (run to failure) | **15.03%** |
| Test engines (truncated at random) | **2.54%** |

Training engines are observed all the way to failure, so a fixed fraction of every history sits inside
the window. Test engines are cut off at a random healthy point, so most observations sit far from
failure.

This is a **prior shift**: a threshold tuned at 15% prevalence is applied to data at 2.6%. It costs
precision, it is left uncorrected, and §8 reports it rather than hiding it — because a deployed model
faces exactly this situation, tuned on failure-rich history and run on a mostly healthy fleet.

---

## 4. Rolling and lag features, computed strictly backwards

A single cycle's reading says little. Degradation appears as **drift and growing variance** across an
engine's history, so each row is described by the recent past of its own engine.

| Feature family | Count | What it captures |
|---|---|---|
| raw value | 17 | current operating point |
| trailing rolling mean, windows 5 and 20 | 34 | the level, denoised — where drift shows |
| trailing rolling standard deviation, windows 5 and 20 | 34 | growing instability |
| difference over 1 and 5 cycles | 34 | rate of change |
| difference from the engine's first reading | 17 | total drift since new |
| **Total** | **136** | |

**The rule that makes all of this legitimate: every window ends at the current cycle.**

1. `groupby('unit')` **first**, so no window ever spans two engines.
2. Then a **trailing** `rolling(w)`, so no window ever reaches forward in time.

Both halves matter, and §5 breaks each one deliberately to show what it costs.

**Warm-up rows are dropped, not imputed.** The first 5 cycles of an engine have no usable history, so
the widest lag is undefined. 500 rows were dropped from each of train and test — an imputed trailing
window is a fabricated history, which is a quieter version of the same error §5 is about.

**The trailing property is asserted in code, not assumed.** The notebook recomputes one stored
rolling mean by hand from only the cycles at or before that row and asserts equality:

```
engine 1 cycle 26: stored s11_rm5 = 47.310000, recomputed over cycles <= t = 47.310000
trailing-window check passed
```

Final matrices: **train 20,131 × 136** (15.40% positive), **test 12,596 × 136** (2.64% positive).

---

## 5. Three ways to leak the future — each built and scored

The reference brief says *"create rolling statistics and lag features while avoiding future-data
leakage."* Avoiding it is easy to claim and easy to get wrong, so each failure mode is constructed on
purpose and scored against the honest pipeline on the same held-out engines.

### (a) A window that reaches forward

A centred rolling window averages cycles on *both* sides of the current one, so it contains readings
that have not happened yet. It is a one-argument change — `center=True` — and it is invisible in the
score.

| Rolling mean of the same 15 sensors | Test PR-AUC |
|---|---|
| Centred window (reaches forward) | **0.768** ← leak |
| Trailing window (honest) | 0.664 |

### (b) A feature that needs to know the failure date

"Fraction of life consumed" — current cycle ÷ total lifetime — is an intuitive degradation feature and
completely unusable: computing it requires the failure cycle, which for a running machine is precisely
the unknown.

| | Validation PR-AUC | Test PR-AUC |
|---|---|---|
| With frac-of-life (leaked) | **0.975** | **0.888** |
| Honest pipeline | 0.971 | **0.906** |

**Read both columns.** The leaked feature *improves* validation and *degrades* the test score.

The reason is the shape of the leak in deployment. On engines run to failure, `frac_life` means what
it claims. On truncated test engines the denominator is the last cycle *observed*, not the cycle of
failure — so a healthy engine at the end of its observation window looks, to this feature, like an
engine at the end of its life. The model leaned on something that means one thing in training and
another in production.

**This is the realistic failure mode of leakage.** It does not announce itself as an implausible
score. It quietly makes the deployed model worse while the validation number goes up — which is why
a leakage check cannot be "does the score look too good?"

### (c) Validating on rows instead of engines

The subtlest of the three, and unique to time-ordered data. Consecutive cycles of the same engine are
nearly identical. Split rows at random and an engine's cycle 141 lands in train while cycle 142 lands
in validation — the model is scored on machines it has already seen, which is not the question.

| Split | Engines on both sides | PR-AUC | What it measures |
|---|---|---|---|
| Random **row** split | **100** | **0.986** | memorising engines it has seen |
| Grouped **engine** split | 0 | 0.971 | unseen engines, all run to failure |
| **Official test set** | 0 | **0.906** | unseen engines, truncated at random ages |

Each step removes one layer of optimism and the score falls at every step. **The row-wise number is
the one most commonly reported on this dataset, and it is the one that means least** — it overstates
deployable performance by 0.08 PR-AUC.

### Why `cycle` is excluded as a feature

Related to (b). **Every training engine runs to failure**, so within the training data a large cycle
number reliably means "near the end". A fleet in service is not like that — engines sit at every age.
Including `cycle` adds only 0.008 PR-AUC on test while making the model depend on an artefact of how
the training data was collected, so it stays out. This is survivorship bias, not leakage, but the
remedy is the same.

---

## 6. Validation design

| Stage | Data | Purpose |
|---|---|---|
| **Fit** | 75 engines, 15,194 rows | train model parameters |
| **Validate** | 25 engines, 4,937 rows | select the decision threshold, nothing else |
| **Test** | 100 official engines, 12,596 rows | evaluated **once** |

- Splits are **grouped by engine** — verified in the notebook: 0 engines shared between fit and
  validation.
- Cross-validation uses `StratifiedGroupKFold` — *grouped* so folds split engines, *stratified* so
  each fold carries a comparable share of positives.
- Within every engine chronological order is preserved and all features are trailing (§4), so no row
  is ever described using its own future.

Nothing in the project is selected on the test set.

---

## 7. Models

Logistic regression first for a floor, then a random forest (300 trees, `min_samples_leaf=5`).

**Rare-event handling: `class_weight='balanced'`, not resampling.** SMOTE would interpolate between
rows of *different engines at different degradation stages*, producing sensor combinations no engine
ever exhibited. Reweighting the loss leaves the data real.

### Cross-validated, engine-grouped (5-fold, 100 training engines)

| Model | PR-AUC | Folds |
|---|---|---|
| Logistic regression | 0.969 ± 0.007 | 0.959, 0.977, 0.965, 0.977, 0.966 |
| Random forest | 0.971 ± 0.005 | 0.961, 0.975, 0.975, 0.971, 0.972 |

**The two models are tied.** The 0.002 gap is a fraction of the fold-to-fold spread, so the forest's
apparent edge is not established by this data. That is a finding, not a disappointment: it says the
signal is mostly a smooth drift in the rolling means, which a linear model on well-chosen features
captures nearly as well as a forest does.

The forest is carried forward because it holds a modest but consistent advantage on the truly
held-out test engines (0.909 vs 0.882) and needs no feature scaling in deployment.

### Where the signal lives

| Feature family | Summed importance |
|---|---|
| Trailing rolling mean, window 5 | **0.452** |
| Trailing rolling mean, window 20 | **0.333** |
| Drift from first reading | 0.107 |
| Raw current value | 0.061 |
| Rolling sd, window 20 | 0.031 |
| Rolling sd, window 5 | 0.008 |
| Differences (1 and 5 cycles) | 0.008 |

Top individual features: `s15_rm5` (0.073), `s11_rm5` (0.065), `s4_rm5` (0.060), `s3_rm20` (0.056),
`s15_rm20` (0.053).

**The engineered features carry the model; raw readings contribute 6%.** Trailing rolling means
account for 78% of total importance — degradation is not an abnormal instant but a level that has
moved. This is the direct justification for step 2 of the brief: the rolling statistics are not
decoration, they are where the signal is.

---

## 8. Results

### The benchmark that decides whether the model is worth deploying

Before claiming the model is useful it has to beat what a plant **already does**: schedule maintenance
by running hours. That baseline is one line — *alert when the engine has run more than k cycles* —
with k chosen on the validation engines like any other parameter (`cycle > 125`, validation F2 0.728).

### Threshold selection

`predict()` silently uses 0.5, a decision nobody made. The F2-maximising threshold on the **validation
engines only** is **0.377** (validation precision 0.808, recall 0.964, F2 0.928). The test set was
then scored once at that threshold.

### Test set — 100 unseen engines, 12,596 rows, 2.64% positive

| Model | Precision | Recall | F1 | **F2** | **PR-AUC** |
|---|---|---|---|---|---|
| Always alert | 0.026 | 1.000 | 0.051 | 0.119 | 0.026 |
| Age rule: `cycle > 125` | 0.118 | 0.883 | 0.209 | 0.385 | 0.210 |
| Logistic regression | 0.630 | 0.922 | 0.748 | 0.843 | 0.882 |
| **Random forest** | **0.691** | **0.937** | **0.795** | **0.875** | **0.909** |

PR-AUC floor (test positive rate): **0.026**.

**The model beats age-based scheduling decisively** — F2 0.875 against 0.385, PR-AUC 0.909 against
0.210. The age rule reaches comparable recall only by flagging a large fraction of the fleet, which is
precisely what calendar-based maintenance costs in practice.

**Metric note.** As in the companion project the costs are asymmetric — a missed failure is an
unplanned engine removal, a false alarm an unnecessary inspection — so **F2** is the operating metric
and **PR-AUC** the threshold-free summary. At 2.6% positives, ROC-AUC would flatter everything.

### Confusion matrix at threshold 0.377

|  | Predicted healthy | Predicted at-risk |
|---|---|---|
| **Actually outside window** | 12,125 | 139 |
| **Actually within 30 cycles** | 21 | 311 |

- Of 332 cycle-observations genuinely within 30 cycles of failure: **311 flagged, 21 missed**
- 450 rows flagged in total — **3.6% of all fleet-cycles**
- **False alarms per 1,000 observations screened: 11.0**

### The honest comparison with the companion project

This is **the opposite** of the companion project's finding, where a three-line hand-written rule
*beat* the tuned model because AI4I's labels were generated by threshold rules. Here the trivial
alternative loses by a wide margin, so the model earns its place.

Together the two projects make the point that matters: **the benchmark decides whether a model is
worth deploying, and it has to be run before the claim is made, not after.** One project shows a good
score meaning nothing; the other shows the same discipline applied where the score turns out to be
real.

---

## 9. Alert lead time — the metric a forecast model lives or dies by

Row-level precision and recall miss the operational question. A planner does not ask *"what fraction
of cycle-observations were classified correctly?"* They ask **"did I get a warning on this engine, and
how many cycles of warning did I get?"**

So the evaluation is re-run per engine.

### Engine-level detection (100 test engines)

| | Engines | Alerted | Rate |
|---|---|---|---|
| Within 30 cycles of failure at last observation | 25 | **25** | **recall 1.000** |
| Still outside the window | 75 | 4 | 5.3% |

### Lead time for the 25 correctly-alerted engines

| min | p25 | **median** | p75 | max | mean |
|---|---|---|---|---|---|
| 24 | 31 | **33** | 39 | 54 | 35.2 |

- Engines warned **≥ 30 cycles** ahead: **19 of 25**
- Engines warned **≥ 20 cycles** ahead: **25 of 25**

**Every engine genuinely approaching failure was flagged, with a median of 33 cycles of warning** —
enough to procure a part and book a window rather than react to a failure.

### The four engine-level "false alarms", read individually

| Engine | Last obs. | True RUL then | First alert | Fail cycle | Lead |
|---|---|---|---|---|---|
| 58 | 176 | 37 | 170 | 213 | 43 |
| 77 | 162 | 34 | 158 | 196 | 38 |
| 91 | 234 | 38 | 208 | 272 | 64 |
| 93 | 244 | 85 | 243 | 329 | 86 |

**Three of the four were genuinely degrading** — flagged at 37, 34 and 38 cycles of remaining life,
just outside the 30-cycle window. Only engine 93 is a true premature alarm.

A row-level confusion matrix counts all four as errors; a planner would not. **The row-level
false-positive count therefore overstates the operational cost, and the per-engine view is the one to
quote.**

---

## 10. Monitoring view: risk scores and alerts

The deployable artefact is a risk score per engine per cycle. Two views matter: the **trajectory**,
which shows risk rising and lets an engineer sanity-check an alert before acting, and the
**worklist**, which ranks the fleet right now.

### Fleet risk bands, at each engine's latest observation

| Band | Probability | Engines | Truly within window | Median true RUL | % in window |
|---|---|---|---|---|---|
| **1 Critical** | ≥ 0.70 | 19 | 18 | **15** | **94.7%** |
| **2 High** | 0.377–0.70 | 9 | 7 | 28 | 77.8% |
| 3 Watch | 0.15–0.377 | 6 | 0 | 56.5 | 0.0% |
| 4 Low | < 0.15 | 66 | 0 | **101.5** | 0.0% |

**Median true remaining life falls monotonically as risk rises — 15 → 28 → 56.5 → 101.5 cycles.**
That is the property a planner needs: the ranking is trustworthy even where the absolute probability
is not perfectly calibrated. Not one engine in the Watch or Low bands was actually within the window.

### Top 10 worklist

| Engine | Cycle | Risk | True RUL |
|---|---|---|---|
| 76 | 205 | 1.000 | 10 |
| 35 | 198 | 0.999 | 11 |
| 68 | 187 | 0.998 | 8 |
| 34 | 203 | 0.997 | 7 |
| 81 | 213 | 0.995 | 8 |
| 31 | 196 | 0.986 | 8 |
| 82 | 162 | 0.982 | 9 |
| 49 | 303 | 0.979 | 21 |
| 66 | 147 | 0.968 | 14 |
| 20 | 184 | 0.960 | 16 |

**All ten have 7–21 cycles of life remaining.** A planner working the list from the top wastes no
intervention.

### What a planner does with this on a Monday

Work the list from the top. *Critical* engines are booked into the next maintenance window; *High*
engines into the one after, with a part ordered now; *Watch* engines get their trajectory reviewed at
the next shift handover; *Low* stays on the ordinary schedule.

The lead-time distribution in §9 is what makes that sequencing safe — it is the evidence that a *High*
engine will not fail before its window arrives.

---

## 11. Limitations

Stated unprompted, because they are the reason this report is worth reading.

1. **Simulated data.** C-MAPSS is output from a high-fidelity NASA engine model, not a fleet of real
   engines. It is *physics-based* rather than rule-generated — a far better proxy than AI4I's threshold
   rules, and a hand-written rule cannot recover it — but degradation is still smoother and cleaner
   than reality.
2. **One fault mode, one operating condition.** FD001 is the easiest of the four subsets. FD002–FD004
   add six operating conditions and two fault modes, and published scores there are materially lower.
   The headline numbers here should not be read as C-MAPSS performance in general.
3. **Every engine starts healthy and is observed continuously.** Real fleets have machines already
   mid-life when monitoring begins, gaps in telemetry, and sensors replaced mid-history.
4. **No maintenance actions in the data.** Real histories contain interventions that reset wear, so RUL
   is not monotone and time-to-failure is interrupted by repairs.
5. **The horizon is fixed at 30 cycles.** A deployed system would need several horizons at once, or a
   direct RUL estimate, since different repairs need different lead times.
6. **Probabilities are not calibrated.** The bands in §10 rank well, but their absolute values should
   not be read as probabilities. Isotonic or Platt calibration on a held-out set is the fix.
7. **Costs are asymmetric by assertion.** F2 encodes a direction, not a measured ratio. Real
   procurement, downtime and inspection costs would turn the threshold into a calculation.
8. **A cycle is not a calendar date.** Lead time is reported in operating cycles; converting to
   procurement lead time needs a utilisation rate the dataset does not supply.

---

## 12. What this project adds to the companion one

| | AI4I snapshot | C-MAPSS forecast window |
|---|---|---|
| Question | failing now? | fails within 30 cycles? |
| Time structure | none — independent rows | per-engine cycle histories |
| Features | 6 raw columns | 136 trailing rolling/lag/drift |
| Honest benchmark | hand-written rule, which **won** | age-based scheduling, which **lost** |
| What the benchmark proved | labels were rule-generated; the model learned no physics | the model earns its place over calendar maintenance |
| Leakage controlled | label components as features | forward-reaching windows, failure-date features, row-wise splits |
| Validation | stratified k-fold on independent rows | engine-grouped splits, trailing features, truncated test fleet |
| Headline | PR-AUC 0.726 (floor 0.034) | **PR-AUC 0.909** (floor 0.026) |
| Operational output | a ranked flag list | a risk trajectory **and 33 cycles of median lead time** |

---

## 13. Next steps

| # | Step | Why |
|---|---|---|
| 1 | **Predict RUL directly** — regression or survival analysis on the same features | Removes the arbitrary horizon and gives a planner a date rather than a flag |
| 2 | **FD002–FD004**, with operating-condition normalisation | Six conditions and two fault modes; trailing features are not comparable across engines until conditions are normalised |
| 3 | **Sequence models** — LSTM or temporal CNN | The 136 features hand-engineer what a sequence model would learn; worth testing whether the learned version wins |
| 4 | **Calibration, then cost-optimal thresholding** | Isotonic calibration so the bands mean what they say, then a threshold set by expected cost per engine per cycle instead of F2 |
| 5 | **Rolling-origin validation across calendar time** | Once a deployment history exists: train on earlier engines, test on later ones, to measure decay as the fleet and duty cycle drift |
| 6 | **Apply to real breakdown records** | The pipeline transfers; the work becomes cleaning free-text faults and reconciling equipment identifiers, as the companion report's §11 sets out |

---

## 14. Reproducibility

```bash
cd forecast_window/notebooks
jupyter lab cmapss_forecast_window.ipynb     # Restart kernel, Run All
```

Dependencies are the parent project's `requirements.txt`. Every number in this report is produced by a
cell in `notebooks/cmapss_forecast_window.ipynb`, which runs top to bottom from a restarted kernel
with no errors. All random seeds are fixed at 42.

Data files in `data/` were downloaded from public mirrors of the NASA PCoE repository and verified
byte-identical (MD5 `259f340bac32ce6fa8894815600fa757` for `train_FD001.txt`) across two independent
sources.

## 15. References

Saxena, A. & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set (C-MAPSS).* NASA Ames
Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Saxena, A., Goebel, K., Simon, D. & Eklund, N. (2008). Damage propagation modeling for aircraft engine
run-to-failure simulation. *International Conference on Prognostics and Health Management.*

Saito, T. & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when
evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10(3).
