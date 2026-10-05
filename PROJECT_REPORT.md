# Predictive Maintenance for Industrial Equipment

**Project report**

| | |
|---|---|
| **Author** | Sumanta Biswas |
| **Date** | 5 October 2026 |
| **Task** | Predict machine failure from sensor readings and turn the prediction into a maintenance decision |
| **Dataset** | AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository), 10,000 rows × 14 columns |
| **Deliverables** | `notebooks/ai4i_01.ipynb` (runs top to bottom), `README.md`, this report |
| **Stack** | Python 3.14, pandas 3.0, scikit-learn 1.9, matplotlib, seaborn |

---

## 1. Problem statement

Industrial equipment fails. When it fails without warning, the cost is unplanned downtime, an
emergency repair at premium rates, possible collateral damage to tooling or workpiece, and lost
output. When it is serviced unnecessarily, the cost is an inspection slot and some labour.

These two costs differ by one to two orders of magnitude. That asymmetry — not accuracy — is what
a predictive maintenance model has to be built around.

**The task as implemented:** given a snapshot of sensor readings from a machine (air temperature,
process temperature, rotational speed, torque, tool wear) plus its product quality grade, classify
whether the machine is in a failure state, emit a calibrated failure probability, and convert that
probability into a ranked maintenance worklist with an explicit alert threshold.

**The user of the output:** a maintenance planner who holds a fixed-interval preventive schedule
and a queue of jobs competing for limited maintenance windows. The model does not stop a machine
and is not an alarm system. It changes the **order** of that queue.

---

## 2. Scope: how this differs from the canonical brief

The reference approach for this problem describes a *forecast-window* formulation: collect
timestamped sensor data, build rolling and lag features, define a future failure window, and label
each observation against it. This project implements a *current-state* formulation instead. The
reason is in the data, and it is worth stating plainly rather than hiding.

| Reference step | Status | Reason |
|---|---|---|
| 1. Collect timestamped sensor data | **Not applicable** | AI4I 2020 has no timestamp column. Rows are independent snapshots, not a machine's trajectory through time. |
| 2. Rolling statistics and lag features, avoiding future leakage | **Not applicable** | Rolling and lag features are undefined without time ordering. Deriving them from row index `UDI` would invent an ordering the data does not have. |
| 3. Define a failure prediction window and label each observation | **Not applicable** | The supplied label describes the machine's state *in that row*, not an event within a horizon. Re-labelling against a window would require fabricated time ordering. |
| 4. Train classification models; handle rare failures with sampling/weights | **Implemented** | Logistic regression and random forest, `class_weight='balanced'` against a 3.39% positive rate. |
| 5. Evaluate recall, precision, PR-AUC, false alarm count | **Implemented** | All four reported, plus F2 and the full confusion matrix. |
| 6. Monitoring view with risk scores and high-risk alerts | **Implemented** | Probability-banded risk table and a ranked top-N worklist (§9). |

**Why this was not forced.** A prediction window requires ordered observations per machine. AI4I
supplies neither an entity id that recurs nor a time index. Constructing lag features over `UDI`
would produce features that look legitimate, validate well, and mean nothing — the same class of
error as the target leakage demonstrated in §4. The honest move on a snapshot dataset is to solve
the snapshot problem correctly and document exactly what would change with time-ordered data (§11).

Everything in the pipeline — the column audit, leakage control, the cost-derived metric, off-test
threshold selection, the trivial-alternative benchmark — transfers unchanged to the windowed
formulation. Only the labelling step and the validation scheme change, and §11 specifies how.

---

## 3. Dataset and column audit

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(Matzka, 2020), UCI ML Repository, CC BY 4.0. 10,000 rows, 14 columns, **no missing values** — the
first indication the data is synthetic, since real plant records have gaps.

Every column was assigned a role **before** any modelling, because the audit is what determines
what the model is permitted to see.

| Column | Role | Note |
|---|---|---|
| `UDI` | Identifier | Sequential 1–10,000. No signal; also a proxy for row order. |
| `Product ID` | Identifier | Its **first character is `Type`**, verified for all 10,000 rows. The remainder is a unique serial — memorisation surface only. |
| `Type` | Feature | Product quality grade L / M / H. |
| `Air temperature [K]` | Feature | 295.3–304.5 K |
| `Process temperature [K]` | Feature | 305.7–313.8 K |
| `Rotational speed [rpm]` | Feature | 1,168–2,886 rpm |
| `Torque [Nm]` | Feature | 3.8–76.6 Nm |
| `Tool wear [min]` | Feature | 0–253 min |
| `Machine failure` | **Label** | Target. 339 positives (3.39%). |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | **Label** | Failure-mode flags — *components* of the target. See §4. |

Dropping the two identifiers prevents **memorisation**, not leakage — a distinction that matters,
because a tree given `Product ID` would fit the training set perfectly and generalise nothing.

**Six genuine inputs:** five sensor readings plus the quality grade.

---

## 4. Target leakage, demonstrated rather than asserted

`Machine failure` is very nearly the logical OR of the five failure-mode flags. Those flags are
therefore not features — they are the label, decomposed.

This was verified in code and then exploited deliberately, to show what the leaked score looks
like: a decision tree trained on the five mode flags alone scores **F1 = 0.985** on held-out data.
It has learned `OR(flags)` and nothing about the machine.

At prediction time on a running machine, nobody knows which failure mode is occurring — that is
the thing being predicted. The cell remains in the notebook, labelled as an intentional
demonstration.

**The plant equivalent is subtler and more dangerous**, because the columns will not be named so
conveniently. On real breakdown records, any field populated *because* the failure happened is
leakage: repair cost, downtime hours, spares consumed against the work order, the fault
description, the work-order closing code. Each would yield an excellent validation score and a
model that cannot predict anything in advance.

From §5 onward, only the six audited features are used.

---

## 5. Class imbalance and the baseline

| | Value |
|---|---|
| Failure rate | **3.39%** (339 of 10,000) |
| Failure rate by grade | L 3.92%, M 2.78%, H 2.21% |
| Mode counts | HDF 115, OSF 98, PWF 95, TWF 46, RNF 19 |

The mode flags sum to more than 339 because they are **not mutually exclusive** — one row can be
simultaneously overstrained and outside its power envelope.

**Majority-class baseline** (predict "no failure" for every row):

| Metric | Value |
|---|---|
| Accuracy | **0.9661** |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |

A do-nothing model is 96.6% accurate and catches none of the 339 failures. Every model number in
this report is measured against *this*, not against zero.

The transferable lesson: a vendor reporting **"97% accurate breakdown prediction"** on a problem
with a 3% failure rate may have built nothing at all. The question to ask is always recall, or the
confusion matrix.

**`RNF` is not modellable** and was excluded: it fires on 19 rows, of which only 1 coincides with a
recorded machine failure. It is injected noise with no relationship to the sensors.

---

## 6. Choosing the metric before modelling

Written and committed **before any model was fitted**, so the metric could not be retrofitted to
whichever result happened to look best.

| | Cost |
|---|---|
| **False negative** — missed failure | Unplanned downtime, emergency repair at premium cost, possible collateral damage to tooling, lost output. On a line with no redundancy this is the expensive case. |
| **False positive** — unnecessary inspection | One inspection slot, some labour, at worst a short planned stoppage taken at a time of the planner's choosing. |

A false negative costs **one to two orders of magnitude** more than a false positive. A metric
weighting the two equally — accuracy, or plain F1 — therefore encodes the wrong trade-off.

> **Metric: F2** (recall weighted roughly four times precision). **Threshold: leans toward recall.**
> **Reported alongside: PR-AUC**, which summarises the precision–recall trade-off across *all*
> thresholds and is the right summary statistic under heavy imbalance, where ROC-AUC is optimistic.

---

## 7. Models and validation design

**Split.** Stratified 60 / 20 / 20 — 6,000 fit, 2,000 validation, 2,000 test (203 / 68 / 68
failures). Stratified because at a 3.39% positive rate an unstratified split can land a materially
different number of failures in each part by chance. Fixed `random_state=42` throughout.

**Preprocessing.** `StandardScaler` on the five numeric features; `OneHotEncoder(drop='first')` on
`Type`. One-hot rather than integer-encoded: the grades L/M/H are ordered, but an integer would
additionally impose *equal spacing*, which nothing justifies. Both models share the identical
`ColumnTransformer` so they see identical inputs.

**Model 1 — logistic regression.** First because it is interpretable, fast, and sets a floor: a
linear boundary on six features. It requires the scaling, since rotational speed (~1,500) would
otherwise dominate torque (~40) by sheer magnitude in a penalised objective.

**Model 2 — random forest** (300 trees, balanced class weights). Chosen because the failure
conditions are plausibly *interactions*: high torque matters only at high tool wear; a small
temperature difference matters only at low rotational speed. A linear model cannot represent those
without explicit interaction terms; a tree finds them.

**Rare-event handling.** `class_weight='balanced'` on both models, which reweights the loss by
inverse class frequency. Preferred over SMOTE here because synthetic minority oversampling would
interpolate between failure rows in sensor space, and the generating thresholds (§9) make those
interpolations physically meaningless.

**Validation.** Stratified 5-fold cross-validation on the training portion for the spread of the
estimate; threshold selected on the **validation slice only**; the test set evaluated **once**.

Legitimacy of k-fold here rests on the rows being independent snapshots with no timestamp. On real
time-ordered breakdown records it would be invalid — shuffling would train the model on the future
to predict the past — and rolling-origin (walk-forward) validation would be required.

### Feature importance

| Feature | RF importance | Logistic coefficient |
|---|---|---|
| Torque [Nm] | 0.324 | +2.461 |
| Rotational speed [rpm] | 0.289 | +1.721 |
| Tool wear [min] | 0.191 | +0.767 |
| Air temperature [K] | 0.110 | +1.841 |
| Process temperature [K] | 0.072 | −1.168 |
| Type = L | 0.008 | +0.099 |
| Type = M | 0.006 | −0.085 |

The two rankings disagree informatively. The logistic model spreads weight across torque and both
temperatures linearly; the forest concentrates it on torque, speed and tool wear — consistent with
the interaction story rather than independent additive effects. The quality grade is near-irrelevant
to both.

---

## 8. Results

### Threshold selection

`predict()` silently applies a 0.5 probability cut-off, which is a decision nobody made. Under the
cost asymmetry of §6, 0.5 is the wrong place to stand.

The F2-maximising threshold was located on the precision–recall curve of the **validation slice**:
**0.267**. The test set was then scored once at that threshold.

### Test set — 2,000 rows, 68 failures

| Model | Precision | Recall | F1 | **F2** | PR-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.034 |
| Logistic regression (0.5) | 0.147 | 0.824 | 0.249 | 0.428 | 0.393 |
| Random forest (0.5) | 0.647 | 0.647 | 0.647 | 0.647 | 0.726 |
| **Random forest (tuned, 0.267)** | **0.445** | **0.838** | 0.582 | **0.713** | 0.726 |
| Hand-written rule (§10) | 1.000 | 0.838 | 0.912 | 0.866 | — |

PR-AUC is threshold-independent, so it is identical for a model at any cut-off; the 0.034 baseline
figure is the test positive rate, which is the floor a PR curve cannot fall below in expectation.

**Cross-validated on the training portion (5-fold):**

| Model | F2 | PR-AUC |
|---|---|---|
| Logistic regression | 0.408 ± 0.022 | 0.441 ± 0.030 |
| Random forest | 0.675 ± 0.036 | 0.751 ± 0.041 |

The fold-to-fold standard deviation (0.036) is far smaller than the gap between the two models
(0.267 in F2), so the ranking is established by this data and can be claimed.

### What the threshold change bought

| | Threshold 0.5 | Threshold 0.267 |
|---|---|---|
| Failures caught | 44 of 68 | **57 of 68** |
| Failures missed | 24 | **11** |
| Unnecessary inspections | 24 | 71 |
| Recall | 0.647 | **0.838** |

Confusion matrix at the chosen threshold: **TN 1,861 · FP 71 · FN 11 · TP 57**.

Trading 47 additional inspections for 13 additional caught failures is correct *given* the stated
cost asymmetry, and incorrect if that asymmetry is wrong. The threshold is the single parameter a
planner should be allowed to move, and the argument for its value is economic, not statistical.

### False alarm rate — the number a planner actually feels

| | Value |
|---|---|
| Machines flagged | 128 of 2,000 (6.4%) |
| Of which false alarms | 71 |
| **False alarms per 1,000 machines screened** | **35.5** |
| Worklist precision | 0.445 — roughly 4 in 9 flags are real |

---

## 9. Monitoring view: risk scores and alerts

The deployable artefact is not a label, it is a **ranked risk list**. The model's probability output
is banded so a planner sees triage, not a binary verdict.

| Band | Probability | Machines | Actual failures | Failure rate | Planner action |
|---|---|---|---|---|---|
| **Critical** | ≥ 0.60 | 46 | 35 | **76.1%** | Into the next planned window |
| **High** | 0.27–0.60 | 82 | 22 | 26.8% | Into the next planned window |
| Watch | 0.10–0.27 | 89 | 5 | 5.6% | Stay on preventive schedule; review trend at handover |
| Low | < 0.10 | 1,783 | 6 | 0.3% | No action |

The failure rate rises monotonically across the bands — 0.3% → 5.6% → 26.8% → 76.1% — a 250-fold
spread between the bottom band and the top. **This is the model's real contribution:** as a *ranker*
it separates risk cleanly, even though as a *classifier* at a fixed threshold it loses to the rule
in §10.

Precision down the ranked worklist, which is what a planner actually experiences:

| Worklist depth | Real failures | Precision |
|---|---|---|
| Top 10 | 10 | **1.000** |
| Top 20 | 20 | **1.000** |
| Top 50 | 36 | 0.720 |
| Top 128 (all flagged) | 57 | 0.445 |

**The top 20 machines on the list are all genuine failures.** A planner with capacity for 20
interventions per period wastes none of them. Precision degrades as the list is worked deeper, which
is the correct and expected shape — it tells the planner where to stop given their available
capacity, rather than forcing a single global threshold on them.

**How a planner uses this on a Monday morning.** The list is ordered by probability. *Critical* and
*High* machines enter the next available planned maintenance window ahead of an unflagged machine
whose calendar interval merely happens to be due. *Watch* machines are left on the preventive
schedule but their sensor trend is reviewed at the next shift handover. *Low* requires no action.

The decision the model improves is therefore not "repair or not" — it is **"which of these gets the
window first"**. The confusion matrix sets the expectation honestly: most failures are caught, some
are missed, and a number of healthy machines are pulled in at the cost of an inspection each.

---

## 10. Key finding: the model loses to a hand-written rule

AI4I is synthetic, and its failure modes were generated by deterministic threshold rules on the
sensor columns. Rather than assume this, the claim was tested.

Three rules were read off scatter plots of the mode flags against the relevant sensor quantities —
including mechanical power, which had to be constructed as torque × angular velocity — and then
checked exactly:

| Mode | Rule | Rows reproduced |
|---|---|---|
| HDF — heat dissipation | (process − air temp) < 8.6 K **and** speed < 1,380 rpm | **100.00%** (115/115) |
| PWF — power | power < 3,500 W **or** power > 9,000 W | **100.00%** (95/95) |
| OSF — overstrain | tool wear × torque > grade threshold (L 11,000 / M 12,000 / H 13,000) | **100.00%** (98/98) |
| TWF — tool wear | *none exists* — fires randomly on 46 of 894 rows in the 198–253 min wear window | — |

Their disjunction, written in three lines with **no training and no data**, scored on the same test
set:

| | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| **Hand-written rule** | **1.000** | 0.838 | **0.912** | **0.866** |
| Random forest (tuned) | 0.445 | 0.838 | 0.582 | 0.713 |

**The rule beats the tuned model, with perfect precision and identical recall.** It never raises a
false alarm, because the rules it encodes *are* the rules that generated the labels. Its recall
stops at 0.838 only because tool-wear failure is genuinely random within a wear window and no
deterministic rule can recover randomness.

This reframes everything above it:

1. **The model's score is not evidence of predictive skill.** It measures how well a forest
   approximates three threshold rules from 6,000 samples — and it does not reach them.
2. **A high F1 on AI4I means very little.** Reporting one without this check would be misleading.
3. **The value of the project is the pipeline, not the model**: the audit, the leakage control, the
   cost-derived metric, the off-test threshold, and the benchmark against a trivial alternative.

**Why a rule would not suffice on real data.** Real failures are not generated by three clean
inequalities. Thresholds vary by machine, by tooling, by ambient season and by product; they drift
as equipment ages; and the interactions are neither known in advance nor stable. The reason to fit
a model on real plant data is precisely that nobody can write the rule down — the opposite of the
situation here.

---

## 11. Limitations

Stated unprompted, because they are the reason this report is worth reading.

1. **Synthetic data.** Labels come from deterministic generators (§10), so the achievable ceiling is
   a hand-written rule. No conclusion about real-world performance follows from these numbers.
2. **Current-state classification, not remaining useful life.** Real predictive maintenance usually
   estimates *time to failure*. This classifies an independent snapshot. A planner would rather know
   how long a machine has left than that it is probably failing right now.
3. **No prediction window, no temporal features.** Per §2, the dataset supports neither. The
   windowed formulation in the reference brief remains unimplemented and is the first item of §12.
4. **k-fold would be invalid on real records.** It is legitimate here only because rows are
   independent snapshots. On time-ordered data it leaks the future.
5. **No deployment, drift monitoring, or retraining loop.** The monitoring view of §9 is an offline
   table, not a running service.
6. **Costs are stated as orders of magnitude, not currency.** The threshold is therefore defensible
   in direction but not optimal in value.

**How real breakdown records would differ from this dataset:**

- **No synchronous sensor stream** — retrospective work orders and downtime logs instead, so
  features become equipment history and operating parameters, not live readings.
- **Free-text fault descriptions** written by whoever attended the breakdown, with inconsistent
  terminology and spelling. Turning those into consistent labels is most of the work, and is a
  harder problem than the model.
- **Inconsistent equipment naming** across systems and years, making a per-machine history
  non-trivial to assemble.
- **Missing and late entries** — downtime recorded at shift end from memory, short stoppages
  unlogged entirely.
- **True time ordering**, requiring rolling-origin validation.
- **Far fewer failure events** — hundreds across years, not 339 in 10,000 clean rows.

---

## 12. Next steps

| # | Step | Why |
|---|---|---|
| 1 | **Re-formulate as a prediction window on time-ordered data.** Per-machine sequences, rolling means and standard deviations over trailing windows, lag features computed strictly backwards, label = "fails within the next *H* hours", rolling-origin validation. | Closes §2 steps 1–3 — the gap between this project and the canonical brief. |
| 2 | **Replace synthetic labels with real breakdown records.** The pipeline transfers; the target becomes actual downtime events. | Most of the effort is cleaning free-text faults and reconciling equipment identifiers. |
| 3 | **Move from classification to remaining useful life.** Survival or regression on run-to-failure sequences; NASA C-MAPSS is the public dataset to learn it on. | A horizon is more actionable to a planner than a probability. |
| 4 | **Replace the fixed threshold with expected-cost minimisation.** With real downtime, emergency-repair and inspection costs, the threshold becomes a calculation rather than a choice. | Objective becomes expected cost per machine per period. |
| 5 | **Gradient boosting and calibration.** XGBoost/LightGBM with isotonic calibration, so banded probabilities in §9 mean what they say. | Risk bands are only useful if calibrated. |
| 6 | **Drift monitoring and scheduled retraining.** | A model trained on last year's operating envelope silently decays. |

---

## 13. Reproducibility

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
jupyter lab notebooks/ai4i_01.ipynb
```

Every number in this report is produced by a cell in `notebooks/ai4i_01.ipynb`, which runs top to
bottom from a restarted kernel. All random seeds are fixed at 42.

## 14. References

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C — CC BY 4.0.

Saito, T. & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot
when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10(3).
