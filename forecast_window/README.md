# Forecast-Window Failure Prediction on NASA C-MAPSS

**Will this engine fail within the next 30 operating cycles?**

The *forecast-window* formulation of predictive maintenance: time-ordered sensor histories per
machine, features built from strictly trailing windows, and a label defined over a future horizon.

Companion to the [parent project](../README.md), which solves the *snapshot* problem ("is this
machine failing right now?") on AI4I 2020. That dataset has no timestamps, so rolling features, lag
features and a prediction window are undefined on it. This project uses run-to-failure data where
they are defined.

**Notebook:** [`notebooks/cmapss_forecast_window.ipynb`](notebooks/cmapss_forecast_window.ipynb)
**Full write-up:** [`PROJECT_REPORT.md`](PROJECT_REPORT.md)

## Data

NASA C-MAPSS turbofan degradation, subset **FD001** — 100 engines run from healthy to failure, 21
sensors plus 3 operating settings per cycle. 20,631 training rows; the 100 **test** engines are
truncated at random healthy points, making them a fleet at mixed ages (13,096 rows).

## Method

1. **Audit.** 7 constant columns dropped; `unit` kept as a grouping key only, never a feature.
2. **Label.** `y = 1` if remaining useful life ≤ 30 cycles. H is a planning decision: long enough to
   procure a part, short enough to be actionable.
3. **Features.** 136 columns — trailing rolling means and standard deviations (windows 5 and 20),
   1- and 5-cycle differences, drift from each engine's first reading. `groupby` first so no window
   crosses an engine, trailing only so none reaches forward. Asserted in code, not assumed.
4. **Leakage, demonstrated three ways** (§5 of the report) — a centred window, a feature needing the
   failure date, and a row-wise split.
5. **Validation.** 75 fit engines / 25 validation engines / 100 official test engines, grouped so no
   engine appears twice. `StratifiedGroupKFold` for the spread. Threshold chosen on validation only.
6. **Benchmark against age-based scheduling** — what a plant already does.

## Results (official test set: 100 unseen engines, 12,596 rows, 2.64% positive)

| Model | Precision | Recall | F1 | F2 | PR-AUC |
|---|---|---|---|---|---|
| Always alert | 0.026 | 1.000 | 0.051 | 0.119 | 0.026 |
| Age rule (`cycle > 125`) | 0.118 | 0.883 | 0.209 | 0.385 | 0.210 |
| Logistic regression | 0.630 | 0.922 | 0.748 | 0.843 | 0.882 |
| **Random forest** | **0.691** | **0.937** | **0.795** | **0.875** | **0.909** |

PR-AUC floor is 0.026. Engine-grouped 5-fold CV: logistic regression 0.969 ± 0.007, random forest
0.971 ± 0.005 — **tied**, so the forest's edge is not established by cross-validation. It is carried
forward on the validation engines (0.971 vs 0.961, chosen before the test set was touched) and
because it needs no scaling.

**The model beats age-based scheduling decisively** (F2 0.875 vs 0.385). That comparison, not the
absolute score, is the argument for building it — and it is the opposite of the parent project's
finding, where a hand-written rule *beat* the model.

### Lead time — the metric that matters for a forecast model

| | Value |
|---|---|
| Engines within 30 cycles of failure | 25 |
| **Of those, alerted** | **25 (recall 1.000)** |
| **Median warning before failure** | **33 cycles** (min 24, max 54) |
| Warned ≥ 20 cycles ahead | 25 of 25 |
| Engines flagged while outside the window | 4 of 75 — and **3 were genuinely degrading** (RUL 37, 34, 38) |

### Monitoring view

| Band | Engines | Truly within window | Median true RUL |
|---|---|---|---|
| Critical (≥ 0.70) | 19 | 18 (94.7%) | 15 |
| High (0.377–0.70) | 9 | 7 (77.8%) | 28 |
| Watch (0.15–0.377) | 6 | 0 | 56.5 |
| Low (< 0.15) | 66 | 0 | 101.5 |

Median true remaining life falls monotonically as risk rises. The top 10 worklist entries all have
7–21 cycles of life left.

## The leakage findings

| Leak | Effect |
|---|---|
| Centred rolling window (reaches forward) | PR-AUC 0.768 vs 0.664 honest |
| Fraction-of-life feature (needs the failure date) | validation **0.975** vs 0.971, but test **0.888** vs 0.906 |
| Random row split instead of engine split | 0.986 → 0.971 (engine-wise) → 0.906 (official test) |

The middle one is the instructive one: **the leaked feature improves validation and degrades
deployment.** Leakage does not announce itself as an implausible score.

## Limitations

C-MAPSS is simulated (physics-based, not rule-generated — unlike AI4I, a hand-written rule cannot
recover it, but degradation is still cleaner than reality). FD001 is the easiest of four subsets —
one fault mode, one operating condition. No maintenance actions in the data, so RUL is monotone where
real histories are not. Probabilities are not calibrated; the bands rank well but should not be read
as probabilities. See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) §11 for the full list.

## Run

```bash
cd notebooks
jupyter lab cmapss_forecast_window.ipynb     # Restart kernel, Run All
```

Dependencies: the parent project's [`requirements.txt`](../requirements.txt). Seeds fixed at 42.

## Data provenance

`data/` holds `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt`, downloaded from public mirrors
of the NASA PCoE repository and verified byte-identical across two independent sources
(`train_FD001.txt` MD5 `259f340bac32ce6fa8894815600fa757`).

Saxena, A. & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set (C-MAPSS).* NASA Ames
Prognostics Data Repository.
