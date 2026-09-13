# Predictive Maintenance on the AI4I 2020 Dataset

Binary classification of machine failure from sensor snapshots, with the metric and decision
threshold derived from the cost of a missed failure versus an unnecessary inspection — and a
direct test of whether the model has learned anything a hand-written rule could not.

**Notebook:** [`notebooks/ai4i_01.ipynb`](notebooks/ai4i_01.ipynb)

## Data

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(Matzka, 2020), UCI Machine Learning Repository, CC BY 4.0. 10,000 rows: air and process
temperature, rotational speed, torque, tool wear and product quality grade, with a machine
failure label and five failure-mode flags. **The dataset is synthetic.**

## Method

1. **Column audit.** Every column assigned a role before modelling. `UDI` and `Product ID` are
   identifiers (memorisation risk — and `Product ID`'s first character duplicates `Type`).
   The five failure-mode flags are components of the label, so using them is target leakage:
   a tree trained on them scores F1 0.985 and has learned nothing.
2. **Baseline.** 3.39% failure rate. Predicting "no failure" everywhere is 96.6% accurate with
   zero recall — accuracy is the wrong instrument.
3. **Metric chosen before modelling.** A missed failure costs far more than an unnecessary
   inspection, so F2 (recall-weighted) rather than accuracy or F1.
4. **Models.** Logistic regression, then random forest, on the six audited features.
5. **Validation.** Stratified 60/20/20 fit/validation/test split. Threshold selected on the
   validation slice only; test set evaluated once. 5-fold CV for the spread.
6. **Benchmark against a hand-written rule** reconstructed from the data.

## Results (test set, 2,000 rows, 68 failures)

| Model | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| Majority baseline | 0.000 | 0.000 | 0.000 | 0.000 |
| Logistic regression (0.5) | 0.147 | 0.824 | 0.249 | 0.428 |
| Random forest (0.5) | 0.647 | 0.647 | 0.647 | 0.647 |
| Random forest (tuned, 0.267) | 0.445 | 0.838 | 0.582 | 0.713 |
| **Hand-written rule** | **1.000** | **0.838** | **0.912** | **0.866** |

Cross-validated F2: logistic regression 0.408 ± 0.022, random forest 0.675 ± 0.036.

Lowering the threshold from 0.5 to 0.267 raised recall from 0.647 to 0.838: 57 of 68 failures
caught instead of 44, at the cost of 71 unnecessary inspections.

## Limitation

**A three-line rule with no training outperforms the model.** The heat-dissipation, power and
overstrain failure modes are each reproduced exactly (100% of rows) by simple thresholds on
temperature difference, mechanical power and tool wear × torque — because those thresholds
generated the labels. The rule's recall stops at 0.838 only because tool-wear failure is random
within a wear window.

So the model's score is not evidence of predictive skill, and a high F1 on this dataset means
little on its own. The value here is the pipeline: the audit, leakage control, a cost-derived
metric, a threshold chosen off the test set, and a benchmark against a trivial alternative.

Real breakdown records would differ in ways that matter more than the model: retrospective
work orders instead of sensor streams, free-text fault descriptions, inconsistent equipment
naming, missing downtime entries, far fewer failure events, and time ordering — which would
make k-fold invalid and require rolling-origin validation.

## Next steps

- Apply the pipeline to real, time-ordered breakdown records with rolling-origin validation
- Move from failure classification to remaining-useful-life estimation (NASA C-MAPSS)
- Replace the fixed threshold with expected-cost minimisation using real downtime and inspection costs

## Run

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
jupyter lab notebooks/ai4i_01.ipynb
```
