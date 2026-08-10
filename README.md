# pricing-heterogeneity

Cross-fitted Double Machine Learning (DR-learner) applied to a customer pricing
A/B test, with full causal validation.

## The question

A pricing A/B test shows a positive average effect. Does that mean the
discount should be rolled out to everyone?

## What this project does

1. Generates a customer dataset with a **known ground-truth treatment effect**,
   so estimator accuracy can be measured rather than assumed.
2. Makes treatment assignment **confounded** (not randomly assigned) — the kind
   of messy assignment real marketing data actually has.
3. Injects realistic data problems: missing values (MCAR), outliers, and
   effects that vary over time.
4. Estimates individual treatment effects using a **cross-fitted doubly-robust
   learner** (double machine learning), and compares it against a naive
   T-learner baseline to show what the simpler approach gets wrong.
5. Validates the estimator: covariate balance, propensity overlap, GATES
   calibration, Qini ranking, two falsification tests, and a sensitivity
   analysis for unmeasured confounding.
6. Runs a **fairness audit** on the resulting policy (four-fifths rule).
7. Compares business policies (treat nobody / treat everyone / targeted) on
   **profit**, not just conversion rate.
8. Designs the confirmatory A/B test (power analysis, sample size, guardrails).

## Headline result

Discounting **every** customer looked good on conversion rate but destroyed
profit once margin was accounted for, because many customers who would have
converted anyway were discounted unnecessarily. A treatment-effect-targeted
policy recovered that loss and outperformed blanket discounting by a wide
margin. Exact figures are in [`outputs/results.json`](outputs/results.json)
and reproduced in the notebook.

## Files

| File | Purpose |
|---|---|
| `pricing_heterogeneity.ipynb` | Full analysis, runnable end-to-end in Colab/Jupyter. Outputs are saved inline. |
| `pricing_heterogeneity.py` | Same analysis as a plain script (`python pricing_heterogeneity.py`). |
| `requirements.txt` | Dependencies. |
| `outputs/results.json` | All numeric results in one file — the source of truth for any number quoted about this project. |
| `outputs/customer_level_results.csv` | Per-customer estimated treatment effects and segments. |
| `outputs/segment_summary.csv` | Segment-level summary. |
| `outputs/policy_comparison.csv` | Profit comparison across pricing policies. |
| `outputs/figures/main_analysis.png` | Diagnostic and results plots. |

## Run it yourself

```bash
pip install -r requirements.txt
python pricing_heterogeneity.py
```

or open `pricing_heterogeneity.ipynb` in Google Colab and run all cells
(~3–4 minutes).

## Method notes

- Ground-truth effects are used **only** to score the estimator, never to fit
  it — this is what makes it possible to report estimator accuracy (e.g.
  correlation with true effect) at all, which isn't measurable on real data
  without a held-out experiment.
- Nuisance models are cross-fitted (trained out-of-fold) to avoid overfitting
  bias in the doubly-robust score.
- The fairness audit and fifth policy comparison are there because a
  profit-optimal targeting policy is not automatically a fair or legal one —
  worth checking before recommending deployment.
