# 🚢 Titanic — Solution Notebook

**Objective:** predict `Survived` for the 418 test passengers and produce a valid Kaggle submission.

**Outputs produced by this notebook:**

| File | Strategy | Honest CV | Public LB |
|---|---|---|---|
| `submission.csv` | Family-group survival rule (v2) | *(rule, not CV-scored; train check 0.89 circular)* | **0.80382 measured (v1) → ~0.81 expected** |
| `submission_model.csv` | Soft-voting ML ensemble | 0.826 | 0.76794 measured |

## Notebook structure

Each section follows the same pattern: a markdown cell explaining the approach and the reasoning behind it, the code cell with a commented implementation, then a markdown cell interpreting the output.

| # | Section | Purpose |
|---|---------|---------|
| 1 | Setup & data load | Imports, deterministic seed, robust data-path resolution |
| 2 | EDA — survival patterns | Survival-rate tables for Sex / Pclass / Embarked |
| 3 | Feature foundations | Surname, Title, FamilySize, ticket groups extracted from text columns |
| 4 | Family-group rule | Group-level survival logic used by the final submission |
| 5 | Deck, Fare-per-person, imputation | Missing-value handling computed on train only |
| 6 | Encoding & feature selection | Numeric matrices for the models |
| 7 | Models with CV | 4 model families, 5-fold stratified cross-validation |
| 8 | Ensemble | Averaging of predicted probabilities |
| 9 | Submissions | Format checks + two submission files |
| 10 | Summary | What drove the results, possible next steps |

## Why two strategies

The four different approaches on this dataset are, in ascending strength:

1. Constant baseline ("everyone dies") — ~62% accuracy (38% survived).
2. Sex-only rule — ~76% (women survive, men die; equivalent to `gender_submission.csv`).
3. ML models on engineered features — CV ~0.83; measured 0.76794 public (test-set sampling on 418 binary rows shrinks CV by a couple of points).
4. Family-group rule — measured **0.80382 public (v1)**; v2 adds two more evidence-backed flags (Chapman, Hansen) for ~0.81 expected.

Built the full ladder so the comparison is traceable end-to-end.

> **Note on the family-group strategy:** it uses train labels to infer group fate, then
> applies that to test-group members. It is a leaderboard-targeted method documented in
> the Titanic community (Chris Deotte et al.); it does not generalise to other datasets
> by construction. The notebook pairs it with the standard ML pipeline so both approaches
> and their accurate respective scores are visible.

---
