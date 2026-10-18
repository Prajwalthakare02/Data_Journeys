# 🚢 Titanic — Kaggle Competition Notebook

End-to-end, fully-commented Jupyter solution for Kaggle's classic [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. Every section follows the same pattern: **explanation markdown → commented code → output interpretation**.

Produces two submission files built on two traceable strategies.

## Results (measured on the public leaderboard)

| Submission file | Strategy | Honest CV | Public LB |
|---|---|---|---|
| `submissions/submission.csv` | Family-group survival rule (v2) | train check 0.89 *(circular)* | **0.80382 (v1) · v2 ~0.81 expected** |
| `submissions/submission_model.csv` | Soft-voting ML ensemble | 0.826 | 0.76794 |

Reference baselines: "everyone dies" = 0.62 · "all women survive" = 0.76589.
The group rule beats both references *and* the ML pipeline on the board — structural information beat model tuning here.

## What's inside

**Feature engineering** (fit on train only, applied to test — no leakage):
- `Title` from names (Mr/Mrs/Miss/**Master**/Rare)
- `Surname`, `FamilySize`, `IsAlone`
- `TicketGroupSize` via train+test ticket counts
- `Deck`/`HasCabin` from Cabin, `FarePerPerson`
- **Model-based Age imputation** with a RandomForest fit on known-age train rows

**The family-group rule** (the LB-targeted strategy):
- Families reconstructed with **union-find**: passengers merged transitively when they share a surname OR a ticket (maids/step-relatives included — 780 merged components across train+test)
- Any woman/child known (train) to have died → the whole family's women/children die in test; all-known-survived → survive
- Extra check: families where every known train member died (≥2) doom their test women too
- Adult men always predicted dead — **measured justification**: train men with surviving-family evidence die at the same rate (0.198 vs 0.160) → no male overrides
- Net: recovers **14 doomed women + 8 saved boys** on test with evidence-backed flags
- Idea documented in the Kaggle community (Chris Deotte's family-group notebooks)

**Models & validation:**
- LogisticRegression / RandomForest / HistGradientBoosting / XGBoost
- Stratified 5-fold CV: 0.79 / 0.82 / 0.83 / 0.83
- Soft-voting probability ensemble (`submission_model.csv`)

**Two real leakage catches during development** (documented inside the notebook):
1. A label-derived feature (`RulePred`) entered the model matrix — CV jumped to a fake 0.89; removed in Section 6 with an explanatory note
2. First rule draft let *adult men* of "safe families" survive — the sanity check caught it (190 predicted survivors vs 152 test women); corrected in Section 4

## Files

```
Titanic-Kaggle/
├── Titanic.ipynb           # the notebook (20 cells, executed, outputs included)
├── README.md
├── submissions/
│   ├── submission.csv        # FINAL: family-group rule
│   └── submission_model.csv  # companion: ML ensemble
├── nb_src/                 # notebook source fragments (md + py cells)
└── tools/
    └── assemble_notebook.py  # rebuilds Titanic.ipynb from nb_src fragments
```

Competition data (`train.csv`, `test.csv`, `gender_submission.csv`) is **not** redistributed here — download it from the [competition Data tab](https://www.kaggle.com/competitions/titanic/data) and drop the files next to `Titanic.ipynb` (the notebook locates them automatically, locally or on Kaggle).

## Run it

```bash
# 1) put train.csv + test.csv (from Kaggle) in this folder

# 2) optional: rebuild the notebook from source fragments
python tools/assemble_notebook.py

# 3) execute all cells (writes outputs + both submission files)
jupyter nbconvert --to notebook --execute --inplace Titanic.ipynb
```

Requires Python 3.10+ with `pandas numpy scikit-learn xgboost matplotlib jupyter`.

## Submitting

1. Kaggle → Titanic → **Submit Predictions**
2. Upload `submissions/submission.csv` (family-group rule)
3. Optionally also `submissions/submission_model.csv` (ensemble) to see the ML baseline on the board
4. 10 submissions/day; rely on CV reasoning rather than leaderboard-fitting

## Attribution & honesty note

The family-group strategy replicates a technique published by Chris Deotte and others on Kaggle. It is leaderboard-targeted and dataset-specific, not a generally transferable method — which is exactly why the standard ML pipeline is built alongside it for fair comparison.
