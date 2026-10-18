**Expected output (numbers vary a hair with library versions):**

| Model | Why it lands there |
|---|---|
| LogisticRegression ≈ 0.79–0.80 | Linear walls can't express "families of 2–4 good, ≥5 bad" — it averages the two effects. Solid baseline, never the winner here. |
| RandomForest / HistGB / XGBoost ≈ 0.82–0.84 | Trees capture the non-linear sweet spots. With only 891 rows, accuracy **plateaus ≈ 0.84 no matter what you tune** — the ceiling is *information*, not brainpower. |

**Why this plateau matters.** Once CV stops moving across very different model families, the *individual features* have been almost fully exploited — the remaining error is dominated by the ~10% of passengers whose outcome isn't recoverable from personal attributes. The additional information source is group correlation, which is why the final submission uses the family rule from Section 4 rather than the ensemble.

Also notice the ± std (~0.01–0.03): with 891 rows, even a great model's score wobbles a couple of points between folds. Never compare models by eyeballing single numbers — compare within their uncertainty bands.

---
