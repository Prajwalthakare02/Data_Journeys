# 📋 Section 10 — Results summary & next steps

**Pipeline used (all 10 sections):** load → EDA → family structure → group rule →
imputation → encoding → CV-validated models → ensemble → two submissions.

## What drove the results

| Driver | Impact | Evidence in this notebook |
|---|---|---|
| Feature engineering | ~+4 CV points vs raw columns | Section 7 models reach ~0.83 on 14 engineered features; raw baselines stay ~0.80 |
| Family-group signal | The public-LB jump, ~93–97% expected | Section 4: 12 doomed women + 8 saved boys recovered by group evidence, with ~90% model/rule agreement |
| Overfitting control on 891 rows | Reliable CV numbers | Tree depth/leaf limits, and ±0.01–0.02 CV spread vs model families in Section 7 |
| Anti-leakage discipline | Scores that mean what they say | Every statistic fit on train only (Section 5); `RulePred` detected and excluded from model features (Section 6) |
| Ensembling | ~0.83 plateaus regardless of model family | Section 7–8; little headroom left in individual features at ~0.84 |

## Possible next steps

| Idea | Why it would help |
|---|---|
| Tune tree ensembles (Optuna/grid) | Turns 0.826 → ~0.835 CV; honest but small gains on 891 rows |
| Add `Title_Pclass`, route-based ticket encodings | The two remaining text features not fully represented |
| Model the group signal as a feature (family death-rate target encoding) | Blend rule logic into a single ML pipeline |
| Verify the rule on a train holdout by families | Estimates the rule's true generalisation more honestly than the circular check |

## Notebook maintenance

- Rebuild with: `python ML/Kaggle/tools/assemble_notebook.py` (source cells live in `ML/Kaggle/nb_src/`).
- Re-execute with: `jupyter nbconvert --to notebook --execute --inplace ML/Kaggle/Titanic.ipynb` from the repo root.
- Old experiments from previous sessions are stored in `ML/Kaggle/archive/`.
