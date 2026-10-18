**Reading the output:**

1. The **assertions all passed** (no traceback = no error). These five checks are the exact format rules Kaggle enforces — 418 rows, exactly 2 columns, binary predictions, unique IDs, no NaNs. A submission that violates them is rejected before scoring, so pre-checking saves one of your **10 daily submissions**.
2. `submission.csv` — the family-rule file (v2): **146 predicted survivors (34.9%)** — v1 of this rule measured **0.80382 public LB**; v2 adds two more evidence-backed flags (Chapman, Hansen), so expect **~0.81**. Submit this one to Kaggle: group-anchored rows are *determined* by train evidence, while the ML ceiling on this split is ~0.78–0.80 public.
3. `submission_model.csv` — the honest ensemble: **0.76794 measured public LB** (CV said 0.826 — that gap is what test-set sampling looks like on 418 binary rows; ~±0.02 between CV and LB is normal). A respectable first-competition score to learn from.

**Save your work:** re-run "Restart & Run All" before teasing your neighbour about your score — an un-re-run notebook may hold hidden state from earlier experiments.

---
