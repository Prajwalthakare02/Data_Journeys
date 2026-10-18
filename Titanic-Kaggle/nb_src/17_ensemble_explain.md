**What to take from the numbers:**

- **Ensemble CV ≈ 0.83–0.84** — typically between the best individual model and +0.005 above it. The honest ceiling again: ~0.84. If your ensemble CV *drops* below the best single model, the models were too correlated (all of them basically learning "female + Pclass") — diversity is what earns the small ensemble gain.
- **Ensemble vs rule agreement ≈ 92–96%** — the two strategies disagree on only ~2–5% of test rows, and they disagree **exactly where the rule has families evidence and the model doesn't** (or vice versa). That gap is where the extra leaderboard points live.

---
