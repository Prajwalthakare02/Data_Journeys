**Reading the output (this is the intuition everything else builds on):**

| Signal | What the numbers say | Why it makes sense |
|---|---|---|
| **Baseline** | 38% survived overall | 1502 of 2224 aboard died — the *majority* prediction ("everyone died") scores 62%. Good ML must beat that. |
| **Sex** | ♀ 74% vs ♂ 19% | "Women and children first" lifeboat protocol. **Sex alone is the strongest predictor in this dataset.** |
| **Pclass** | 63% → 47% → 24% | First class cabins were nearer the boat deck; third class was deep below and lifeboats left early during confusion. Wealth = proximity = survival. |
| **Embarked** | C 55% > Q 39% > S 34% | *Not* a real causal pattern. Cherbourg (C) simply boarded proportionally more 1st-class passengers. Embarked is a weak *proxy* for Pclass. |

**Takeaway:** with Sex ~74/19 and Pclass ~63/24 already separating survival strongly, the models built later must go beyond the raw columns to beat ~0.76 meaningfully — that is what Sections 3–5 (feature engineering) and Section 4 (family groups) supply.

---
