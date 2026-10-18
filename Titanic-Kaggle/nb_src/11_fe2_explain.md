**Output cheat-sheet:**

- **Deck** — decks B/C/D/E (paid 1st-class) ≈ 55–75% survival vs `U` (no cabin listed) dropped to 30%: mystery solved — "no cabin" means "didn't have a cabin" = 3rd class/below deck. Deck U is a genuine feature, not garbage.
- **Fare-per-person quartiles** — clean monotonic ladder (~25% → ~58%): each quartile you climb, survival rises ~10 points. Wealth, again.
- **Embarked/Fare fills** — trivially small (2 + 1 rows); mode/median-by-class is the right level of caution for such tiny holes. Over-engineering 2 rows creates complexity without information.
- **Age after RF imputation** — no NaN left *anywhere*; mean survivor age ~28 vs ~31 for the dead. The few-years gap is real (children > adults) but modest — age is a *weak* individual signal, another reason Section 4's family-group trick beats any age-driven model.

> **Data-leakage check:** every fill here uses `train['col']` statistics (mode/median/RF fit). The test set is only ever *transformed*. That is what makes the cross-validation scores in Section 7 valid estimates of test performance.

---
