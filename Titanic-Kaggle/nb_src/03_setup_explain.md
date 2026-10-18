**What you should see:** `train: (891, 12)` and `test: (418, 11)`, then a 9-column preview.

- **891** training passengers each with **12** columns; the extra column is `Survived` — our **target** (label). The test set lacks it because that's the thing we must predict: for the 418 test passengers, 1 = survived, 0 = died.
- `PassengerId` is just a row number — like an index in a book. It will never predict survival, but the submission file needs it, so we keep a copy around instead of deleting everything.
- We keep `train_raw` / `test_raw` untouched. If feature engineering ever goes wrong, we re-load from the *raw copies in memory* instead of re-reading disk — a tiny habit that saves lots of debugging.

---
