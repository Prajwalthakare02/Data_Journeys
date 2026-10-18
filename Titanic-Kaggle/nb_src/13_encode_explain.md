**What you should see:** a `(891, 14)` train matrix and `(418, 14)` test matrix, 14 numeric
features, and **zero missing values, all numeric dtypes**. The 14 features are:

| Feature | Meaning |
|---|---|
| Pclass, Age, SibSp, Parch, Fare | raw numbers, cleaned/imputed |
| FamilySize, IsAlone, TicketGroupSize | group structure we engineered |
| HasCabin, Deck_enc | cabin/deck status (proxy for wealth) |
| FarePerPerson | honest wealth signal |
| Sex_enc, Title_enc, Embarked_enc | encoded categoricals |

**Two encoding warnings worth remembering:**
1. `LabelEncoder` taught on train+test values means every category in both sets has an integer. Doing it train-only would crash the moment test contains an unseen deck letter.
2. Because `Title_Pclass` style combo columns were deliberately skipped, note that our model features stay modest (~13) — deliberately. More features ≠ better model; Titanic is a *small* dataset, and every extra feature risks overfitting noise.

---
