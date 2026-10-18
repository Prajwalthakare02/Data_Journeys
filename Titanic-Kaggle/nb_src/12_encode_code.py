# ============================================================================
# SECTION 6 - ENCODING & FINAL FEATURE MATRIX
# ============================================================================
# Models eat numbers only. Our remaining string columns (Sex, Title, Embarked,
# Deck) are CATEGORICAL. Two viable encodings:
#   * One-hot: 1 column per category — safe for Linear models.
#   * Label-encoding: 1 integer per category — fine for TREE models, which
#     learn splits, not arithmetic.
# We use LabelEncoder (trees do the heavy lifting for us). Fit it on the
# COMBINATION of train+test values for that column so a category that only
# appears in test never becomes an invisible/unguessable value.

categorical = ['Sex', 'Title', 'Embarked', 'Deck']
le_store = {}                                     # keep encoders if you re-encode
for col in categorical:
    le = LabelEncoder().fit(pd.concat([train[col], test[col]]))
    for df in [train, test]:
        df[col + '_enc'] = le.transform(df[col])
    le_store[col] = le

# FINAL SELECTION — what we keep and what we throw away, and WHY:
#   Drop: PassengerId (row number), Name (extracted), Ticket (encoded),
#         Cabin (extracted), Surname (group key only), original strings
#         (duplicates of the _enc columns).
#   Drop: RulePred — CRITICAL ANTI-LEAK LESSON: it was DERIVED FROM THE
#         TARGET (train labels). A label-derived feature makes CV explode
#         (~0.89 instead of ~0.84) with zero real-world validity. The rule
#         is used for the RULE submission; it must never feed the models.
DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Surname', 'Sex', 'Title',
        'Embarked', 'Deck', 'SexNum', 'TitleNum', 'RulePred']

feature_cols = [c for c in train.columns if c not in DROP + ['Survived']]
X_train = train[feature_cols].copy()
y_train = train['Survived'].copy()
X_test = test[feature_cols].copy()

print('Final feature matrix:', X_train.shape, '| features:')
print(feature_cols)
print('\nMissing values — train:', int(X_train.isna().sum().sum()),
      '| test:', int(X_test.isna().sum().sum()))
print('Feature dtypes all numeric:', bool(X_train.dtypes.apply(lambda t: pd.api.types.is_numeric_dtype(t)).all()))