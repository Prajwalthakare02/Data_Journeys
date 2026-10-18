# ============================================================================
# SECTION 7 - MODELS HONESTLY CROSS-VALIDATED
# ============================================================================
# The core question cross-validation answers: "How does this model perform on
# data it has NEVER seen?"
# 5-Fold STRATIFIED cross-validation answers it: the 891 train rows are split
# into 5 buckets (~178 each). We train on 4 buckets and score ourselves on the
# held-out one, five times, rotating the held-out bucket. Stratified = each
# bucket keeps the 38%/62% class ratio, so every fold sees both classes.
# mean ± std over the five folds = estimated true accuracy.

from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

models = {
    # Linear model: needs feature scaling (Age ~0-80 vs HasCabin 0/1) so
    # the regularisation treats every feature fairly.
    'LogisticRegression': Pipeline([('scale', StandardScaler()),
                                    ('lr', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),

    # Random Forest: many decision trees on random feature/data subsets.
    # Handles raw categories fine, no scaling needed. max_depth/leaf limits
    # keep trees from memorising 891 rows (overfitting control).
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=8,
                                           min_samples_leaf=2, random_state=RANDOM_SEED,
                                           n_jobs=-1),

    # sklearn's gradient booster (no external deps). Boosts shallow trees that
    # each fix the previous tree's errors. Very strong on small tabular data.
    'HistGB': HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1,
                                             max_iter=200, random_state=RANDOM_SEED),

    # XGBoost: the competition-standard gradient booster.
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=RANDOM_SEED, n_jobs=-1,
                                 eval_metric='logloss'),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'{name:20s} CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}   folds={np.round(scores,3)}')