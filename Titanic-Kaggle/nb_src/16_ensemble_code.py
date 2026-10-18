# ============================================================================
# SECTION 8 - ENSEMBLE (soft voting of probabilities)
# ============================================================================
# Ensembles win because DIFFERENT models make DIFFERENT mistakes. If a
# linear model errs on weird-family cases while a tree handles them, and the
# tree errs on smooth-wealth cases the linear model nails, their average is
# better than either. We keep predictions' PROBABILITIES (0..1), not labels,
# because probabilities average smoothly.
#
# cross_val_predict(..., method='predict_proba') returns, for every TRAIN row,
# the probability given to it by a model that never saw that row during its
# own fold's training — the honest collage of all 5 folds (called "OOF",
# out-of-fold). Averaging the four models' OOF probabilities is our ensemble.

oof_probs, test_probs = {}, {}
for name, model in models.items():
    oof_probs[name] = cross_val_predict(model, X_train, y_train, cv=cv,
                                        method='predict_proba', n_jobs=-1)[:, 1]

    fitted = model.fit(X_train, y_train)
    test_probs[name] = fitted.predict_proba(X_test)[:, 1]

oof_avg = pd.DataFrame(oof_probs).mean(axis=1).values
test_avg = pd.DataFrame(test_probs).mean(axis=1).values

print('Out-of-fold (honest) accuracies:')
for name in models:
    print(f'  {name:20s} {(oof_probs[name].round() == y_train).mean():.4f}')
print(f'  {"ENSEMBLE(AVERAGE)":20s} {((oof_avg > .5).astype(int) == y_train).mean():.4f}')

# How much do our two strategies agree on the test set?
rule_test = full.loc[full['Survived'].isna(), 'RulePred'].values
agree = (rule_test == (test_avg > .5).astype(int)).mean()
print(f'\nEnsemble vs family-rule agreement on 418 test rows: {agree:.2%}')