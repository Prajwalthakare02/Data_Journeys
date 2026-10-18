# ============================================================================
# SECTION 5 - REMAINING FEATURES & IMPUTATION (no data leakage!)
# ============================================================================
# Imputation = filling missing values. Two rules professionals never break:
#   1. Every statistic is COMPUTED ON TRAIN ONLY, then applied to test.
#      (Imagine the test row was a real future customer — you couldn't peek.)
#   2. Impute with the most-informative statistic, not just the global median.

# ---- 5a. Deck & HasCabin ----------------------------------------------------
# 'Cabin' is 77% missing. First letter = deck number (B, C, ...). Missing cabin
# is ITSELF informative: steerage passengers had no assigned cabin.
for df in [train, test]:
    df['Deck'] = df['Cabin'].fillna('U').str[0]      # U = Unknown
    df['HasCabin'] = df['Cabin'].notna().astype(int)
print('Survival by Deck:'); print(train.groupby('Deck')['Survived'].mean().round(3))

# ---- 5b. Fare-per-person -----------------------------------------------------
# Fare = TOTAL ticket price paid by the WHOLE group sharing the ticket.
# Divide by group size to get an honest per-person wealth signal.
for df in [train, test]:
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
print('\nSurvival by FarePerPerson quartile:')
print(train.groupby(pd.qcut(train['FarePerPerson'], 4))['Survived'].mean().round(3))

# ---- 5c. Embarked (2 train NaN) ----------------------------------------------
# Only 2 values missing -> fill with the train mode (S). Cheap & safe.
embarked_mode = train['Embarked'].mode()[0]
train['Embarked'] = train['Embarked'].fillna(embarked_mode)
test['Embarked'] = test['Embarked'].fillna(embarked_mode)

# ---- 5d. Fare (1 test NaN) ----------------------------------------------------
# 1 test passenger with no Fare. Fill with the train MEDIAN FARE OF THEIR CLASS:
# a 3rd-class passenger's missing fare should impute to a 3rd-class fare.
fare_by_class = train.groupby('Pclass')['Fare'].median()
test['Fare'] = test['Fare'].fillna(test['Pclass'].map(fare_by_class))
for df in [train, test]:                      # refill FarePerPerson after fill
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# ---- 5e. Age — MODEL-BASED imputation -----------------------------------------
# 20% of Ages are missing. A global median would erase the fact that age
# patterns differ by class & title (a Master is ~4, a Mrs ~35). Instead we let
# a RandomForest LEARN Age from the other columns, then fill the holes.
#   * fit on: train rows where Age is known
#   * predict: every row (train or test) where Age is missing
age_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'HasCabin']

# Title/Sex still need numbers here -> quick integer map of the 5 titles / 2 sexes.
sex_num = {'male': 0, 'female': 1}
title_num = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4}
for df in [train, test]:
    df['SexNum'] = df['Sex'].map(sex_num)
    df['TitleNum'] = df['Title'].map(title_num)
age_features += ['SexNum', 'TitleNum']

known = train[train['Age'].notna()]
rf_age = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
rf_age.fit(train.loc[known.index, age_features], known['Age'])

for df in [train, test]:
    hole = df['Age'].isna()
    df.loc[hole, 'Age'] = rf_age.predict(df.loc[hole, age_features])

print('\nAge after imputation -> train NaN:', train['Age'].isna().sum(),
      '| test NaN:', test['Age'].isna().sum(), end='  ')
print('| survivors(=1) mean age:', train.query('Survived == 1')['Age'].mean().round(1),
      '| victims mean age:', train.query('Survived == 0')['Age'].mean().round(1))