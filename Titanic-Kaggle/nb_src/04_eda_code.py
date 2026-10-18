# ============================================================================
# SECTION 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# Standard practice: look at the data before building models. EDA establishes
# which patterns exist and how strong they are, so model choices have a basis.
# A model can only learn patterns that exist. EDA finds those patterns first.
#
# Question 1: How imbalanced are the classes?  -> train['Survived'].mean()
# Question 2: Which raw columns already split survival strongly?
#             Sex, Pclass, Embarked are the classic ones.
# We compute the SURVIVAL RATE for each category value. `groupby(X)['Survived'].mean()`
# means: "for every distinct value of X, what fraction survived?"

print(f'Overall survival rate: {train["Survived"].mean():.2%} ({train["Survived"].sum()} of {len(train)})\n')

for col in ['Sex', 'Pclass', 'Embarked']:
    table = train.groupby(col, observed=True)['Survived'].agg(['mean', 'count'])
    table.columns = ['survival_rate', 'n']
    print(f'--- Survival by {col} ---')
    print(table.round(3), '\n')

# A quick visual: survival rate for men vs women. One glance > ten tables.
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
train.groupby('Sex')['Survived'].mean().plot.bar(ax=axes[0], color=['salmon', 'steelblue'])
axes[0].set_title('Survival rate by Sex')
axes[0].axhline(train['Survived'].mean(), color='k', ls='--', lw=1)
train.groupby('Pclass')['Survived'].mean().plot.bar(ax=axes[1], color='seagreen')
axes[1].set_title('Survival rate by Passenger Class')
axes[1].axhline(train['Survived'].mean(), color='k', ls='--', lw=1)
plt.tight_layout(); plt.show()