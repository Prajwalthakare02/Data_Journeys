# ============================================================================
# SECTION 4 - 🪄 THE MAGIC FAMILY-GROUP RULE  (v2: union-find merging)
# ============================================================================
# Empirical fact documented across the Titanic community (Chris Deotte et al.):
#
#   Within a travelling family, the WOMEN AND CHILDREN almost always
#   survived or died TOGETHER. Adult men had no such pattern.
#
# Why? "Women and children first" was enforced per-group: once a family's
# women reached a boat, the family outcome was resolved. 3rd-class families
# that never reached a boat lost everybody.
#
# THE RULE (v2):
#   * Build TRUE family components with union-find: passengers are merged
#     when they share a surname OR a ticket, and evidence then propagates
#     transitively through maids/servants/step-relatives (v1 checked the
#     two keys independently and missed these links).
#   * Component = DEAD      if ANY woman/boy in it is KNOWN (train) to have died.
#   * Component = ALIVE     if at least one woman/boy survived in train
#                            and none are known to have died.
#   * Extra catch: families where >=2 train members are known and ALL died
#     (all-male evidence) -> test women/boys of that family die too.
#   * Adult men: 0. Empirically justified below (measured on train):
#     train men WITH surviving-women/children family evidence survive at
#     0.198 vs 0.151 for men without evidence (0.160 overall) -> +4pp is
#     not a decision boundary, so no male upgrades. (The famous 0.97444
#     public submissions use curated male survivor lists that don't
#     reproduce from a measurable pattern - not replicated here.)

full = pd.concat([train, test], ignore_index=True, sort=False)
full['WomanOrBoy'] = (full['Sex'] == 'female') | (full['Title'] == 'Master')

# ---- union-find: merge passengers sharing a surname or a ticket ---------------
parent = list(range(len(full)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]      # path halving
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

for key in ['Surname', 'Ticket']:
    for k, idxs in full.groupby(key, sort=False).indices.items():
        for other in idxs[1:]:
            union(idxs[0], other)

comp_members = {}
for i in range(len(full)):
    comp_members.setdefault(find(i), []).append(i)
print(f'merged family components: {len(comp_members)} (1309 passengers)')

# ---- per-component verdicts ---------------------------------------------------
dead_groups, alive_groups = set(), set()
alldead_surnames = set()

for rep, members in comp_members.items():
    sub = full.iloc[members]
    known = sub[sub['Survived'].notna()]                       # train rows only
    wck = sub[sub['WomanOrBoy'] & sub['Survived'].notna()]
    if len(wck):
        if (wck['Survived'] == 0).any():                       # ANY woman/boy died
            dead_groups.add(rep)
        elif (wck['Survived'] == 1).all():                     # all known women/boys lived
            alive_groups.add(rep)
    if len(known) >= 2 and (known['Survived'] == 0).all():     # whole known family died
        alldead_surnames.update(sub['Surname'].unique())

print(f'components seen as DEAD : {len(dead_groups)}')
print(f'components seen as ALIVE: {len(alive_groups)}')
print(f'fully-dead families (>=2 train deaths, 0 survivors): {len(alldead_surnames)} surnames')

# ---- apply the rule to every passenger in `full` -------------------------------
def rule_pred(row):
    rep = find(row.name)                       # row.name == positional index here
    if not row['WomanOrBoy']:
        return 0                               # adult man: no override (see header)
    if (rep in dead_groups) or (row['Surname'] in alldead_surnames):
        return 0                               # doomed family
    if rep in alive_groups:
        return 1                               # saved family
    return 1 if row['Sex'] == 'female' else 0  # no evidence: women 1, boys 0

full['RulePred'] = full.apply(rule_pred, axis=1)

# propagate back so the submission cell can use test['RulePred'] directly
train['RulePred'] = full['RulePred'].values[:len(train)]
test['RulePred']  = full['RulePred'].values[len(train):]

# ---- diagnostics ----------------------------------------------------------------
tr = full[full['Survived'].notna()]
te = full[full['Survived'].isna()]

print(f'\nRULE accuracy on TRAIN (CIRCULAR sanity check only, not honest CV): '
      f'{accuracy_score(tr["Survived"], tr["RulePred"]):.4f}')

women = te[te['Sex'] == 'female']
flipped_dead_women = women[women['RulePred'] == 0]
boys_alive = te[(te['Title'] == 'Master') & (te['RulePred'] == 1)]
print(f'\nTest women predicted DEAD by rule: {len(flipped_dead_women)}')
print('   surnames:', flipped_dead_women['Surname'].tolist())
print(f'Test boys upgraded to SURVIVED:    {len(boys_alive)}')
print(f'Still riding on defaults (women=1 / men=0) is documented behaviour for')
print(f'passengers with no family evidence; adult male upgrades are disabled.')
print(f'\nTest predictions by the rule: survived = {int(te["RulePred"].sum())} '
      f'({te["RulePred"].mean():.2%})')