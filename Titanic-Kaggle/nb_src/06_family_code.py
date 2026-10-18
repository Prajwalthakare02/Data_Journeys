# ============================================================================
# SECTION 3 - FAMILY FOUNDATIONS (Surname, Title, FamilySize, Ticket groups)
# ============================================================================
# Names and tickets are pure TEXT — models can't read them. But they hide the
# most valuable structure in the dataset: PEOPLE WHO TRAVELLED TOGETHER.
# A mother and her daughter usually share a surname AND a ticket. If we can
# reconstruct "who was in whose group", survival stops being an individual
# coin-flip and becomes a GROUP property. This is the foundation of every
# high-scoring Titanic submission.

# ---- 3a. TITLE ------------------------------------------------------------
# "Braund, Mr. Owen Harris" -> the word before the dot is an honorific.
# It encodes Sex (Mr/Mrs/Miss), age bracket (Master = young boy),
# and social rank (Dr/Rev/Col). Regex ',\s*([^\.]+)\.' grabs that word.
def extract_title(name):
    m = re.search(r',\s*([^\.]+)\.', name)
    return m.group(1).strip() if m else 'Rare'

for df in [train, test]:
    df['Title'] = df['Name'].apply(extract_title)

# Normalise exotic spellings to the 4 core titles + 'Rare'.
# (Mlle = Mademoiselle = French 'Miss', Mme = Madame = French 'Mrs', ...)
title_map = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'the Countess': 'Rare',
             'Lady': 'Rare', 'Countess': 'Rare', 'Dona': 'Rare', 'Don': 'Rare',
             'Jonkheer': 'Rare', 'Sir': 'Rare', 'Dr': 'Rare', 'Rev': 'Rare',
             'Col': 'Rare', 'Major': 'Rare', 'Capt': 'Rare'}
train['Title'] = train['Title'].replace(title_map)
test['Title'] = test['Title'].replace(title_map)
print('Title counts:\n', train['Title'].value_counts())

# ---- 3b. SURNAME ----------------------------------------------------------
# Family name is everything before the first comma.
for df in [train, test]:
    df['Surname'] = df['Name'].str.split(',').str[0]

# ---- 3c. FAMILY SIZE -------------------------------------------------------
for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1      # +1 = the passenger himself
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ---- 3d. TICKET GROUP SIZE (train + test COMBINED) --------------------------
# A couple might have the husband only in train and the wife only in test.
# Counting tickets on the CONCATENATION sees both — and that's legal: we use
# no survival labels here, only "who shares a ticket", which is available
# for the whole dataset before we ever peek at answers.
ticket_shared = pd.concat([train['Ticket'], test['Ticket']]).value_counts()
for df in [train, test]:
    df['TicketGroupSize'] = df['Ticket'].map(ticket_shared)

print('\nTicket groups of size >= 5:', (ticket_shared >= 5).sum())
print('Survival by TicketGroupSize:')
print(train.groupby('TicketGroupSize')['Survived'].mean().round(3))
print('\nSurvival by FamilySize:')
print(train.groupby('FamilySize')['Survived'].mean().round(3))