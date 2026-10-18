# ============================================================================
# SECTION 1 - SETUP & DATA LOAD
# ================================================= #
# We import the tools we need. Rather than memorising imports, understand roles:
# * pandas / numpy   -> the "Excel + calculator" of Python (tables + math)
# * matplotlib       -> charts
# * sklearn          -> reliable, textbook implementations of every classic ML tool
# * xgboost          -> the best-spending gradient booster; state of the art for tabular data
#
# RANDOM_SEED = 42: we freeze the "randomness" so anyone who re-runs this notebook
# gets the exact same numbers. Reproducibility is a professional habit.
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score

import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# DATA_DIR -> find the folder that actually contains train.csv / test.csv,
# no matter where the notebook runs: local repo root, notebook folder, or
# Kaggle's kernel sandbox (../input/titanic or /kaggle/input/titanic).
DATA_DIR = None
for cand in [os.path.join('.', 'ML', 'Kaggle'), '.', os.path.join('..', 'ML', 'Kaggle'),
             os.path.join('..', 'input', 'titanic'), '/kaggle/input/titanic']:
    if os.path.isfile(os.path.join(cand, 'train.csv')) and os.path.isfile(os.path.join(cand, 'test.csv')):
        DATA_DIR = cand
        break
if DATA_DIR is None:
    raise FileNotFoundError('Could not locate train.csv/test.csv - set DATA_DIR manually.')

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# GOLDEN RULE: never modify the raw files. We work on copies so the original
# data can always tell us the truth if something goes wrong.
train_raw, test_raw = train.copy(), test.copy()

print('train shape:', train.shape)   # expect (891, 12)
print('test  shape:', test.shape)    # expect (418, 11)  <- no Survived column!
train.head(3)