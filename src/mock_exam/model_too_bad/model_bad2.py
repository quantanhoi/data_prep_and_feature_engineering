"""
MODEL LOOKS TERRIBLE – FIND THE ENCODER BUG
 • 1 000 rows, 5 colours, 5-class label
 • Train split deliberately lacks ‘cyan’ and ‘magenta’
 • BUG: pd.factorize() called separately on train and test
   (order-of-appearance mapping)  → test accuracy ≈ 0.20
 • FIX: use the mapping learned on train for the test split
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score

rng      = np.random.RandomState(0)
classes  = np.array(['red','green','blue','cyan','magenta'])

# ------------------------------------------------------------
# 1.  make a data set whose target is strongly tied to colour
# ------------------------------------------------------------
n = 1_000
X = pd.DataFrame({"colour": rng.choice(classes, n)})

def noisy_identity(col):
    # correct label 80 % of the time, otherwise random wrong class
    y = col.copy()
    mask = rng.rand(n) > 0.80
    for i in np.where(mask)[0]:
        y.iat[i] = rng.choice(classes[classes != y.iat[i]])
    return y

y = noisy_identity(X['colour'])

# ------------------------------------------------------------
# 2.  train/test split WITHOUT shuffling
#     first 700 rows = train  (only red/green/blue)
#     last 300 rows  = test   (contains cyan & magenta)
# ------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, shuffle=False)

# sanity-check the category sets
print("Train categories:", X_tr['colour'].unique())
print("Test  categories:", X_te['colour'].unique(), "\n")

# ------------------------------------------------------------
# 3.  BUG – encode each split independently  (order matters!)
# ------------------------------------------------------------
def encode_series(s):
    codes, uniques = pd.factorize(s, sort=False)
    return codes, uniques

X_tr_enc, uniq_tr = encode_series(X_tr['colour'])      # fit here
X_te_enc, _       = encode_series(X_te['colour'])      # ❌ re-fit here

clf = LogisticRegression(max_iter=400,
                         multi_class='multinomial').fit(
                         X_tr_enc.reshape(-1,1), y_tr)

print("Accuracy WITH bug :", round(
        accuracy_score(y_te, clf.predict(X_te_enc.reshape(-1,1))), 3))

# ------------------------------------------------------------
# 4.  FIX – transform test split with the train mapping only
# ------------------------------------------------------------
mapping = {c: i for i, c in enumerate(uniq_tr)}
X_te_fixed = X_te['colour'].map(mapping).fillna(-1).astype(int)

print("Accuracy after fix:", round(
        accuracy_score(y_te, clf.predict(X_te_fixed.values.reshape(-1,1))), 3))
