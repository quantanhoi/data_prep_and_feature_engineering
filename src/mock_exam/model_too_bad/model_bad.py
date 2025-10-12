"""
EXERCISE – ‘Model far too BAD’

Goal
-----
1. Train / test split on real data.
2. Observe a hopeless metric (~0.20 accuracy).
3. Find the single buggy line (marked ❌) that causes the disaster.
4. Remove / correct it → accuracy climbs to ~0.95.

Real-world failure mode illustrated
-----------------------------------
Someone reshuffled the label vector after the split, destroying the
X–y alignment in the training fold.  The model therefore learns pure
noise and flunks the test set.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 1. Load a real, multi-class dataset  (3 balanced classes).
X, y = load_wine(return_X_y=True, as_frame=True)

# 2. Train / test split (stratified so each set keeps the class mix).
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 3. ❌  BUG – shuffle *only* the label vector → X_tr and y_tr no longer match
rng = np.random.RandomState(0)
y_tr = rng.permutation(y_tr.values)          # comment this line to FIX the bug

# 4. Simple, sensible model inside an sklearn Pipeline
clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, multi_class='multinomial')
)
clf.fit(X_tr, y_tr)

# 5. Evaluation
pred = clf.predict(X_te)
print("Test accuracy  :", accuracy_score(y_te, pred))
print("\nPer-class report:\n", classification_report(y_te, pred))
