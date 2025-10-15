# Model Diagnostic Guide: Too Bad or Too Good?

## ⚠️ CRITICAL VALIDATION PRINCIPLE

```
┌──────────────────────────────────────────────────────────────┐
│ ALL diagnostics should be confirmed with cross-validation!  │
│                                                               │
│ ❌ Single train-test split can be misleading:                │
│    • Lucky/unlucky split                                     │
│    • Unrepresentative test set                               │
│    • Overconfident metrics                                   │
│                                                               │
│ ✅ ALWAYS use Pipeline + cross_val_score for reliable        │
│    estimates before making final decisions                   │
└──────────────────────────────────────────────────────────────┘
```

## Quick Decision Tree

```
                    ┌─────────────────────────┐
                    │  Train your model       │
                    │  Get train & test scores│
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Compare scores to       │
                    │ baseline & expectations │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Test ≈ Random│ │Test ≈ Train  │ │Test >> Train │
        │    Guess     │ │Both Very High│ │  (Overfitting)│
        │  TOO BAD ❌  │ │  TOO GOOD ❌ │ │   NORMAL ⚠️  │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               ▼                ▼                ▼
        [See Section A]  [See Section B]  [See Section C]
```

---

## Section A: Model Too Bad (Underfitting)

### 🔍 Diagnostic Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ SYMPTOM: Test accuracy ≈ random guessing (or barely better)│
│ Example: 52% accuracy on binary classification             │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Step 1: Check if train     │
        │ accuracy is ALSO bad       │
        └─────────────┬──────────────┘
                      │
        ┌─────────────┼─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│Train ≈ Random │           │Train is GOOD  │
│Test ≈ Random  │           │Test is BAD    │
└───────┬───────┘           └───────┬───────┘
        │                           │
        ▼                           ▼
[Data Problem]              [Generalization Problem]
        │                           │
        ▼                           ▼
┌───────────────────────────────────────────────┐
│ A1. DATA QUALITY ISSUES                       │
├───────────────────────────────────────────────┤
│ Check these in order:                         │
│                                               │
│ 1. ❓ Wrong feature encoding?                │
│    └→ Train and test encoded separately?    │
│       Different mappings for same categories?│
│                                               │
│ 2. ❓ Features not scaled?                   │
│    └→ Using distance-based algorithm         │
│       (KNN, SVM, Neural Net)?                │
│                                               │
│ 3. ❓ Missing values not handled?            │
│    └→ Check for NaN in features              │
│       Model silently ignoring rows?          │
│                                               │
│ 4. ❓ Target variable corrupted?             │
│    └→ Check label distribution               │
│       Labels shuffled or misaligned?         │
│                                               │
│ 5. ❓ Features not informative?              │
│    └→ All features constant/random?          │
│       Check correlation with target          │
└───────────────────────────────────────────────┘

┌───────────────────────────────────────────────┐
│ A2. ALGORITHM/HYPERPARAMETER ISSUES           │
├───────────────────────────────────────────────┤
│ Check these:                                  │
│                                               │
│ 1. ❓ Model too simple?                      │
│    └→ Linear model for non-linear problem?   │
│       Try more complex model                 │
│                                               │
│ 2. ❓ Regularization too strong?             │
│    └→ C too small (SVM/LogReg)?              │
│       alpha too large (Ridge/Lasso)?         │
│       Try reducing regularization            │
│                                               │
│ 3. ❓ Learning rate too high?                │
│    └→ Neural network not converging          │
│       Try smaller learning rate              │
│                                               │
│ 4. ❓ Not enough training iterations?        │
│    └→ max_iter reached before convergence    │
│       Increase iterations or epochs          │
└───────────────────────────────────────────────┘
```

### 🛠️ Fix Procedures for "Too Bad"

**Fix 1: Different Encodings**
```python
# ❌ WRONG: Separate encoding creates different mappings
train['cat'] = train['cat'].astype('category').cat.codes  # A:0, B:1, C:2
test['cat'] = test['cat'].astype('category').cat.codes    # A:0, C:1, B:2 ← BROKEN!

# ✅ CORRECT
encoder = LabelEncoder()
train['cat_enc'] = encoder.fit_transform(train['cat'])
test['cat_enc'] = encoder.transform(test['cat'])
```

**Fix 2: Missing Scaling**
```python
# ✅ For KNN/SVM/Neural Nets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Fix 3: Model Too Simple**
```python
# Try: RandomForest, SVC(kernel='rbf'), or MLPClassifier
```

---

## Section B: Model Too Good (Data Leakage)

### 🔍 Diagnostic Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ SYMPTOM: Suspiciously high accuracy (>95% on complex task) │
│ AND train ≈ test (both very high)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Step 1: Check if problem   │
        │ is actually easy           │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │ Is this a toy dataset?     │
        │ (Iris, simple XOR, etc.)   │
        └─────┬──────────────┬───────┘
              │              │
             YES            NO
              │              │
              ▼              ▼
        [Legitimate]    [LEAKAGE SUSPECTED]
        Performance          │
                            ▼
                ┌────────────────────────┐
                │ Check these LEAK POINTS│
                │ in sequential order:   │
                └────────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   [B1. Target         [B2. Features      [B3. Preprocessing
    Leakage]            from Future]       Before Split]
```

### 🔍 Leak Detection Checklist

```
┌──────────────────────────────────────────────────────────────┐
│ B1. TARGET LEAKAGE (Most Severe)                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ ❓ Are any features derived from the target?                │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Examples:                                           │  │
│    │ • df['price_vs_mean'] = price / price.mean()       │  │
│    │   (if 'price' is the target!)                      │  │
│    │                                                     │  │
│    │ • df['high_risk'] = df.groupby('customer_id')      │  │
│    │                        ['default'].transform('mean')│  │
│    │   (if 'default' is the target!)                    │  │
│    │                                                     │  │
│    │ • Target encoding:                                 │  │
│    │   means = df.groupby('category')['target'].mean()  │  │
│    │   (includes test set targets!)                     │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                               │
│ ✅ FIX: Remove features that use target variable             │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│ B2. FUTURE INFORMATION LEAKAGE (Time-based)                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ ❓ Does feature include information from AFTER prediction?  │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Examples:                                           │  │
│    │ • Predicting loan default at time T                │  │
│    │   but using payment_history up to time T+30        │  │
│    │                                                     │  │
│    │ • Predicting customer churn in Jan                 │  │
│    │   but using 'complaints_in_feb' feature            │  │
│    │                                                     │  │
│    │ • Stock price prediction using future prices       │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                               │
│ ✅ FIX: Only use information available BEFORE prediction     │
│                                                               │
│ ⚠️  SPECIAL CASES:                                            │
│    Time-series data:                                         │
│      • Use TimeSeriesSplit for validation                    │
│      • Never shuffle data                                    │
│      • Train on past, test on future                         │
│                                                               │
│    Grouped data (patients, stores, customers):               │
│      • Use GroupKFold or GroupShuffleSplit                   │
│      • Ensure same entity never in both train & test         │
│      • Example: All visits from patient #42 in train OR test │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│ B3. PREPROCESSING LEAKAGE (Order of Operations)              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ ❓ Was preprocessing done BEFORE train/test split?          │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Common mistakes:                                    │  │
│    │                                                     │  │
│    │ ❌ scaler.fit(X_full)  # All data                  │  │
│    │    X_scaled = scaler.transform(X_full)             │  │
│    │    X_train, X_test = split(X_scaled)  # Too late!  │  │
│    │                                                     │  │
│    │ ❌ df.fillna(df.median())  # Uses test medians!    │  │
│    │    X_train, X_test = split(df)                     │  │
│    │                                                     │  │
│    │ ❌ df = remove_outliers(df)  # Uses test stats!    │  │
│    │    X_train, X_test = split(df)                     │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                               │
│ ⚠️  UNDERSTANDING PREPROCESSING LEAKAGE SEVERITY                │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ HIGH-RISK (Statistical Leakage):                    │  │
│    │ • StandardScaler/MinMaxScaler/RobustScaler          │  │
│    │ • Target Encoding/Mean Encoding                     │  │
│    │ • Median/Mean Imputation                            │  │
│    │ • Outlier removal based on statistics               │  │
│    │ PROBLEM: Learns statistics from test data           │  │
│    │ → Wrong mean/std/median in production               │  │
│    │ → Test metrics severely inflated                    │  │
│    │                                                     │  │
│    │ MEDIUM-RISK (Vocabulary Leakage):                   │  │
│    │ • LabelEncoder/OrdinalEncoder/OneHotEncoder         │  │
│    │ PROBLEM: Learns full category vocabulary early      │  │
│    │ → CV scores unrealistically optimistic              │  │
│    │ → Production: unknown categories cause errors       │  │
│    │ → Especially bad with rare/emerging categories     │  │
│    │ IMPACT: Moderate to High (depends on data)         │  │
│    │ FIX: OneHotEncoder(handle_unknown='ignore') helps   │  │
│    │      BUT still fit on train for honest CV scores   │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                               │
│ 📊 LEAKAGE REFERENCE TABLE:                                  │
│    ┌──────────────────────┬────────────┬───────────────────┐ │
│    │ Technique            │ Risk Level │ Why?              │ │
│    ├──────────────────────┼────────────┼───────────────────┤ │
│    │ LabelEncoder         │ ⚠️  Medium │ Inflates CV      │ │
│    │ OrdinalEncoder       │ ⚠️  Medium │ Vocabulary leak  │ │
│    │ OneHotEncoder        │ ⚠️  Medium │ Rare cats leak   │ │
│    │ StandardScaler       │ 🔴 HIGH    │ Stats leak       │ │
│    │ MinMaxScaler         │ 🔴 HIGH    │ Stats leak       │ │
│    │ RobustScaler         │ 🔴 HIGH    │ Stats leak       │ │
│    │ Target Encoding      │ 🔴 HIGH    │ Target leak      │ │
│    │ Imputation (median)  │ 🔴 HIGH    │ Stats leak       │ │
│    │ Outlier removal      │ 🔴 HIGH    │ Stats leak       │ │
│    └──────────────────────┴────────────┴───────────────────┘ │
│                                                               │
│ ⚠️ CRITICAL: ALL preprocessing MUST fit on train only     |
│ Fitting on full data = invalid evaluation = wasted work     │
| MEDIUM-RISK escalates to HIGH when:                         |
|• High-cardinality features (>100 categories)                |
|• Rare categories (<1% frequency)                            |
|• Time-series data (new categories emerge)                   |
|• Production system (unknown categories will appear)         |
|                                                             |
|Example:                                                     |
|• Encoding "DayOfWeek" (7 categories) = truly medium risk    |
|• Encoding "ProductID" (10,000 IDs) = effectively HIGH risk  |
|                                                             |
│                                                             │
├──────────────────────────────────────────────────────────────┤
│ B4. DUPLICATE/OVERLAPPING DATA                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ ❓ Are there duplicate rows?                                │
│    └→ Check: df.duplicated().sum()                          │
│                                                               │
│ ❓ Are train and test from same entities?                   │
│    └→ Example: Same patient appears in both sets            │
│       (different visits but same underlying person)          │
│                                                               │
│ ✅ FIX: Remove duplicates OR group by entity before split    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 🛠️ Fix Procedures for "Too Good"

**Fix 1: Remove Target Leakage**
```python
# ❌ Feature uses target: df['price_ratio'] = price / price.mean()
# ✅ Drop leaky features
X = df.drop(['target', 'price_ratio', 'price_zscore'], axis=1)
```

**Fix 2: Use Pipeline (Best Practice)**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Pipeline fits each step on train only automatically
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

---

## Section C: Normal Overfitting

### 🔍 Diagnostic Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ SYMPTOM: Train accuracy much higher than test accuracy     │
│ Example: Train=92%, Test=78%                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Gap size assessment        │
        └─────────────┬──────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │Gap < 5%  │  │Gap 5-15% │  │Gap > 15% │
  │ Normal   │  │ Moderate │  │ Severe   │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │             │             │
       ▼             ▼             ▼
    [OK]      [Try C1-C3]    [Must Fix C1-C5]
```

### 🛠️ Fix Procedures for Overfitting

```
┌──────────────────────────────────────────────────────────────┐
│ C1. More training data / Data augmentation / SMOTE           │
│ C2. Regularization: Ridge(alpha=1.0), SVC(C=0.1), Dropout    │
│ C3. Simpler model: Reduce max_depth, fewer layers           │
│ C4. Feature selection: SelectKBest, remove redundant features│
│ C5. Cross-validation: Always validate with CV (see below)    │
└──────────────────────────────────────────────────────────────┘

Cross-Validation (Choose by data type):
  
  Standard (i.i.d.): cross_val_score(model, X_train, y_train, cv=5)
  
  Time-series: 
    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_score(model, X_train, y_train, cv=tscv)
  
  Grouped (patients/stores):
    gkf = GroupKFold(n_splits=5)
    cross_val_score(model, X_train, y_train, cv=gkf, groups=train_groups)
  
  ⚠️ Never use KFold for time-series or grouped data!
```

---

## Quick Reference: Expected Performance Ranges

**Classification:**
```
Binary (balanced)     Random: 50%    Good: 80-90%   Suspicious: >95%
Binary (90% negative) Baseline: 90%  Good: 96-98%   Suspicious: >99%
Multi-class (10)      Random: 10%    Good: 70-85%   Suspicious: >90%
```

**Regression:**
```
R²     Good: 0.7-0.9   Excellent: 0.9-0.95   Suspicious: >0.99
RMSE   Good: << std(y) Excellent: ≈ 0        Suspicious: < noise floor
```

⚠️ **Leakage Red Flags:**
- R² > 0.99 on real-world noisy data
- Train ≈ Test AND both very high
- RMSE below measurement noise

---

## Code Template: Complete Diagnostic Workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and prepare features
X = df.drop('target', axis=1)
y = df['target']

# Step 2: Split FIRST (avoid leakage - even for baselines!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Calculate baselines from TRAIN only (no peeking at test!)
baseline_random = 1 / len(y_train.unique())
baseline_majority = y_train.value_counts().max() / len(y_train)

print(f"Baseline (random): {baseline_random:.3f}")
print(f"Baseline (majority): {baseline_majority:.3f}")

# Step 4: Preprocess (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\n" + "="*60)
print("DIAGNOSTIC RESULTS:")
print("="*60)
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")
print(f"Gap:            {(train_score - test_score):.3f}")

# Step 7: Diagnose
print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if test_score <= baseline_random + 0.05:
    print("❌ MODEL TOO BAD - Performance ≈ random guessing")
    print("   Check: data quality, encoding, scaling, model choice")
    
elif test_score > 0.95 and abs(train_score - test_score) < 0.03:
    print("⚠️  MODEL TOO GOOD - Suspiciously high accuracy")
    print("   Check: target leakage, preprocessing order, duplicates")
    
elif train_score - test_score > 0.15:
    print("⚠️  SEVERE OVERFITTING - Train >> Test")
    print("   Check: model complexity, regularization, feature selection")
    
elif train_score - test_score > 0.05:
    print("⚠️  MODERATE OVERFITTING")
    print("   Consider: regularization, cross-validation")
    
else:
    print("✅ NORMAL PERFORMANCE - Proceed with tuning")

# Step 8: Cross-validation for confirmation (ALWAYS DO THIS!)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print("⚠️  Single train-test splits can be misleading - always validate with CV!")
```

---

## Summary: Decision Priorities

1. **Test ≈ random?** → Check data quality, encoding, scaling, model choice
2. **Test ≈ train AND both very high?** → Data leakage (target, preprocessing, duplicates)
3. **Train >> test?** → Overfitting (regularize, simplify, more data)
4. **Always use cross-validation** → Single splits are unreliable

**Golden Rules:** 
- Split FIRST, preprocess SECOND (fit on train, transform on test)
- Use Pipeline to enforce correct order automatically
- All preprocessing (encoders AND scalers) must fit on train only
- Not following these rules = invalid evaluation
```
