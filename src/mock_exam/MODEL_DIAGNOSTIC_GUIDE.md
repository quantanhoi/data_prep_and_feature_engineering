# Model Diagnostic Guide: Too Bad or Too Good?

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

#### Fix 1: Feature Encoding Issues

**Problem:** Train and test have different encodings

```python
# ❌ WRONG: Separate encoding
train['category'] = train['category'].astype('category').cat.codes
test['category'] = test['category'].astype('category').cat.codes
# → Train: {A:0, B:1, C:2}
# → Test:  {A:0, C:1, B:2}  ← DIFFERENT!

# ✅ CORRECT: Fit encoder once, use for both
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train['category'])  # Fit on train only

train['category_encoded'] = encoder.transform(train['category'])
test['category_encoded'] = encoder.transform(test['category'])
# Both have same mapping!
```

#### Fix 2: Missing Scaling

**Problem:** Features on different scales (KNN, SVM, Neural Nets)

```python
# Check feature ranges
print(X_train.describe())
#        age    income      balance
# mean   35     50000      5000
# std    10     20000      10000  ← Very different scales!

# ✅ FIX: Add scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain model
model.fit(X_train_scaled, y_train)
```

#### Fix 3: Model Too Simple

**Problem:** Linear model for non-linear problem

```python
# ❌ WRONG: Linear model for complex patterns
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# ✅ FIX: Try non-linear model
from sklearn.ensemble import RandomForestClassifier
# or from sklearn.svm import SVC with rbf kernel
# or from sklearn.neural_network import MLPClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
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
│ ✅ FIX: SPLIT FIRST, then preprocess                         │
│                                                               │
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

#### Fix 1: Remove Target Leakage

```python
# ❌ WRONG: Feature uses target
df['price_ratio'] = df['median_house_value'] / df['median_house_value'].mean()
# If median_house_value IS the target, this is leakage!

# ✅ FIX: Remove the leaky feature
X = df.drop(['median_house_value', 'price_ratio'], axis=1)
y = df['median_house_value']
```

#### Fix 2: Fix Preprocessing Order

```python
# ❌ WRONG ORDER
df_imputed = df.fillna(df.median())  # ← Uses ALL data
X = df_imputed.drop('target', axis=1)
y = df_imputed['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ✅ CORRECT ORDER
X = df.drop('target', axis=1)
y = df['target']

# 1. SPLIT FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Then preprocess
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)      # Fit on train
X_test_imputed = imputer.transform(X_test)            # Transform test
```

#### Fix 3: Use Pipeline (Recommended)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Pipeline ensures correct order automatically
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Pipeline fits each step on train only
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
│ C1. INCREASE TRAINING DATA                                    │
├──────────────────────────────────────────────────────────────┤
│ More data helps model learn general patterns                 │
│                                                               │
│ • Collect more samples                                       │
│ • Data augmentation (images, text)                           │
│ • Synthetic data generation (SMOTE for imbalanced)           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ C2. ADD REGULARIZATION                                        │
├──────────────────────────────────────────────────────────────┤
│ Penalize model complexity                                     │
│                                                               │
│ Linear Models:                                                │
│   from sklearn.linear_model import Ridge, Lasso              │
│   model = Ridge(alpha=1.0)  # Increase alpha                 │
│                                                               │
│ SVM:                                                          │
│   from sklearn.svm import SVC                                │
│   model = SVC(C=0.1)  # Decrease C                           │
│                                                               │
│ Neural Networks:                                              │
│   from sklearn.neural_network import MLPClassifier           │
│   model = MLPClassifier(alpha=0.01)  # L2 penalty            │
│                                                               │
│ Deep Learning (Keras/PyTorch):                                │
│   • Add Dropout layers                                       │
│   • Add L1/L2 regularization to layers                       │
│   • Early stopping                                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ C3. REDUCE MODEL COMPLEXITY                                   │
├──────────────────────────────────────────────────────────────┤
│ Simpler model → less overfitting                             │
│                                                               │
│ Random Forest:                                                │
│   • Reduce max_depth: max_depth=5 instead of None            │
│   • Increase min_samples_split: min_samples_split=20         │
│   • Reduce n_estimators: n_estimators=50 instead of 200      │
│                                                               │
│ Neural Networks:                                              │
│   • Reduce number of layers                                  │
│   • Reduce neurons per layer                                 │
│                                                               │
│ Polynomial Features:                                          │
│   • Lower degree: degree=2 instead of degree=3               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ C4. FEATURE SELECTION                                         │
├──────────────────────────────────────────────────────────────┤
│ Remove irrelevant/redundant features                         │
│                                                               │
│ from sklearn.feature_selection import SelectKBest, f_classif │
│                                                               │
│ selector = SelectKBest(f_classif, k=10)                      │
│ X_train_selected = selector.fit_transform(X_train, y_train)  │
│ X_test_selected = selector.transform(X_test)                 │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ C5. CROSS-VALIDATION                                          │
├──────────────────────────────────────────────────────────────┤
│ Better estimate of true performance                          │
│                                                               │
│ from sklearn.model_selection import cross_val_score          │
│                                                               │
│ scores = cross_val_score(model, X_train, y_train, cv=5)      │
│ print(f"CV Mean: {scores.mean():.3f} (+/- {scores.std():.3f})│
└──────────────────────────────────────────────────────────────┘
```

---

## Complete Diagnostic Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                START: Train Model & Evaluate                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Calculate Baseline:         │
        │ • Random: 1/n_classes       │
        │ • Majority: max_count/total │
        └─────────────┬───────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Get train_score, test_score │
        └─────────────┬───────────────┘
                      │
        ┌─────────────┼─────────────────────────┐
        │             │                         │
        ▼             ▼                         ▼
┌───────────────┐ ┌──────────────┐ ┌────────────────────┐
│test ≈ random? │ │test ≈ train? │ │train >> test?      │
│               │ │(both high)?   │ │                    │
│  YES          │ │  YES          │ │  YES               │
└───────┬───────┘ └──────┬───────┘ └─────────┬──────────┘
        │                │                    │
        │                │                    │
        ▼                ▼                    ▼
┌───────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC PATHS                        │
├───────────────┬───────────────────┬───────────────────────┤
│               │                   │                       │
│  TOO BAD      │    TOO GOOD       │    OVERFITTING        │
│               │                   │                       │
│ 1. Check data │ 1. Target leakage │ 1. More data          │
│    quality    │    • Features use │ 2. Regularization     │
│               │      target       │ 3. Simpler model      │
│ 2. Check      │                   │ 4. Feature selection  │
│    encoding   │ 2. Future info    │ 5. Cross-validation   │
│               │    leakage        │                       │
│ 3. Check      │                   │                       │
│    scaling    │ 3. Preprocessing  │                       │
│               │    before split   │                       │
│ 4. Check      │    • Imputer      │                       │
│    model      │    • Scaler       │                       │
│    choice     │    • Feature eng  │                       │
│               │                   │                       │
│ 5. Check      │ 4. Duplicates     │                       │
│    hyperparams│    • Same entity  │                       │
│               │      in both sets │                       │
└───────────────┴───────────────────┴───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Fix issues, retrain, repeat │
        └─────────────────────────────┘
```

---

## Quick Reference: Expected Performance Ranges

### Classification Tasks

```
Problem Type          Random    Acceptable   Good    Suspicious
─────────────────────────────────────────────────────────────────
Binary (balanced)     50%       65-75%       80-90%  >95%
Binary (imbalanced)   90%*      92-95%       96-98%  >99%
Multi-class (10)      10%       40-60%       70-85%  >90%
Multi-class (100)     1%        20-40%       50-70%  >85%

* If 90% are negative class, predicting all negative = 90% accuracy
```

### Regression Tasks

```
Metric      Terrible   Poor      Acceptable   Good    Excellent
────────────────────────────────────────────────────────────────
R²          <0.3       0.3-0.5   0.5-0.7      0.7-0.9 >0.9
RMSE        Check if close to mean(y) → bad
            Much smaller than mean(y) → good
```

---

## Common Patterns to Recognize

### Pattern 1: The "Perfect" Model (99.9% accuracy)

```
Diagnosis: DATA LEAKAGE
Checks:
  ✓ Target in features?
  ✓ Preprocessing before split?
  ✓ Duplicate rows?
```

### Pattern 2: The "Random Guesser" (50% on binary)

```
Diagnosis: DATA QUALITY or WRONG MODEL
Checks:
  ✓ Features scaled? (if using KNN/SVM)
  ✓ Encoding consistent?
  ✓ Features informative?
```

### Pattern 3: The "Train Champion" (Train 99%, Test 70%)

```
Diagnosis: OVERFITTING
Checks:
  ✓ Model too complex?
  ✓ Need regularization?
  ✓ Enough training data?
```

### Pattern 4: The "Gradual Learner" (Train 75%, Test 73%)

```
Diagnosis: NORMAL - Slight underfitting
Checks:
  ✓ Try more complex model
  ✓ Add features
  ✓ Reduce regularization
```

---

## Code Template: Complete Diagnostic Workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and split data
X = df.drop('target', axis=1)
y = df['target']

# Calculate baselines
baseline_random = 1 / len(y.unique())
baseline_majority = y.value_counts().max() / len(y)

print(f"Baseline (random): {baseline_random:.3f}")
print(f"Baseline (majority): {baseline_majority:.3f}")

# Step 2: Split FIRST (avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Preprocess (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\n" + "="*60)
print("DIAGNOSTIC RESULTS:")
print("="*60)
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")
print(f"Gap:            {(train_score - test_score):.3f}")

# Step 6: Diagnose
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

# Step 7: Cross-validation for confirmation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

---

## Summary: Decision Priorities

1. **First, check if test ≈ random** → DATA QUALITY issues
2. **Then, check if test ≈ train AND both very high** → DATA LEAKAGE
3. **Then, check if train >> test** → OVERFITTING
4. **Finally, tune and optimize** → NORMAL workflow

**Remember:** 
- Split FIRST, preprocess SECOND
- Fit on train, transform on test
- Use pipelines to avoid mistakes
- Baselines help identify "too bad" and "too good"
