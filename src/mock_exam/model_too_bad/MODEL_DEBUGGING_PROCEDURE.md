# Model Debugging & Train-Test Split Procedures

## Table of Contents
1. [Procedure for Checking if Model is Too Bad or Too Good](#procedure-for-checking-model)
2. [Procedure for Correct Train-Test Split](#procedure-for-correct-train-test-split)
3. [Common Pitfalls Checklist](#common-pitfalls-checklist)

---

## Procedure for Checking if Model is Too Bad or Too Good

### Step 1: Establish Baseline Metrics

**Calculate theoretical baselines:**
```python
# For classification:
# - Random guessing: 1 / n_classes
# - Majority class: max(class_counts) / total_samples

baseline_random = 1 / len(np.unique(y))
baseline_majority = y.value_counts().max() / len(y)

print(f"Random baseline: {baseline_random:.3f}")
print(f"Majority baseline: {baseline_majority:.3f}")
```

**Expected performance ranges:**
- **Too Bad**: Accuracy ≈ random baseline (or worse!)
- **Suspiciously Good**: Accuracy > 0.95 (unless problem is trivial)
- **Reasonable**: Somewhere between baselines and realistic upper bound

### Step 2: Check for "Model Too Bad" Issues

#### 2.1 Data Leakage in REVERSE (Data Corruption)

**Symptoms:**
- Accuracy ≈ random guessing
- Training accuracy is also poor

**Check:**
```python
# ❌ BAD: Are train and test using different encodings?
print("Train unique values:", X_train['feature'].unique())
print("Test unique values:", X_test['feature'].unique())

# Check if encoding was done separately
# Look for pd.factorize(), LabelEncoder.fit(), etc. called twice
```

**Common causes:**
- Separate encoding on train and test (like `model_bad2.py`)
- Features scaled/normalized independently
- Target variable encoded differently

**Fix:**
```python
# ✓ GOOD: Fit on train, transform on test
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X_train_encoded = encoder.fit_transform(X_train['feature'])
X_test_encoded = encoder.transform(X_test['feature'])  # Use same mapping
```

#### 2.2 Feature-Target Mismatch

**Symptoms:**
- Model trains but predictions are random

**Check:**
```python
# Verify features actually correlate with target
import seaborn as sns
import matplotlib.pyplot as plt

# For numerical features
correlation = pd.concat([X_train, y_train], axis=1).corr()
print(correlation[y_train.name].sort_values(ascending=False))

# For categorical features
pd.crosstab(X_train['categorical_feature'], y_train)
```

**Fix:**
- Ensure you're using relevant features
- Check for data loading errors (wrong columns, shuffled indices)

#### 2.3 Wrong Train-Test Split Strategy

**Symptoms:**
- Train has different distribution than test

**Check:**
```python
# Compare distributions
print("Train class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True))

# For time series: check temporal ordering
print("Train date range:", df_train['date'].min(), "to", df_train['date'].max())
print("Test date range:", df_test['date'].min(), "to", df_test['date'].max())
```

**Fix:**
```python
# For imbalanced data, use stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# For time series, ensure temporal ordering
# (don't use random split!)
```

### Step 3: Check for "Model Too Good" Issues

#### 3.1 Data Leakage

**Symptoms:**
- Test accuracy > 0.95 (suspiciously high)
- Test accuracy > Train accuracy (major red flag!)

**Check:**
```python
# ❌ BAD: Did you fit transformers on ALL data?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← Leakage!
X_train, X_test = train_test_split(X_scaled, ...)

# ❌ BAD: Is test data leaking into train?
print("Train indices:", X_train.index)
print("Test indices:", X_test.index)
print("Overlap:", set(X_train.index) & set(X_test.index))  # Should be empty!
```

**Common causes:**
- Scaling/normalization before split
- Feature engineering using full dataset statistics
- Duplicate rows in dataset
- Target leakage (target info in features)

**Fix:**
```python
# ✓ GOOD: Split FIRST, then fit on train only
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

#### 3.2 Target Leakage

**Symptoms:**
- Near-perfect accuracy
- Features that "shouldn't" be that predictive

**Check:**
```python
# Look for features that are derived from target
# or contain future information

# Check feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
# Are top features suspiciously perfect predictors?
```

**Fix:**
- Remove features that leak target information
- Check temporal ordering (no future data in past predictions)

#### 3.3 Test Set Too Small or Unrepresentative

**Symptoms:**
- High variance in test scores across different splits

**Check:**
```python
from sklearn.model_selection import cross_val_score

# Check stability across folds
scores = cross_val_score(model, X, y, cv=5)
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**Fix:**
- Use larger test set (typically 20-30%)
- Use cross-validation for robust evaluation

### Step 4: Debugging Checklist

Run through this checklist systematically:

```python
def debug_model(X_train, X_test, y_train, y_test, model, predictions):
    """
    Comprehensive model debugging procedure
    """
    print("="*60)
    print("MODEL DEBUGGING REPORT")
    print("="*60)
    
    # 1. Basic statistics
    print("\n1. DATASET STATISTICS")
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    # 2. Class distribution
    print("\n2. CLASS DISTRIBUTION")
    print("   Train:", dict(pd.Series(y_train).value_counts(normalize=True)))
    print("   Test:", dict(pd.Series(y_test).value_counts(normalize=True)))
    
    # 3. Check for overlap
    print("\n3. DATA INTEGRITY")
    if hasattr(X_train, 'index') and hasattr(X_test, 'index'):
        overlap = set(X_train.index) & set(X_test.index)
        print(f"   Index overlap: {len(overlap)} (should be 0)")
    
    # 4. Baseline metrics
    print("\n4. BASELINE METRICS")
    n_classes = len(np.unique(y_train))
    random_baseline = 1.0 / n_classes
    majority_class = pd.Series(y_train).value_counts(normalize=True).max()
    print(f"   Random guessing: {random_baseline:.3f}")
    print(f"   Majority class: {majority_class:.3f}")
    
    # 5. Model performance
    print("\n5. MODEL PERFORMANCE")
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, predictions)
    print(f"   Train accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")
    print(f"   Difference: {abs(train_acc - test_acc):.3f}")
    
    # 6. Diagnosis
    print("\n6. DIAGNOSIS")
    if test_acc <= random_baseline * 1.1:
        print("   ⚠️  MODEL TOO BAD: At random baseline!")
        print("   → Check for encoding mismatches")
        print("   → Verify feature-target alignment")
        print("   → Check data loading process")
    elif test_acc > 0.95 and train_acc < 0.98:
        print("   ⚠️  POSSIBLE DATA LEAKAGE: Test too good!")
        print("   → Check if transformers were fit on all data")
        print("   → Look for target leakage in features")
        print("   → Verify train-test split procedure")
    elif test_acc > train_acc + 0.05:
        print("   🚨 MAJOR RED FLAG: Test > Train!")
        print("   → Definite data leakage")
        print("   → Check split procedure immediately")
    elif abs(train_acc - test_acc) > 0.15:
        print("   ⚠️  HIGH VARIANCE: Large train-test gap")
        print("   → Model may be overfitting")
        print("   → Consider regularization or simpler model")
    elif train_acc > 0.99 and test_acc > 0.95:
        print("   ⚠️  TOO GOOD TO BE TRUE")
        print("   → Unless problem is trivial, check for leakage")
        print("   → Verify features don't contain target info")
    else:
        print("   ✓ Performance seems reasonable")
        print(f"   → Test accuracy is {test_acc/random_baseline:.1f}x better than random")
    
    print("\n" + "="*60)

# Usage:
# debug_model(X_train, X_test, y_train, y_test, model, predictions)
```

---

## Procedure for Correct Train-Test Split

### The Golden Rule
**Split FIRST, Fit ONLY on Train, Transform BOTH**

### Step-by-Step Procedure

#### Step 1: Initial Data Split

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           # Maintain class distribution (for classification)
)

# ✓ At this point, consider test set "locked away"
```

**Key parameters:**
- `test_size`: 0.2-0.3 (20-30%) is typical
- `random_state`: Always set for reproducibility
- `stratify`: Use for classification with imbalanced classes
- `shuffle`: Default True (but False for time series!)

#### Step 2: Exploratory Data Analysis (EDA)

```python
# ✓ GOOD: Only look at TRAINING data during EDA
print("Training data shape:", X_train.shape)
print("\nTraining data summary:")
print(X_train.describe())

print("\nMissing values:")
print(X_train.isnull().sum())

print("\nClass distribution:")
print(y_train.value_counts(normalize=True))

# DO NOT look at X_test or y_test during EDA!
```

#### Step 3: Fit Preprocessing on Train Only

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Example: Multiple preprocessing steps

# 3a. Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)  # Fit on train
X_test_imputed = imputer.transform(X_test)        # Only transform test

# 3b. Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)  # Fit on train
X_test_scaled = scaler.transform(X_test_imputed)        # Only transform test

# 3c. Encode categorical variables
encoder = LabelEncoder()
if 'category' in X_train.columns:
    X_train['category_encoded'] = encoder.fit_transform(X_train['category'])
    X_test['category_encoded'] = encoder.transform(X_test['category'])
```

**Critical rules:**
- ✓ `.fit()` or `.fit_transform()` on **train** only
- ✓ `.transform()` on **test** only
- ❌ Never `.fit()` on test data
- ❌ Never `.fit_transform()` on test data

#### Step 4: Feature Engineering

```python
# ✓ GOOD: Calculate statistics from TRAIN only
train_mean = X_train['feature'].mean()
train_std = X_train['feature'].std()

# Apply same transformation to both sets
X_train['normalized'] = (X_train['feature'] - train_mean) / train_std
X_test['normalized'] = (X_test['feature'] - train_mean) / train_std

# ❌ BAD: Don't do this!
# full_mean = X['feature'].mean()  # Uses test data!
# X_train['normalized'] = (X_train['feature'] - full_mean) / X['feature'].std()
```

#### Step 5: Train Model

```python
from sklearn.linear_model import LogisticRegression

# Train on training data only
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Never fit on X_test!
```

#### Step 6: Evaluate Model

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 6a. Training performance (sanity check)
train_predictions = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training accuracy: {train_accuracy:.3f}")

# 6b. Test performance (THE REAL METRIC)
test_predictions = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test accuracy: {test_accuracy:.3f}")

# 6c. Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))
```

#### Step 7: Cross-Validation (Optional but Recommended)

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Create a pipeline to ensure correct preprocessing in each fold
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Cross-validation on TRAINING data only
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Final evaluation on held-out test set
pipeline.fit(X_train, y_train)
final_score = pipeline.score(X_test, y_test)
print(f"Final test score: {final_score:.3f}")
```

### Complete Example: Correct Procedure

```python
"""
Complete example of correct train-test split procedure
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_your_data()  # Your data loading function

# ============================================================
# STEP 1: SPLIT FIRST (test set is now "locked away")
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================================
# STEP 2: EDA on TRAINING data only
# ============================================================
print("\n=== Training Data Summary ===")
print(X_train.describe())
print("\nClass distribution:")
print(y_train.value_counts(normalize=True))

# ============================================================
# STEP 3: Build preprocessing pipeline (best practice)
# ============================================================
# Using Pipeline ensures preprocessing is done correctly
preprocessing_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
]

model_steps = [
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
]

pipeline = Pipeline(preprocessing_steps + model_steps)

# ============================================================
# STEP 4: Cross-validation on training data
# ============================================================
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                            scoring='accuracy')
print(f"CV scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ============================================================
# STEP 5: Train final model on full training set
# ============================================================
pipeline.fit(X_train, y_train)

# ============================================================
# STEP 6: Evaluate on test set (ONLY ONCE!)
# ============================================================
print("\n=== Final Evaluation on Test Set ===")
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
print(f"Difference: {abs(train_score - test_score):.3f}")

# Detailed test set metrics
test_predictions = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

# ============================================================
# STEP 7: Sanity checks
# ============================================================
print("\n=== Sanity Checks ===")

# Check 1: No overlap between train and test indices
if hasattr(X_train, 'index'):
    overlap = set(X_train.index) & set(X_test.index)
    print(f"✓ Index overlap: {len(overlap)} (should be 0)")

# Check 2: Test accuracy reasonable compared to baselines
n_classes = len(np.unique(y_train))
random_baseline = 1.0 / n_classes
print(f"✓ Random baseline: {random_baseline:.3f}")
print(f"✓ Test is {test_score/random_baseline:.1f}x better than random")

# Check 3: Train-test gap not too large
if abs(train_score - test_score) < 0.15:
    print("✓ Train-test gap is acceptable")
else:
    print("⚠️  Large train-test gap - possible overfitting")

# Check 4: Test not better than train (would indicate leakage)
if test_score <= train_score + 0.02:
    print("✓ Test ≤ Train (no obvious leakage)")
else:
    print("🚨 Test > Train - CHECK FOR DATA LEAKAGE!")
```

---

## Common Pitfalls Checklist

### ❌ Things That Will Break Your Model

1. **Fitting on ALL data before split**
   ```python
   # ❌ WRONG
   X_scaled = scaler.fit_transform(X)
   X_train, X_test = train_test_split(X_scaled, y)
   ```

2. **Fitting transformers on test data**
   ```python
   # ❌ WRONG
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.fit_transform(X_test)  # ← Re-fitting!
   ```

3. **Using test statistics for train data**
   ```python
   # ❌ WRONG
   X_train['normalized'] = (X_train['feature'] - X_test['feature'].mean())
   ```

4. **Index overlap (not resetting indices after split)**
   ```python
   # ❌ WRONG
   X_train = X.iloc[:800]
   X_test = X.iloc[:200]  # Overlap with train!
   ```

5. **Not using stratification for imbalanced data**
   ```python
   # ❌ RISKY (might get different class distributions)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

6. **Looking at test data during EDA**
   ```python
   # ❌ WRONG (introduces bias)
   print(X_test.describe())  # Don't look!
   print(y_test.value_counts())  # Don't look!
   ```

7. **Tuning hyperparameters on test set**
   ```python
   # ❌ WRONG (test set should be untouched until final evaluation)
   for param in params:
       model.set_params(**param)
       model.fit(X_train, y_train)
       score = model.score(X_test, y_test)  # ← Should use validation set!
   ```

### ✓ Things You Should Always Do

1. **Split first, then fit**
2. **Use Pipeline for preprocessing**
3. **Set random_state for reproducibility**
4. **Use stratification for classification**
5. **Perform cross-validation on training data**
6. **Check baselines (random, majority class)**
7. **Verify no index overlap**
8. **Compare train vs test performance**
9. **Use test set ONLY for final evaluation**
10. **Document your preprocessing steps**

---

## Quick Reference: The Correct Order

```
1. Load data
2. Split into train/test → TEST SET IS NOW "LOCKED"
3. EDA on TRAIN only
4. Fit preprocessing on TRAIN → Transform BOTH
5. Fit model on TRAIN
6. Cross-validate on TRAIN (optional but recommended)
7. Evaluate on TEST (only once at the end!)
8. Sanity checks
```

**Remember: Your test set should be a surprise exam, not a practice test!**

