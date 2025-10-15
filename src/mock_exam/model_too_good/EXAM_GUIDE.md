# Data Leakage Detection Exam - Complete Guide

## 📋 Overview

You've inherited a model that achieves **99.8% accuracy** on both training and test sets! Your manager is suspicious and asks you to investigate. This exam tests your ability to detect and fix data leakage issues.

**Scenario:** A junior data scientist trained this model and thinks they've made a breakthrough. You need to find what went wrong.

**Reality:** After fixing all bugs, accuracy should drop to ~75-80% (realistic and trustworthy performance).

**Time Estimate:** 45 minutes

---

## 🚀 Quick Start

### Step 1: Generate the Dataset
```bash
cd src/mock_exam/model_too_good
python generate_dataset.py
```

### Step 2: Choose Your Path

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook data_leakage_exam.ipynb
```

**Option B: Python Scripts**
```bash
python leaky_model.py   # See the bugs in action (99.8% accuracy!)
# Find and fix the bugs...
python solution.py      # Compare with correct version (~78% accuracy)
```

### Step 3: Your Mission
1. Find all **7 data leakage bugs** in the code
2. Explain what each bug does wrong
3. Fix the code using sklearn Pipeline
4. Achieve realistic performance (~75-80% accuracy)

---

## 🐛 The 7 Data Leakage Bugs

### Bug #1: Feature Engineering on Full Dataset ⭐⭐⭐

**The Code:**
```python
df['price_vs_mean'] = df['median_house_value'] / df['median_house_value'].mean()
df['price_zscore'] = (df['median_house_value'] - df['median_house_value'].mean()) / df['median_house_value'].std()
```

**Why It's Wrong:**
- Calculates mean and std from the **ENTIRE dataset** (including test data)
- When scaling test data, it uses statistics that **include test information**
- The model indirectly "sees" test data patterns through these statistics

**Example:**
```
Full dataset: [100, 200, 300, 400, 500]
Mean = 300 (calculated from ALL data, including future test data)

Train data: [100, 200, 300]
Test data: [400, 500]  ← These influenced the mean of 300!
```

**The Fix:**
```python
# Split FIRST
X_train, X_test = train_test_split(X)

# Compute statistics ONLY from training data
train_mean = X_train['median_house_value'].mean()
train_std = X_train['median_house_value'].std()

# Apply to both using TRAIN statistics
X_train['price_vs_mean'] = X_train['median_house_value'] / train_mean
X_test['price_vs_mean'] = X_test['median_house_value'] / train_mean
```

**Better:** Use Pipeline with custom transformer

---

### Bug #2: Target Encoding Before Split ⭐⭐⭐⭐⭐

**The Code:**
```python
ocean_proximity_means = df.groupby('ocean_proximity')['high_value'].mean()
df['ocean_proximity_target_encoded'] = df['ocean_proximity'].map(ocean_proximity_means)
```

**Why It's Wrong:**
- Uses the **TARGET VARIABLE** ('high_value') to create a feature
- Computes means from the entire dataset including test set
- Each category's encoding includes information from test set labels
- **This is SEVERE target leakage!**

**Example:**
```
Full dataset:
ocean_proximity | high_value
INLAND         | 0
INLAND         | 1  ← Test row, shouldn't influence encoding!
INLAND         | 0

Encoding: INLAND → mean([0, 1, 0]) = 0.33
But the test row (value=1) influenced this encoding!
```

**The Fix:**
- Remove target encoding entirely, OR
- Use proper CV-based target encoding with libraries like `category_encoders`
- **Never use test set labels** for encoding

```python
# Better: Use standard one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
```

---

### Bug #3: Imputation Before Split ⭐⭐⭐

**The Code:**
```python
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```

**Why It's Wrong:**
- Computes median/mode from the **entire dataset**
- Test set missing values are filled using statistics that **include test data**
- The imputer "learns" from test set

**Example:**
```
Full dataset: [10, 20, NaN, 40, 50]
Median = 30 (calculated from [10, 20, 40, 50])

After split:
Train: [10, 20, NaN]  ← Filled with median=30
Test: [40, 50]  ← These values influenced median=30!
```

**The Fix:**
```python
# Split FIRST
X_train, X_test = train_test_split(X)

# Fit imputer ONLY on training data
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)

# Transform both
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

---

### Bug #4: Scaling Before Split ⭐⭐⭐

**The Code:**
```python
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

**Why It's Wrong:**
- StandardScaler learns **mean and std** from entire dataset
- Test data is scaled using mean/std that **includes test information**
- One of the **most common** data leakage mistakes!

**Mathematical Impact:**
```
StandardScaler formula: (x - μ) / σ

Wrong way:
μ = mean(all_data)  ← Includes test data!
σ = std(all_data)   ← Includes test data!

Correct way:
μ = mean(train_data)  ← Only training data
σ = std(train_data)   ← Only training data
```

**The Fix:**
```python
# Split FIRST
X_train, X_test = train_test_split(X)

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test using TRAIN statistics
X_test_scaled = scaler.transform(X_test)
```

---

### Bug #5: Target Leakage in Features ⭐⭐⭐⭐⭐

**The Code:**
```python
suspicious_features = ['median_house_value', 'price_vs_mean', 'price_zscore', 
                       'ocean_proximity_target_encoded']
```

**Why It's Wrong:**
- `median_house_value` **DIRECTLY determines** the target `high_value`
- Target is defined as: `high_value = (median_house_value > threshold)`
- Including this feature means the model **sees the answer**!
- `price_vs_mean` and `price_zscore` are derived from the target
- These features **wouldn't exist in production**!

**Example:**
```python
# Target creation:
threshold = df['median_house_value'].quantile(0.75)
df['high_value'] = (df['median_house_value'] > threshold).astype(int)

# Then using median_house_value as a feature:
X = df[['median_house_value', ...]]  # ← This IS the target!
y = df['high_value']

# Model learns: if median_house_value > threshold → high_value = 1
# Of course it gets 99% accuracy!
```

**Production Reality:**
```
Prediction time:
- User input: bedrooms, location, age
- You DON'T have: median_house_value (that's what you're trying to predict!)
```

**The Fix:**
```python
# Only use features available at prediction time
X = df[['latitude', 'longitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income', 'ocean_proximity']]

# Remove ALL target-derived features
```

---

### Bug #6: Inconsistent Categorical Encoding ⭐⭐

**The Code:**
```python
# First: Target encoding on full dataset (Bug #2)
ocean_proximity_means = df.groupby('ocean_proximity')['high_value'].mean()
df['ocean_proximity_target_encoded'] = df['ocean_proximity'].map(ocean_proximity_means)

# Later: OneHotEncoder fitted on just training data
encoder = OneHotEncoder()
encoder.fit(X_train[['ocean_proximity']])
```

**Why It's Wrong:**
- Same categorical variable treated **two different ways**
- First used for target encoding (on full data)
- Then used for one-hot encoding (on train only)
- Inconsistent and confusing methodology

**The Fix:**
```python
# Choose ONE encoding method and stick with it
# Use it consistently within a Pipeline

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
# Fit on train, transform both train and test
```

---

### Bug #7: Meaningless Cross-Validation ⭐⭐⭐

**The Code:**
```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
```

**Why It's Wrong:**
- While CV is run on training data only, the data has **already been preprocessed**
- Preprocessing used statistics from the **full dataset** (including test)
- CV scores are **inflated and meaningless**
- Should use Pipeline with CV to properly separate folds

**Example:**
```
Wrong workflow:
1. Preprocess ALL data (leakage!)
2. Split into train/test
3. Run CV on train  ← CV scores look great but are meaningless!

Correct workflow:
1. Split into train/test
2. Create Pipeline (preprocess + model)
3. Run CV on Pipeline ← Each fold does its own preprocessing!
```

**The Fix:**
```python
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# CV on the pipeline
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

---

## 📊 Expected Results

### Before Fixes (Leaky Model)
```
Training Accuracy:  99.80%  ❌ TOO GOOD!
Test Accuracy:      99.80%  ❌ TOO GOOD!
CV Accuracy:        99.80%  ❌ TOO GOOD!

Top Features:
1. median_house_value      ← DIRECTLY LEAKS TARGET!
2. price_vs_mean           ← Derived from target
3. ocean_proximity_encoded ← Target encoded
```

**Why suspiciously high?**
- Almost no gap between train and test
- Real-world problems rarely achieve >95% accuracy
- Model essentially "saw" the answers

### After Fixes (Proper Model)
```
Training Accuracy:  82.34%  ✅ Realistic
Test Accuracy:      78.56%  ✅ Realistic
CV Accuracy:        77.91%  ✅ Realistic

Top Features:
1. median_income           ← Legitimate predictor
2. latitude                ← Geographic information
3. housing_median_age      ← Age of property
```

**Why realistic?**
- Small gap between train and test (healthy overfitting)
- ~75-80% is reasonable for housing prediction
- Features make logical sense

---

## 🎯 Visual Guide: The Wrong vs. Right Way

### ❌ WRONG: The Leaky Pipeline

```
┌─────────────────────────────────┐
│      FULL DATASET               │
│  (train + test mixed together)  │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  1. Fit Scaler on FULL DATASET  │ ❌ LEAK!
│     (learns mean, std from all) │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  2. Fit Imputer on FULL DATASET │ ❌ LEAK!
│     (learns median from all)    │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  3. Create features using       │ ❌ LEAK!
│     statistics from FULL DATA   │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  4. NOW split into train/test   │ ❌ TOO LATE!
│     (data already contaminated) │
└─────────────────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
  ┌─────────┐ ┌─────────┐
  │  Train  │ │  Test   │
  │ (leaked)│ │(leaked) │
  └─────────┘ └─────────┘
        │           │
        └─────┬─────┘
              ▼
┌─────────────────────────────────┐
│  5. Train model                 │
│     Result: 99.8% accuracy      │ ❌ TOO GOOD!
└─────────────────────────────────┘
```

### ✅ CORRECT: The Leak-Free Pipeline

```
┌─────────────────────────────────┐
│         FULL DATASET            │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  1. SPLIT FIRST!                │ ✓ CORRECT!
│     train_test_split()          │
└─────────────────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
  ┌─────────┐ ┌─────────┐
  │  Train  │ │  Test   │
  │  (80%)  │ │  (20%)  │
  └─────────┘ └─────────┘
        │           │
        │           │ (stays separate)
        ▼           │
┌─────────────┐     │
│ 2. Fit      │     │
│    Imputer  │     │
│    on TRAIN │ ✓   │
└─────────────┘     │
        │           │
        ▼           ▼
┌─────────────┐ ┌─────────────┐
│ 3. Transform│ │ 3. Transform│
│    TRAIN    │ │    TEST     │
│   (fitted)  │ │ (not fitted)│
└─────────────┘ └─────────────┘
        │           │
        ▼           ▼
┌─────────────┐ ┌─────────────┐
│ 4. Fit      │ │ 4. Transform│
│    Scaler   │ │    TEST     │
│    on TRAIN │ │             │
└─────────────┘ └─────────────┘
        │           │
        ▼           │
┌─────────────┐     │
│ 5. Train    │     │
│    Model    │     │
└─────────────┘     │
        │           │
        └─────┬─────┘
              ▼
┌─────────────────────────────────┐
│  6. Evaluate on TEST            │
│     Result: 78% accuracy        │ ✓ REALISTIC!
└─────────────────────────────────┘
```

---

## 🔍 Data Leakage Patterns Reference

### Pattern 1: Statistics Leakage
```python
# ❌ WRONG
mean = df['column'].mean()  # Uses ALL data
df['scaled'] = df['column'] / mean
train, test = split(df)

# ✅ CORRECT
train, test = split(df)
mean = train['column'].mean()  # Only training data
train['scaled'] = train['column'] / mean
test['scaled'] = test['column'] / mean
```

### Pattern 2: Target Leakage
```python
# ❌ WRONG
X = df[['house_price', 'bedrooms']]  # house_price IS the target!
y = df['expensive']  # expensive = (house_price > threshold)

# ✅ CORRECT
X = df[['bedrooms', 'location', 'age']]
y = df['expensive']
```

### Pattern 3: Encoding Leakage
```python
# ❌ WRONG
df['cat_encoded'] = df.groupby('category')['target'].transform('mean')
train, test = split(df)

# ✅ CORRECT
train, test = split(df)
encoding = train.groupby('category')['target'].mean()
train['cat_encoded'] = train['category'].map(encoding)
test['cat_encoded'] = test['category'].map(encoding)
```

---

## 🛠️ The Correct Solution: Using Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define feature types
numeric_features = ['latitude', 'longitude', 'housing_median_age', 
                   'total_rooms', 'total_bedrooms', 'population',
                   'households', 'median_income']
categorical_features = ['ocean_proximity']

# Create preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2%}")  # ~82%
print(f"Test Accuracy: {test_score:.2%}")       # ~78%
```

---

## ⚠️ Red Flags Checklist

### Warning Signs of Data Leakage

- [ ] **Accuracy >95%** on realistic problems
- [ ] **No gap** between train and test scores
- [ ] **Perfect accuracy** (99%+) on complex tasks
- [ ] **Operations on `df`** before splitting
- [ ] **fit() or fit_transform()** on full dataset
- [ ] **Features derived from target** in your X
- [ ] **Future information** in time-series data
- [ ] **Test data** in training preprocessing

### Questions to Ask

For each operation:
1. **"Have I split my data yet?"** → If NO, wait!
2. **"Does this use multi-row statistics?"** → If YES, only use train!
3. **"Will I have this feature in production?"** → If NO, remove it!
4. **"Am I fitting or transforming?"** → fit() on train only!

---

## 💡 The Golden Rules

### Rule 1: Split First
```
ALWAYS:  1. Split  →  2. Preprocess  →  3. Train
NEVER:   1. Preprocess  →  2. Split  →  3. Train
```

### Rule 2: Fit Only on Training
```
ALWAYS:  scaler.fit(X_train)
         X_train_scaled = scaler.transform(X_train)
         X_test_scaled = scaler.transform(X_test)

NEVER:   scaler.fit(X)  # ALL data including test
```

### Rule 3: Use Pipeline
```
Pipeline prevents mistakes by ensuring:
- Correct order of operations
- Fit on train, transform on test
- Reproducible preprocessing
```

### Rule 4: Remove Target Leakage
```
Ask: "Would I have this feature when making predictions?"
- Price (when predicting price category) → NO
- Future sales (when predicting today) → NO
- Test results (when predicting outcomes) → NO
- User demographics → YES
```

---

## 📚 Bug Summary Table

| # | Bug | Type | Severity | Impact |
|---|-----|------|----------|--------|
| 1 | Feature engineering on full dataset | Statistics Leakage | High | Major |
| 2 | Target encoding before split | Target Leakage | **Critical** | Severe |
| 3 | Imputation before split | Statistics Leakage | High | Major |
| 4 | Scaling before split | Statistics Leakage | High | Major |
| 5 | Including target-derived features | Target Leakage | **Critical** | Severe |
| 6 | Inconsistent categorical encoding | Methodology | Medium | Moderate |
| 7 | CV on pre-leaked data | Methodology | Medium | Moderate |

---

## ⏱️ Time Management

- **Understanding the problem:** 5-10 minutes
- **Finding bugs:** 15-20 minutes
- **Implementing fixes:** 15-20 minutes
- **Verification & testing:** 5 minutes

**Total: ~45 minutes**

---

## ✅ Success Criteria

### Minimum Pass (60%)
- Find 3-4 bugs
- Basic understanding of data leakage
- Some improvement in performance

### Target (75%)
- Find 5-6 bugs
- Good understanding of concepts
- Proper use of some Pipeline components
- Realistic performance achieved

### Excellence (90%)
- Find **all 7 bugs**
- Deep understanding with clear explanations
- Complete Pipeline implementation
- Achieve ~75-80% accuracy (realistic)

---

## 🎓 Key Takeaways

1. **ALWAYS split before preprocessing**
   - Train/test split should be the FIRST operation
   - Never fit transformers on the full dataset

2. **Use sklearn Pipeline**
   - Prevents accidental leakage
   - Ensures correct order of operations
   - Makes code reproducible and production-ready

3. **Be suspicious of high accuracy**
   - If accuracy >95% on real-world problems, investigate!
   - Perfect scores usually indicate leakage

4. **Remove features that won't exist in production**
   - Ask: "Will I have this when making predictions?"
   - Remove anything derived from the target

5. **Target encoding is dangerous**
   - Requires careful implementation with CV
   - Easy to leak information
   - Use standard encodings when possible

6. **Cross-validation must be nested properly**
   - Use Pipeline with CV
   - Each fold does its own preprocessing

---

## 📁 File Structure

```
model_too_good/
├── EXAM_GUIDE.md              # This file - start here!
├── leaky_model.py             # Buggy code (find the bugs!)
├── solution.py                # Correct implementation
├── data_leakage_exam.ipynb    # Interactive notebook version
├── generate_dataset.py        # Dataset generator
└── housing_with_target_leakage.csv  # Generated dataset
```

---

## 🎯 After Completing

1. **Compare** your solution with `solution.py`
2. **Reflect** on what you learned
3. **Practice** creating leak-free pipelines
4. **Apply** these principles to real projects
5. **Teach** someone else what you learned

---

**Remember: If your model is too good to be true, check for data leakage!** 🔍

**Ready to start? Open `leaky_model.py` or `data_leakage_exam.ipynb` and begin!** 🚀

Good luck! 🍀
