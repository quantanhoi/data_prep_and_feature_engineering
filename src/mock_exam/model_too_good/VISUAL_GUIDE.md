# Visual Guide: Data Leakage Explained

## 🚫 WRONG: The Leaky Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL DATASET                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ train data + test data (ALL MIXED TOGETHER)       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
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
        │     (but data already leaked!)  │
        └─────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │  Train Set   │    │  Test Set    │
        │ (contaminated)│    │(contaminated)│
        └──────────────┘    └──────────────┘
                │                   │
                └─────────┬─────────┘
                          ▼
        ┌─────────────────────────────────┐
        │  5. Train model                 │
        │     Result: 99.8% accuracy      │ ❌ TOO GOOD!
        └─────────────────────────────────┘

PROBLEM: Test set "leaked" into training through preprocessing!
The model already saw patterns from test data.
```

---

## ✅ CORRECT: The Leak-Free Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL DATASET                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  1. SPLIT FIRST!                │ ✓ CORRECT!
        │     train_test_split()          │
        └─────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │  Train Set   │    │  Test Set    │
        │   (80%)      │    │   (20%)      │
        └──────────────┘    └──────────────┘
                │                   │
                │                   │ (stays separate)
                ▼                   │
        ┌─────────────────┐         │
        │ 2. Fit Scaler   │ ✓       │
        │    on TRAIN only│         │
        └─────────────────┘         │
                │                   │
                ▼                   ▼
        ┌─────────────────┐   ┌─────────────────┐
        │ 3. Transform    │   │ 3. Transform    │
        │    TRAIN        │   │    TEST         │
        │    (fitted)     │   │    (not fitted!)│
        └─────────────────┘   └─────────────────┘
                │                   │
                ▼                   ▼
        ┌─────────────────┐   ┌─────────────────┐
        │ 4. Fit Imputer  │   │ 4. Transform    │
        │    on TRAIN only│   │    TEST         │
        └─────────────────┘   └─────────────────┘
                │                   │
                ▼                   ▼
        ┌─────────────────┐   ┌─────────────────┐
        │ 5. Engineer     │   │ 5. Engineer     │
        │    features     │   │    features     │
        │    (train stats)│   │    (same logic) │
        └─────────────────┘   └─────────────────┘
                │                   │
                ▼                   │
        ┌─────────────────┐         │
        │ 6. Train model  │         │
        │    on TRAIN     │         │
        └─────────────────┘         │
                │                   │
                └─────────┬─────────┘
                          ▼
        ┌─────────────────────────────────┐
        │  7. Evaluate on TEST            │
        │     Result: 78% accuracy        │ ✓ REALISTIC!
        └─────────────────────────────────┘

CORRECT: Test set completely separate during preprocessing!
Model only learns from training data.
```

---

## 🔍 Common Leakage Patterns

### Pattern 1: Statistics Leakage
```
❌ WRONG:
mean = df['column'].mean()  # Uses ALL data
df['scaled'] = df['column'] / mean
train, test = split(df)

✅ CORRECT:
train, test = split(df)
mean = train['column'].mean()  # Only training data
train['scaled'] = train['column'] / mean
test['scaled'] = test['column'] / mean
```

### Pattern 2: Target Leakage
```
❌ WRONG:
# Feature that directly reveals the target
X = df[['house_price', 'bedrooms']]  # house_price is target!
y = df['expensive']  # expensive = (house_price > threshold)

✅ CORRECT:
# Only features available at prediction time
X = df[['bedrooms', 'location', 'age']]
y = df['expensive']
```

### Pattern 3: Encoding Leakage
```
❌ WRONG:
# Target encoding before split
df['cat_encoded'] = df.groupby('category')['target'].transform('mean')
train, test = split(df)

✅ CORRECT:
train, test = split(df)
# Compute encoding only from training data
encoding = train.groupby('category')['target'].mean()
train['cat_encoded'] = train['category'].map(encoding)
test['cat_encoded'] = test['category'].map(encoding)
```

---

## 📊 The Impact

### Leaky Model Performance
```
               Leaky Model    Real World
Train Acc:        99.8%         50-60%  😱
Test Acc:         99.8%         50-60%  😱
Kaggle Public:    99.8%          --
Kaggle Private:    --           50-60%  💥 FAIL!
Production:        --           50-60%  💸 COST $$$
```

### Proper Model Performance
```
               Proper Model   Real World
Train Acc:        82.3%         78-80%  ✓
Test Acc:         78.6%         78-80%  ✓
Kaggle Public:    78.6%         78-80%  ✓
Kaggle Private:   78.3%         78-80%  ✓
Production:       78.1%         78-80%  ✓ SUCCESS!
```

---

## 🎯 Decision Tree: Should I Split Now?

```
                    Start
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Have I split my data yet?   │
        └─────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
           NO                  YES
            │                   │
            ▼                   ▼
    ┌────────────────┐   ┌────────────────┐
    │ STOP!          │   │ Good!          │
    │ Split NOW!     │   │ Proceed with   │
    │                │   │ preprocessing  │
    └────────────────┘   └────────────────┘
            │                   │
            ▼                   ▼
    ┌────────────────┐   ┌────────────────┐
    │ Only AFTER     │   │ Always fit on  │
    │ splitting can  │   │ TRAIN, then    │
    │ you preprocess │   │ transform TEST │
    └────────────────┘   └────────────────┘
```

---

## 🧪 Quick Self-Check

For each operation, ask yourself:

1. **"Have I split my data yet?"**
   - If NO → Don't do it yet!
   - If YES → Proceed

2. **"Does this use information from multiple rows?"**
   - mean(), median(), std(), min(), max() → YES
   - groupby(), transform() → YES
   - Row-level operations (/, *, +, -) → NO

3. **"Will I have this feature in production?"**
   - Target variable → NO
   - Future information → NO
   - Test results → NO
   - User input + public data → YES

4. **"Am I fitting or transforming?"**
   - fit() or fit_transform() → Only on TRAIN
   - transform() → Can use on TEST

---

## 💡 The Golden Rules

### Rule 1: Split First
```
ALWAYS:  1. Split  2. Preprocess  3. Train
NEVER:   1. Preprocess  2. Split  3. Train
```

### Rule 2: Fit Only on Training
```
ALWAYS:  scaler.fit(X_train)
         X_train_scaled = scaler.transform(X_train)
         X_test_scaled = scaler.transform(X_test)

NEVER:   scaler.fit(X)  # ALL data
         X_train, X_test = split(X_scaled)
```

### Rule 3: Use Pipeline
```
ALWAYS:  Pipeline([
           ('scaler', StandardScaler()),
           ('model', RandomForest())
         ]).fit(X_train, y_train)

BETTER THAN:  scaler.fit(X_train)
              X_scaled = scaler.transform(X_train)
              model.fit(X_scaled, y_train)
```

### Rule 4: Remove Target Leakage
```
ALWAYS:  X = df[['bedrooms', 'location']]
         y = df['price_category']

NEVER:   X = df[['bedrooms', 'price', 'location']]
         y = df['price_category']  # price → price_category!
```

---

## 🎓 Memory Aid: "The 3 S's"

1. **SPLIT** first (before anything else)
2. **SEPARATE** train and test completely
3. **STATISTICS** only from training data

---

## 📝 Checklist Before Training

- [ ] Removed features that leak target information
- [ ] Split data into train/test FIRST
- [ ] All .fit() calls use only X_train
- [ ] All .transform() on X_test uses fitted transformers
- [ ] No statistics (mean, std, etc.) from full dataset
- [ ] Used Pipeline to prevent mistakes
- [ ] Cross-validation done properly (within pipeline)
- [ ] Performance seems realistic (not 99%+)

---

**Remember:** If your model is too good to be true, check for data leakage!
