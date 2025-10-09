# Mock Exam Answer Key

## Data Leakage Issues Found (7 Major Bugs)

### 🐛 BUG #1: Feature Engineering on Full Dataset
**Location:** Lines 30-32
```python
df['price_vs_mean'] = df['median_house_value'] / df['median_house_value'].mean()
df['price_zscore'] = (df['median_house_value'] - df['median_house_value'].mean()) / df['median_house_value'].std()
```

**Why it's wrong:**
- Calculates mean and std from the ENTIRE dataset (including test data)
- When scaling/normalizing test data, it uses statistics that include test information
- The model indirectly "sees" test data patterns

**Fix:** 
- Move feature engineering into a Pipeline transformer
- Or compute statistics only on training data after split

---

### 🐛 BUG #2: Target Encoding Before Split
**Location:** Lines 34-36
```python
ocean_proximity_means = df.groupby('ocean_proximity')['high_value'].mean()
df['ocean_proximity_target_encoded'] = df['ocean_proximity'].map(ocean_proximity_means)
```

**Why it's wrong:**
- Uses the TARGET VARIABLE ('high_value') to create a feature
- Computes means from the entire dataset including test set
- Each category's encoding includes information from test set labels
- This is severe target leakage!

**Fix:**
- Either remove target encoding completely
- Or use proper cross-validation target encoding (e.g., category_encoders library)
- Never use test set labels for encoding

---

### 🐛 BUG #3: Imputing Before Split
**Location:** Lines 48-54
```python
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```

**Why it's wrong:**
- Computes median/mode from the entire dataset
- Test set missing values are filled using statistics from test data
- The imputer "learns" from test set

**Fix:**
- Split data first
- Fit imputer only on X_train: `imputer.fit(X_train)`
- Transform both: `X_train_imputed = imputer.transform(X_train)`
- Use Pipeline to automate this

---

### 🐛 BUG #4: Scaling Before Split
**Location:** Lines 63-64
```python
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

**Why it's wrong:**
- StandardScaler learns mean and std from entire dataset
- Test data is scaled using mean/std that includes test information
- One of the most common data leakage mistakes!

**Fix:**
- Split first, then fit scaler only on training data
- Use Pipeline to ensure correct order

---

### 🐛 BUG #5: Target Leakage in Features
**Location:** Lines 69-72
```python
suspicious_features = ['median_house_value', 'price_vs_mean', 'price_zscore', 
                      'ocean_proximity_target_encoded']
```

**Why it's wrong:**
- `median_house_value` DIRECTLY determines the target `high_value`
- `price_vs_mean` and `price_zscore` are derived from the target
- These features wouldn't be available in production!
- The model essentially sees the answer

**Fix:**
- Remove all features derived from or related to the target
- Only use features available at prediction time

---

### 🐛 BUG #6: Inconsistent Categorical Encoding
**Location:** Lines 96-108 (OneHotEncoder fitted after target encoding)

**Why it's wrong:**
- Categories were already used for target encoding on full dataset
- Then OneHotEncoder is fitted on just training data
- Inconsistent treatment of the same categorical variable

**Fix:**
- Choose ONE encoding method
- Apply it consistently within a pipeline
- Fit only on training data

---

### 🐛 BUG #7: Meaningless Cross-Validation
**Location:** Lines 135-137
```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
```

**Why it's wrong:**
- While this CV is on training data only, the data has already been preprocessed
- Preprocessing used statistics from the full dataset (including test)
- CV scores are inflated and meaningless
- Should use Pipeline with CV to properly separate folds

**Fix:**
- Use Pipeline
- Run CV on the pipeline, not on preprocessed data

---

## Summary of All Issues

| Bug # | Issue | Type | Severity |
|-------|-------|------|----------|
| 1 | Feature engineering on full dataset | Leakage | High |
| 2 | Target encoding before split | Target Leakage | Critical |
| 3 | Imputation before split | Leakage | High |
| 4 | Scaling before split | Leakage | High |
| 5 | Including target-derived features | Target Leakage | Critical |
| 6 | Inconsistent categorical encoding | Methodology | Medium |
| 7 | CV on pre-leaked data | Methodology | Medium |

---

## Expected Results

### Before Fixes (Leaky Model):
- Training Accuracy: ~99.8%
- Test Accuracy: ~99.8%
- CV Accuracy: ~99.8%
- **Too good to be true!**

### After Fixes (Proper Model):
- Training Accuracy: ~80-85%
- Test Accuracy: ~75-80%
- CV Accuracy: ~75-80%
- **Realistic and trustworthy!**

---

## Key Lessons

1. **ALWAYS split before preprocessing**
   - Train/test split should be the FIRST operation
   - Never fit transformers on the full dataset

2. **Use sklearn Pipeline**
   - Prevents accidental leakage
   - Ensures correct order of operations
   - Makes code reproducible

3. **Be suspicious of high accuracy**
   - If accuracy >95% on real-world problems, investigate
   - Perfect or near-perfect scores usually indicate leakage

4. **Remove features that won't exist in production**
   - Ask: "Will I have this feature when making predictions?"
   - Remove anything derived from the target

5. **Target encoding is dangerous**
   - Requires careful implementation
   - Must use proper CV or dedicated libraries
   - Easy to leak information

6. **Cross-validation must be properly nested**
   - Use Pipeline with CV
   - Each fold should do its own preprocessing

---

## Grading Rubric

### Excellent (90-100%)
- Found all 7 bugs
- Explained each issue clearly
- Fixed code properly using Pipeline
- Achieved realistic performance
- Demonstrated deep understanding

### Good (75-89%)
- Found 5-6 bugs
- Explained most issues correctly
- Fixed most of the code
- Performance improved significantly

### Satisfactory (60-74%)
- Found 3-4 bugs
- Basic understanding of leakage
- Partial fixes implemented
- Some improvement in performance

### Needs Improvement (<60%)
- Found <3 bugs
- Unclear explanations
- Fixes don't resolve leakage
- Performance still unrealistic

---

## Additional Resources

1. **Sklearn Documentation:**
   - [Pipeline and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
   - [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

2. **Articles on Data Leakage:**
   - [Kaggle: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)
   - [Towards Data Science: Common ML Mistakes](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742)

3. **Best Practices:**
   - Always use Pipeline
   - Split first, preprocess second
   - Be paranoid about test data
   - If it's too good to be true, it probably is
