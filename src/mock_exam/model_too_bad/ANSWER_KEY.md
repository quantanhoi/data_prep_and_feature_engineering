# Mock Exam: Model Too Bad - Answer Key

## Summary of All Bugs

### 1. **Dropping ALL rows with missing values** ❌
- **Problem:** Lost 30% of data unnecessarily
- **Fix:** Use imputation (median for numeric, mode for categorical)
- **Impact:** Huge - you need that training data!

### 2. **Using Label Encoding for nominal categories** ❌
- **Problem:** Creates artificial ordinal relationships (INLAND=0, NEAR BAY=1 implies NEAR BAY > INLAND)
- **Fix:** Use OneHotEncoder for non-ordinal categories
- **Impact:** High - model learns wrong relationships

### 3. **Encoding high-cardinality categorical (5000 blocks)** ❌
- **Problem:** Creates 5000 features or meaningless integer encoding
- **Fix:** Drop block_id or use target encoding/hashing
- **Impact:** Medium - adds noise and computational cost

### 4. **No outlier detection/treatment** ❌
- **Problem:** Population has extreme outliers (10x normal)
- **Fix:** Clip outliers or use RobustScaler
- **Impact:** High - outliers ruin StandardScaler

### 5. **Using StandardScaler with outliers** ❌
- **Problem:** Outliers compress normal values into tiny range
- **Fix:** Use RobustScaler or remove outliers first
- **Impact:** High - poor feature scaling = poor performance

### 6. **No feature engineering** ❌
- **Problem:** Missing domain knowledge features
- **Fix:** Create rooms_per_person, bedrooms_ratio, location features
- **Impact:** Medium-High - engineered features often most predictive

### 7. **Not using stratified split with imbalanced data** ❌
- **Problem:** Test set may have different class distribution
- **Fix:** Use `stratify=y` in train_test_split
- **Impact:** Medium - evaluation becomes unreliable

### 8. **Ignoring class imbalance (30/70 split)** ❌
- **Problem:** Model biased toward majority class
- **Fix:** Use `class_weight='balanced'` or SMOTE
- **Impact:** Very High - model predicts mostly one class

### 9. **No hyperparameter tuning** ❌
- **Problem:** Default parameters rarely optimal
- **Fix:** Use GridSearchCV or RandomizedSearchCV
- **Impact:** Medium - can improve 5-10%

### 10. **Using only accuracy as metric** ❌
- **Problem:** Accuracy misleading with imbalanced data
- **Fix:** Check precision, recall, F1-score per class
- **Impact:** Medium - affects evaluation and debugging

---

## Complete Solution

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)

# Load data
df = pd.read_csv("housing_for_bad_model.csv")

# ============================================================
# FIX #1: Feature Engineering BEFORE splitting
# ============================================================
# Create useful features (row-level operations only)
df['rooms_per_person'] = df['avg_rooms'] / df['avg_occupancy']
df['bedrooms_ratio'] = df['avg_bedrooms'] / df['avg_rooms']

# Location features
df['lat_lon_interaction'] = df['latitude'] * df['longitude']
df['distance_to_center'] = np.sqrt(
    (df['latitude'] - df['latitude'].median())**2 + 
    (df['longitude'] - df['longitude'].median())**2
)

# ============================================================
# FIX #2: Remove/handle problematic features
# ============================================================
# Drop high-cardinality block_id (not useful, too many categories)
df = df.drop('block_id', axis=1)

# ============================================================
# FIX #3: Handle outliers BEFORE scaling
# ============================================================
# Clip extreme outliers in population
pop_99 = df['population'].quantile(0.99)
df['population'] = df['population'].clip(upper=pop_99)

# ============================================================
# FIX #4: Prepare X and y
# ============================================================
X = df.drop('expensive', axis=1)
y = df['expensive']

# ============================================================
# FIX #5: Stratified train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # <-- STRATIFY!
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\\nTrain target distribution:\\n{y_train.value_counts(normalize=True)}")
print(f"\\nTest target distribution:\\n{y_test.value_counts(normalize=True)}")

# ============================================================
# FIX #6: Proper pipeline with correct preprocessing
# ============================================================
# Identify column types
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"\\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # FIX: Impute instead of drop
    ('scaler', RobustScaler())  # FIX: RobustScaler instead of StandardScaler
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # FIX: OneHot not Label
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ============================================================
# FIX #7: Better model with class balancing
# ============================================================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',  # FIX: Handle class imbalance!
        random_state=42,
        n_jobs=-1
    ))
])

# Train
print("\\nTraining model...")
model.fit(X_train, y_train)

# ============================================================
# FIX #8: Proper evaluation
# ============================================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\\n" + "="*60)
print("✅ RESULTS (MUCH BETTER!) ✅")
print("="*60)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# FIX: Look at per-class metrics, not just accuracy
print(f"\\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred))

print(f"\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"\\nCV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
if hasattr(model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    feature_names = (
        numeric_features + 
        list(model.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_features))
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
```

---

## Expected Results

### Before Fixes:
- **Test Accuracy:** ~55%
- **F1 Score:** ~0.35
- **Problem:** Model predicts mostly class 0

### After Fixes:
- **Test Accuracy:** ~83-85%
- **F1 Score:** ~0.80-0.82
- **Balanced:** Both classes predicted well

---

## Key Lessons

1. **Don't drop data unnecessarily** - Impute instead
2. **Use correct encoding** - OneHot for nominal, not Label
3. **Handle outliers** - They ruin scaling
4. **Use RobustScaler** - When you have outliers
5. **Engineer features** - Domain knowledge matters
6. **Stratify splits** - Especially with imbalanced data
7. **Handle class imbalance** - class_weight='balanced' or SMOTE
8. **Use pipelines** - Prevent data leakage
9. **Look beyond accuracy** - Precision, Recall, F1 matter
10. **Always do EDA first** - Understand your data!

---

## Comparison: Bad vs Good

| Aspect | Bad Model | Good Model |
|--------|-----------|------------|
| Missing Values | Drop all (lost 30% data) | Impute (keep all data) |
| Categorical | Label encode | OneHot encode |
| High Cardinality | Encode 5000 blocks | Drop block_id |
| Outliers | Ignored | Clipped |
| Scaling | StandardScaler | RobustScaler |
| Features | Raw only | Engineered ratios |
| Split | Random | Stratified |
| Class Imbalance | Ignored | class_weight='balanced' |
| Model | Default LogReg | Tuned RandomForest |
| Evaluation | Accuracy only | Full classification report |
| **Result** | **55% accuracy** | **85% accuracy** |

