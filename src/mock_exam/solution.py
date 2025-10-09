"""
SOLUTION: Fixed version of the leaky model

This demonstrates the CORRECT way to preprocess data and train a model
without data leakage.

Key fixes:
1. Train/test split BEFORE any preprocessing
2. Feature engineering only on training data statistics
3. Removed features that leak target information
4. All transformers fit only on training data
5. Proper pipeline usage to prevent leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Set random seed for reproducibility
np.random.seed(42)


# ==================== CUSTOM TRANSFORMER FOR FEATURE ENGINEERING ====================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features WITHOUT leaking information from test set.
    All statistics are computed only on the training data.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Safe feature engineering - using only row-level operations
        X_copy['rooms_per_household'] = X_copy['total_rooms'] / X_copy['households']
        X_copy['bedrooms_per_room'] = X_copy['total_bedrooms'] / X_copy['total_rooms']
        X_copy['population_per_household'] = X_copy['population'] / X_copy['households']
        
        # Handle division by zero / inf
        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
        
        return X_copy


# ==================== LOAD DATA ====================
print("Loading data...")
df = pd.read_csv("housing_with_target_leakage.csv")
print(f"Dataset shape: {df.shape}")


# ==================== CRITICAL FIX #1: REMOVE TARGET LEAKAGE ====================
# REMOVE 'median_house_value' - this directly leaks the target!
# In production, we won't have the actual house value when predicting.

print("\n🔒 Removing features that leak target information...")
print("   - median_house_value (directly reveals the target)")

leaky_features = ['median_house_value']
df = df.drop(columns=leaky_features)

print(f"Cleaned dataset shape: {df.shape}")


# ==================== CRITICAL FIX #2: SPLIT DATA FIRST ====================
# Split BEFORE any preprocessing to prevent information leakage

print("\n✂️ Splitting data FIRST (before any preprocessing)...")

y = df['high_value']
X = df.drop(columns=['high_value'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")


# ==================== CRITICAL FIX #3: BUILD PROPER PIPELINE ====================
# Use sklearn Pipeline to ensure all transformations are fitted only on training data

print("\n🏗️ Building preprocessing pipeline...")

# Identify numeric and categorical columns
numeric_features = ['median_income', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households']
categorical_features = ['ocean_proximity']

# Numeric pipeline: impute missing values, engineer features, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_engineer', FeatureEngineer()),  # Creates ratio features
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline: preprocessing + model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

print("Pipeline structure:")
print(full_pipeline)


# ==================== TRAIN MODEL ====================
print("\n🤖 Training model with proper pipeline...")

# FIT the entire pipeline on training data only
full_pipeline.fit(X_train, y_train)
print("✅ Model trained successfully!")


# ==================== EVALUATION ====================
print("\n" + "="*60)
print("📈 MODEL PERFORMANCE (After Fixing Data Leakage)")
print("="*60)

# Training accuracy
y_train_pred = full_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Test accuracy
y_test_pred = full_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Cross-validation (now properly done on training data only)
cv_scores = cross_val_score(
    full_pipeline, X_train, y_train, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_test_pred))


# ==================== ANALYSIS ====================
print("\n" + "="*60)
print("🔍 WHAT WAS FIXED?")
print("="*60)

print("""
BUGS FIXED:

1️⃣ REMOVED TARGET LEAKAGE:
   - Removed 'median_house_value' (directly reveals target)
   - Removed features derived from the target (price_vs_mean, price_zscore)
   - Removed target-encoded features created on full dataset

2️⃣ SPLIT DATA FIRST:
   - Now splitting BEFORE any preprocessing
   - This ensures test data is truly unseen

3️⃣ FIT TRANSFORMERS ONLY ON TRAINING DATA:
   - SimpleImputer.fit() only sees training data
   - StandardScaler.fit() only sees training data
   - OneHotEncoder.fit() only sees training categories

4️⃣ USED SKLEARN PIPELINE:
   - Ensures correct order of operations
   - Prevents accidental leakage
   - Makes code more maintainable

5️⃣ FEATURE ENGINEERING IN PIPELINE:
   - Created custom FeatureEngineer transformer
   - Only uses row-level operations (no dataset statistics)
   - Safely creates ratio features

6️⃣ PROPER CROSS-VALIDATION:
   - CV now done on properly preprocessed data
   - Each fold respects the train/test boundary

7️⃣ REMOVED DATASET-WIDE STATISTICS:
   - No more df.mean(), df.std() on full dataset
   - No more groupby().mean() for target encoding before split
   - All statistics computed within pipeline

RESULT:
   - Test accuracy dropped from ~99.8% to ~{test_accuracy:.1%}
   - This is REALISTIC performance for this problem
   - Model generalizes properly to unseen data
   - Train and test accuracy are closer (less overfitting)
""")


# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("🔍 FEATURE IMPORTANCE (from properly trained model)")
print("="*60)

# Get feature names after preprocessing
feature_names = (
    numeric_features + 
    [f for f in X_train.columns if f not in numeric_features + categorical_features]
)

# Add one-hot encoded categorical feature names
ohe = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names += cat_feature_names

# Get feature importances
model = full_pipeline.named_steps['classifier']
feature_importance = pd.DataFrame({
    'feature': feature_names[:len(model.feature_importances_)],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

print("\n✅ These importances are now trustworthy!")
print("   They reflect what the model can actually learn from legitimate features.")


# ==================== KEY TAKEAWAYS ====================
print("\n" + "="*60)
print("📚 KEY TAKEAWAYS")
print("="*60)

print("""
1. ALWAYS split your data BEFORE any preprocessing
2. NEVER use statistics from the test set
3. Use sklearn Pipeline to prevent leakage
4. Remove features that wouldn't exist in production
5. Be suspicious of unrealistically high accuracy
6. Cross-validation must be done properly within the pipeline
7. Target encoding must be done carefully (use CV or dedicated libraries)

Remember: If it's too good to be true, it probably is!
""")
