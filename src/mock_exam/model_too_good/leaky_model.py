"""
Mock Exam: Find and Fix the Data Leakage Issues!

This code achieves 99.8% accuracy - which is suspicious!
Your task: Find ALL the data leakage issues and fix them.

Expected realistic performance after fixes: ~70-85% accuracy
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

# Set random seed for reproducibility
np.random.seed(42)

# ==================== LOAD DATA ====================
print("Loading data...")
df = pd.read_csv("housing_with_target_leakage.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")


# ==================== FEATURE ENGINEERING ====================
print("\n🔧 Engineering features...")

# BUG #1: Creating features using statistics from the ENTIRE dataset!
# This is data leakage - the model learns patterns from test data
df['price_vs_mean'] = df['median_house_value'] / df['median_house_value'].mean()
df['price_zscore'] = (df['median_house_value'] - df['median_house_value'].mean()) / df['median_house_value'].std()

# BUG #2: Target encoding on full dataset before split!
# The encoder sees test labels, causing severe leakage
ocean_proximity_means = df.groupby('ocean_proximity')['high_value'].mean()
df['ocean_proximity_target_encoded'] = df['ocean_proximity'].map(ocean_proximity_means)

# Create some legitimate features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

print(f"Features created. New shape: {df.shape}")


# ==================== HANDLE MISSING VALUES ====================
print("\n🔧 Handling missing values on full dataset...")

# BUG #3: Imputing BEFORE train/test split!
# The imputer learns statistics from test data
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


# ==================== SCALING ====================
print("\n🔧 Scaling features on full dataset...")

# BUG #4: Scaling BEFORE split - scaler sees test data!
scaler = StandardScaler()
numeric_features = ['median_income', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households',
                   'rooms_per_household', 'bedrooms_per_room', 
                   'population_per_household', 'price_vs_mean', 'price_zscore']

df[numeric_features] = scaler.fit_transform(df[numeric_features])
print("Scaling complete!")


# ==================== PREPARE X and y ====================
# BUG #5: Including features that leak target information!
# 'median_house_value' is directly related to 'high_value'
# 'price_vs_mean' and 'price_zscore' are derived from the target!

# Suspicious features that shouldn't exist in production
suspicious_features = ['median_house_value', 'price_vs_mean', 'price_zscore', 
                      'ocean_proximity_target_encoded']

feature_cols = [col for col in df.columns if col not in ['high_value']]
X = df[feature_cols]
y = df['high_value']

print(f"\n📊 Feature columns ({len(feature_cols)}):")
print(feature_cols)


# ==================== TRAIN/TEST SPLIT ====================
print("\n✂️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")


# ==================== ENCODING CATEGORICAL VARIABLES ====================
print("\n🔧 Encoding categorical variables...")

# BUG #6: OneHotEncoder fitted on training data but we already used
# the categories for target encoding earlier (on full dataset)!

if 'ocean_proximity' in X_train.columns:
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    encoded_train = encoder.fit_transform(X_train[['ocean_proximity']])
    encoded_test = encoder.transform(X_test[['ocean_proximity']])
    
    encoded_cols = encoder.get_feature_names_out(['ocean_proximity'])
    
    X_train_encoded = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
    
    X_train = pd.concat([X_train.drop('ocean_proximity', axis=1), X_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop('ocean_proximity', axis=1), X_test_encoded], axis=1)


# ==================== TRAIN MODEL ====================
print("\n🤖 Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained!")


# ==================== EVALUATION ====================
print("\n" + "="*60)
print("📈 MODEL PERFORMANCE")
print("="*60)

# Training accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Test accuracy
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# BUG #7: Cross-validation on data that's already been preprocessed
# using statistics from the full dataset - this CV score is meaningless!
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\n🚨 SUSPICIOUS RESULTS DETECTED! 🚨")
print(f"The accuracy is {test_accuracy:.2%} - this is TOO GOOD TO BE TRUE!")
print("\nYour task:")
print("1. Find all the data leakage issues")
print("2. Explain what each issue causes")
print("3. Fix the code to get realistic performance")
print("\nHint: There are at least 7 major bugs in this code!")


# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("🔍 TOP 10 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
print("\n⚠️ Notice anything suspicious about these top features?")
