"""
Generate datasets for scaling issues mock exam
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("GENERATING DATASETS FOR SCALING EXAM")
print("=" * 80)

# =============================================================================
# Dataset 1: Distance-based model (KNN) - Needs StandardScaler
# Problem: No scaling applied, features have very different scales
# =============================================================================
print("\n📊 Dataset 1: Credit Card Fraud Detection (for KNN)")
print("-" * 80)

n_samples = 1000

# Generate base data
X, y = make_classification(
    n_samples=n_samples,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    weights=[0.95, 0.05],  # Imbalanced
    random_state=42
)

# Create features with VERY different scales (this is the problem!)
df1 = pd.DataFrame({
    'transaction_amount': np.abs(X[:, 0] * 5000 + 1000),  # Range: 0-10000
    'account_age_days': np.abs(X[:, 1] * 500 + 100),      # Range: 0-1000
    'num_transactions': np.abs(X[:, 2] * 50 + 10),        # Range: 0-100
    'avg_transaction': np.abs(X[:, 3] * 200 + 50),        # Range: 0-400
    'credit_limit': np.abs(X[:, 4] * 20000 + 5000),       # Range: 5000-30000
    'is_fraud': y
})

df1.to_csv('data/credit_fraud.csv', index=False)
print(f"✓ Generated credit_fraud.csv")
print(f"  Shape: {df1.shape}")
print(f"  Features have VERY different scales:")
print(f"  - transaction_amount: {df1['transaction_amount'].min():.0f} - {df1['transaction_amount'].max():.0f}")
print(f"  - credit_limit: {df1['credit_limit'].min():.0f} - {df1['credit_limit'].max():.0f}")
print(f"  - num_transactions: {df1['num_transactions'].min():.0f} - {df1['num_transactions'].max():.0f}")

# =============================================================================
# Dataset 2: Data with outliers - Wrong scaler used (MinMax instead of Robust)
# Problem: MinMaxScaler used but data has outliers
# =============================================================================
print("\n📊 Dataset 2: House Prices with Outliers")
print("-" * 80)

n_samples = 800

# Generate base regression data
X, y = make_regression(
    n_samples=n_samples,
    n_features=4,
    noise=10,
    random_state=42
)

# Create features with outliers
df2 = pd.DataFrame({
    'square_feet': np.abs(X[:, 0] * 500 + 1500),
    'num_bedrooms': np.abs(X[:, 1] * 1.5 + 3).astype(int),
    'lot_size': np.abs(X[:, 2] * 2000 + 5000),
    'year_built': np.abs(X[:, 3] * 20 + 1980).astype(int),
})

# Add OUTLIERS to square_feet and lot_size (this is why MinMax is wrong!)
outlier_indices = np.random.choice(n_samples, size=30, replace=False)
df2.loc[outlier_indices, 'square_feet'] *= 3  # Some houses 3x bigger
df2.loc[outlier_indices[:15], 'lot_size'] *= 5  # Some lots 5x bigger

# Generate target with outlier influence
df2['price'] = (
    df2['square_feet'] * 200 + 
    df2['num_bedrooms'] * 10000 + 
    df2['lot_size'] * 50 +
    (df2['year_built'] - 1980) * 1000 +
    np.random.normal(0, 20000, n_samples)
)

df2.to_csv('data/house_prices_outliers.csv', index=False)
print(f"✓ Generated house_prices_outliers.csv")
print(f"  Shape: {df2.shape}")
print(f"  Contains outliers in square_feet and lot_size")
print(f"  square_feet outliers: {df2['square_feet'].nlargest(5).values}")

# =============================================================================
# Dataset 3: Tree-based model data - Scaling applied (but shouldn't be)
# Problem: StandardScaler applied to Random Forest (unnecessary)
# =============================================================================
print("\n📊 Dataset 3: Customer Churn (for Random Forest)")
print("-" * 80)

n_samples = 1200

# Generate classification data
X, y = make_classification(
    n_samples=n_samples,
    n_features=6,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)

df3 = pd.DataFrame({
    'monthly_charges': np.abs(X[:, 0] * 50 + 50),
    'total_charges': np.abs(X[:, 1] * 2000 + 500),
    'tenure_months': np.abs(X[:, 2] * 30 + 1).astype(int),
    'contract_type': np.random.choice([0, 1, 2], n_samples),  # 0=monthly, 1=annual, 2=biannual
    'num_services': np.abs(X[:, 4] * 3 + 1).astype(int),
    'customer_age': np.abs(X[:, 5] * 20 + 25).astype(int),
    'churned': y
})

df3.to_csv('data/customer_churn.csv', index=False)
print(f"✓ Generated customer_churn.csv")
print(f"  Shape: {df3.shape}")
print(f"  For tree-based models (Random Forest)")

# =============================================================================
# Dataset 4: Train/Test scaling done WRONG - Data leakage
# Problem: Scaler fit on entire dataset before train/test split
# =============================================================================
print("\n📊 Dataset 4: Student Performance (Scaling Leakage)")
print("-" * 80)

n_samples = 600

# Generate regression data
X, y = make_regression(
    n_samples=n_samples,
    n_features=5,
    noise=5,
    random_state=42
)

df4 = pd.DataFrame({
    'study_hours': np.abs(X[:, 0] * 10 + 20),
    'attendance_rate': np.clip(X[:, 1] * 15 + 75, 0, 100),
    'previous_grade': np.clip(X[:, 2] * 10 + 70, 40, 100),
    'homework_completion': np.clip(X[:, 3] * 15 + 75, 0, 100),
    'class_participation': np.clip(X[:, 4] * 15 + 70, 0, 100),
})

# Generate target
df4['final_grade'] = (
    df4['study_hours'] * 1.5 +
    df4['attendance_rate'] * 0.3 +
    df4['previous_grade'] * 0.4 +
    df4['homework_completion'] * 0.2 +
    df4['class_participation'] * 0.2 +
    np.random.normal(0, 5, n_samples)
)
df4['final_grade'] = np.clip(df4['final_grade'], 0, 100)

df4.to_csv('data/student_performance.csv', index=False)
print(f"✓ Generated student_performance.csv")
print(f"  Shape: {df4.shape}")
print(f"  For demonstrating train/test scaling leakage")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
print("=" * 80)

print("\nDatasets summary:")
print("1. credit_fraud.csv - KNN without scaling (features have different scales)")
print("2. house_prices_outliers.csv - MinMaxScaler with outliers (should use RobustScaler)")
print("3. customer_churn.csv - Random Forest with scaling (unnecessary)")
print("4. student_performance.csv - Scaling leakage (fit on all data before split)")

print("\n📋 Problems to identify:")
print("  Problem 1: Missing scaling for distance-based model")
print("  Problem 2: Wrong scaler for data with outliers")
print("  Problem 3: Unnecessary scaling for tree-based model")
print("  Problem 4: Data leakage in train/test scaling")
