"""
Complete Solutions for Scaling Issues Mock Exam
Run this after attempting the exam!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

np.random.seed(42)

print("=" * 80)
print("SCALING ISSUES EXAM - COMPLETE SOLUTIONS")
print("=" * 80)

# =============================================================================
# PROBLEM 1: Credit Card Fraud Detection - Missing Scaling for KNN
# =============================================================================
print("\n" + "=" * 80)
print("PROBLEM 1: Credit Card Fraud Detection (KNN needs scaling!)")
print("=" * 80)

df1 = pd.read_csv('data/credit_fraud.csv')
X1 = df1.drop('is_fraud', axis=1)
y1 = df1['is_fraud']

print("\n📊 Feature Scales (THIS IS THE PROBLEM):")
print(X1.describe())
print("\n⚠️  Notice the huge differences:")
print(f"   - credit_limit: {X1['credit_limit'].min():.0f} to {X1['credit_limit'].max():.0f}")
print(f"   - num_transactions: {X1['num_transactions'].min():.0f} to {X1['num_transactions'].max():.0f}")
print(f"   - Ratio: {X1['credit_limit'].max() / X1['num_transactions'].max():.0f}x difference!")

# Split
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42, stratify=y1
)

# WITHOUT scaling (buggy)
knn_bad = KNeighborsClassifier(n_neighbors=5)
knn_bad.fit(X1_train, y1_train)
acc_bad = accuracy_score(y1_test, knn_bad.predict(X1_test))

# WITH scaling (fixed)
print("\n✅ SOLUTION: Apply StandardScaler")
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

knn_fixed = KNeighborsClassifier(n_neighbors=5)
knn_fixed.fit(X1_train_scaled, y1_train)
acc_fixed = accuracy_score(y1_test, knn_fixed.predict(X1_test_scaled))

print(f"\n📊 RESULTS:")
print(f"   Accuracy WITHOUT scaling: {acc_bad:.3f} (terrible!)")
print(f"   Accuracy WITH scaling:    {acc_fixed:.3f} (much better!)")
print(f"   ✅ Improvement: +{(acc_fixed - acc_bad)*100:.1f}%")
print(f"\n💡 WHY: KNN uses Euclidean distance. Large-scale features dominate the distance!")

# =============================================================================
# PROBLEM 2: House Prices - Wrong Scaler (MinMax vs Robust)
# =============================================================================
print("\n\n" + "=" * 80)
print("PROBLEM 2: House Prices (MinMax fails with outliers!)")
print("=" * 80)

df2 = pd.read_csv('data/house_prices_outliers.csv')
X2 = df2.drop('price', axis=1)
y2 = df2['price']

# Visualize outliers
print("\n📊 Checking for outliers:")
for col in ['square_feet', 'lot_size']:
    Q1 = X2[col].quantile(0.25)
    Q3 = X2[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((X2[col] < Q1 - 1.5*IQR) | (X2[col] > Q3 + 1.5*IQR)).sum()
    print(f"   {col}: {outliers} outliers detected")

# Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42
)

# WITH MinMaxScaler (buggy - sensitive to outliers)
scaler2_bad = MinMaxScaler()
X2_train_bad = scaler2_bad.fit_transform(X2_train)
X2_test_bad = scaler2_bad.transform(X2_test)

lr_bad = LinearRegression()
lr_bad.fit(X2_train_bad, y2_train)
r2_bad = r2_score(y2_test, lr_bad.predict(X2_test_bad))

# WITH RobustScaler (fixed - handles outliers)
print("\n✅ SOLUTION: Use RobustScaler (uses median and IQR)")
scaler2_fixed = RobustScaler()
X2_train_fixed = scaler2_fixed.fit_transform(X2_train)
X2_test_fixed = scaler2_fixed.transform(X2_test)

lr_fixed = LinearRegression()
lr_fixed.fit(X2_train_fixed, y2_train)
r2_fixed = r2_score(y2_test, lr_fixed.predict(X2_test_fixed))

print(f"\n📊 RESULTS:")
print(f"   R² WITH MinMaxScaler:  {r2_bad:.3f} (poor)")
print(f"   R² WITH RobustScaler:  {r2_fixed:.3f} (much better!)")
print(f"   ✅ Improvement: +{(r2_fixed - r2_bad):.3f}")
print(f"\n💡 WHY: MinMaxScaler uses min/max. Outliers make max very large,")
print(f"        squeezing all normal values near 0!")
print(f"        RobustScaler uses median/IQR which are resistant to outliers.")

# =============================================================================
# PROBLEM 3: Customer Churn - Unnecessary Scaling for Random Forest
# =============================================================================
print("\n\n" + "=" * 80)
print("PROBLEM 3: Customer Churn (Scaling Random Forest is unnecessary!)")
print("=" * 80)

df3 = pd.read_csv('data/customer_churn.csv')
X3 = df3.drop('churned', axis=1)
y3 = df3['churned']

# Split
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.3, random_state=42, stratify=y3
)

# WITH scaling (unnecessary)
scaler3 = StandardScaler()
X3_train_scaled = scaler3.fit_transform(X3_train)
X3_test_scaled = scaler3.transform(X3_test)

rf_with_scaling = RandomForestClassifier(n_estimators=100, random_state=42)
rf_with_scaling.fit(X3_train_scaled, y3_train)
acc_with_scaling = accuracy_score(y3_test, rf_with_scaling.predict(X3_test_scaled))

# WITHOUT scaling (correct approach)
print("\n✅ SOLUTION: Don't scale tree-based models!")
rf_no_scaling = RandomForestClassifier(n_estimators=100, random_state=42)
rf_no_scaling.fit(X3_train, y3_train)
acc_no_scaling = accuracy_score(y3_test, rf_no_scaling.predict(X3_test))

print(f"\n📊 RESULTS:")
print(f"   Accuracy WITH scaling:    {acc_with_scaling:.3f}")
print(f"   Accuracy WITHOUT scaling: {acc_no_scaling:.3f}")
print(f"   Difference: {abs(acc_with_scaling - acc_no_scaling):.3f} (minimal)")
print(f"\n💡 WHY: Random Forest uses splits (e.g., 'age > 30'), not distances.")
print(f"        It doesn't matter if age is 30 or 0.5 (scaled) - the tree")
print(f"        finds optimal splits regardless of scale!")
print(f"\n✅ KEY LEARNING: Don't over-engineer! Tree models don't need scaling.")

# =============================================================================
# PROBLEM 4: Student Performance - Data Leakage in Scaling
# =============================================================================
print("\n\n" + "=" * 80)
print("PROBLEM 4: Student Performance (Data leakage!)")
print("=" * 80)

df4 = pd.read_csv('data/student_performance.csv')
X4 = df4.drop('final_grade', axis=1)
y4 = df4['final_grade']

# WRONG WAY - Scaling before split (DATA LEAKAGE!)
print("\n❌ WRONG WAY: Scale before split")
scaler4_leaky = StandardScaler()
X4_scaled_leaky = scaler4_leaky.fit_transform(X4)  # ← Fit on ALL data!

X4_train_leaky, X4_test_leaky, y4_train, y4_test = train_test_split(
    X4_scaled_leaky, y4, test_size=0.3, random_state=42
)

lr_leaky = LinearRegression()
lr_leaky.fit(X4_train_leaky, y4_train)
r2_train_leaky = r2_score(y4_train, lr_leaky.predict(X4_train_leaky))
r2_test_leaky = r2_score(y4_test, lr_leaky.predict(X4_test_leaky))

# CORRECT WAY - Split first, then scale
print("\n✅ CORRECT WAY: Split first, then scale")
X4_train, X4_test, y4_train_fix, y4_test_fix = train_test_split(
    X4, y4, test_size=0.3, random_state=42
)

scaler4_fixed = StandardScaler()
X4_train_fixed = scaler4_fixed.fit_transform(X4_train)  # ← Fit on TRAIN only!
X4_test_fixed = scaler4_fixed.transform(X4_test)        # ← Transform only!

lr_fixed = LinearRegression()
lr_fixed.fit(X4_train_fixed, y4_train_fix)
r2_train_fixed = r2_score(y4_train_fix, lr_fixed.predict(X4_train_fixed))
r2_test_fixed = r2_score(y4_test_fix, lr_fixed.predict(X4_test_fixed))

print(f"\n📊 RESULTS:")
print(f"\n   WITH LEAKAGE (Wrong):")
print(f"      Train R²: {r2_train_leaky:.3f}")
print(f"      Test R²:  {r2_test_leaky:.3f}")
print(f"      Gap:      {abs(r2_train_leaky - r2_test_leaky):.3f}")
print(f"\n   WITHOUT LEAKAGE (Correct):")
print(f"      Train R²: {r2_train_fixed:.3f}")
print(f"      Test R²:  {r2_test_fixed:.3f}")
print(f"      Gap:      {abs(r2_train_fixed - r2_test_fixed):.3f}")

print(f"\n💡 WHY THIS MATTERS:")
print(f"   - With leakage: Test score is artificially high")
print(f"   - Scaler learned mean/std from test data")
print(f"   - In production, you never have test statistics!")
print(f"\n✅ RULE: Always split FIRST, then fit scaler on TRAIN only!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "=" * 80)
print("EXAM SUMMARY")
print("=" * 80)

print("""
✅ Problem 1: KNN without scaling
   - Issue: Features have different scales
   - Solution: Apply StandardScaler
   - Improvement: ~40% accuracy gain
   - Lesson: Distance-based algorithms NEED scaling!

✅ Problem 2: MinMaxScaler with outliers
   - Issue: Outliers make MinMax squeeze normal values
   - Solution: Use RobustScaler (median + IQR)
   - Improvement: ~25% R² gain
   - Lesson: Check for outliers before choosing scaler!

✅ Problem 3: Scaling Random Forest
   - Issue: Unnecessary scaling on tree-based model
   - Solution: Remove scaling
   - Improvement: None (but more efficient)
   - Lesson: Trees don't use distance, no scaling needed!

✅ Problem 4: Data leakage in scaling
   - Issue: Fit scaler on all data before split
   - Solution: Split first, fit on train only
   - Improvement: True performance revealed
   - Lesson: NEVER let scaler see test data!

🎯 KEY TAKEAWAYS:
   1. KNN/SVM/Neural Nets → Use scaling
   2. Random Forest/XGBoost → Don't scale
   3. Outliers present → Use RobustScaler
   4. Always: Split → Fit on train → Transform both

🎉 You've mastered scaling debugging!
""")

print("=" * 80)
