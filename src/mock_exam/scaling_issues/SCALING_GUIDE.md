# Scaling Guide: When to Use Which Scaler

## 🎯 Quick Decision Tree

```
Does your algorithm use DISTANCE/GRADIENT calculations?
│
├─ YES → You need scaling!
│   │
│   ├─ Step 1: Check for OUTLIERS
│   │   │
│   │   ├─ FEW extreme outliers that are ERRORS?
│   │   │   └─ Use CLIPPING + MinMaxScaler (remove bad data points)
│   │   │
│   │   ├─ Some outliers but want to KEEP them?
│   │   │   ├─ Use WINSORIZING (cap at percentiles, e.g., 5th-95th)
│   │   │   └─ Or use RobustScaler (median + IQR, resistant to outliers)
│   │   │
│   │   └─ NO outliers → Continue to Step 2
│   │
│   ├─ Step 2: Check DATA DISTRIBUTION
│   │   │
│   │   ├─ Data is NORMALLY distributed?
│   │   │   └─ Use StandardScaler (Z-score normalization)
│   │   │
│   │   ├─ Data is HIGHLY SKEWED (right/left tail)?
│   │   │   └─ Use Box-Cox transformation (makes distribution more normal)
│   │   │
│   │   └─ Need values in SPECIFIC RANGE [0,1]?
│   │       └─ Use MinMaxScaler (ONLY if no outliers!)
│   │
│   └─ Still unsure? → Start with StandardScaler (Z-score)
│
└─ NO (Tree-based: Random Forest, XGBoost) → DON'T scale!
```

---

## 📊 Scaler Comparison Table

| Scaler | Formula | Range | Best For | Outlier Sensitive? |
|--------|---------|-------|----------|-------------------|
| **StandardScaler (Z-score)** | $(x - \mu) / \sigma$ | Unbounded | Normal distribution, SVM, Logistic Regression | ⚠️ YES |
| **MinMaxScaler** | $(x - min) / (max - min)$ | [0, 1] | Neural networks, bounded features | ⚠️⚠️ VERY |
| **RobustScaler** | $(x - median) / IQR$ | Unbounded | Data with outliers | ✅ NO |
| **Clipping + MinMax** | Clip outliers, then MinMax | [0, 1] | Data with extreme outliers to remove | ⚠️ Clips outliers |
| **Winsorizing** | Cap outliers at percentile | Original scale | Reduce outlier impact, keep values | Reduces outliers |
| **Box-Cox** | $\frac{x^\lambda - 1}{\lambda}$ | Unbounded | Make skewed data more normal | Transforms distribution |

---

## 📋 When to Use Each Technique - Detailed Guidelines

### 1. StandardScaler (Z-score) - Your Default Choice ⭐
**Use when:**
- ✅ Algorithm needs scaling (KNN, SVM, Neural Networks, Logistic Regression)
- ✅ Data is approximately **normally distributed** (bell curve)
- ✅ **No major outliers** in your data
- ✅ You want features centered at 0 with std deviation of 1

**Don't use when:**
- ❌ Data has **significant outliers** (use RobustScaler instead)
- ❌ Using **tree-based models** (Random Forest, XGBoost)
- ❌ Data is **highly skewed** (use Box-Cox first)

**Example scenario:** Customer age (20-80), income ($30k-$150k) for KNN classifier - fairly normal distribution, no crazy outliers.

---

### 2. MinMaxScaler - For Bounded Ranges
**Use when:**
- ✅ You **specifically need [0,1] range** (or custom range)
- ✅ Neural networks with **sigmoid/tanh activation** functions
- ✅ **Image pixel data** (already 0-255, scale to 0-1)
- ✅ Data is **uniformly distributed** or bounded
- ✅ **NO OUTLIERS** in your data!

**Don't use when:**
- ❌ Data has **any outliers** (they will squash everything!)
- ❌ You don't specifically need bounded range (use StandardScaler)

**Example scenario:** Image processing (pixel values 0-255 → 0-1), or features already bounded like percentages (0-100%).

**⚠️ WARNING:** One outlier ruins everything!
```
Original: [10, 20, 30, 40, 1000]  ← 1000 is outlier
Scaled:   [0.00, 0.01, 0.02, 0.03, 1.00]  ← Everything else crushed to 0!
```

---

### 3. RobustScaler - The Outlier Fighter
**Use when:**
- ✅ Your data has **moderate outliers** you want to **keep**
- ✅ Outliers represent **real data** (not errors)
- ✅ Algorithm needs scaling (KNN, SVM, etc.)
- ✅ You want scaling that's **resistant to outliers**

**Don't use when:**
- ❌ No outliers (just use StandardScaler, it's simpler)
- ❌ Outliers are **measurement errors** (use Clipping instead)

**Example scenario:** House prices with a few mansions (legitimate outliers), medical data with some extreme but valid measurements.

**How it works:** Uses **median** (not mean) and **IQR** (not std), so outliers don't influence the scaling.

---

### 4. Clipping + MinMaxScaler - Remove Bad Outliers
**Use when:**
- ✅ You have **extreme outliers** that are likely **ERRORS**
- ✅ You want to **remove** outliers completely
- ✅ You know the **valid range** for your feature
- ✅ Then need [0,1] scaling after clipping

**Don't use when:**
- ❌ Outliers are **valid data points**
- ❌ You're not sure what to clip at

**Example scenario:**
```python
# Age should be 0-120, but you have entries like 999 or -5 (errors)
df['age'] = df['age'].clip(lower=0, upper=120)

# Then scale to [0,1]
scaler = MinMaxScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
```

**When to use:** Sensor data with measurement errors, age with impossible values, percentages > 100%.

---

### 5. Winsorizing - Cap Outliers, Keep Values
**Use when:**
- ✅ You have **some outliers** you want to **reduce impact of**
- ✅ Don't want to remove data completely
- ✅ Want to **cap extreme values** at percentiles
- ✅ Outliers are real but too extreme

**Don't use when:**
- ❌ You want to keep original values (use RobustScaler)
- ❌ Outliers are errors (use Clipping)

**Example scenario:**
```python
# Cap at 5th and 95th percentile
from scipy.stats.mstats import winsorize
df['income_winsorized'] = winsorize(df['income'], limits=[0.05, 0.05])

# Income of $1M becomes max of 95th percentile (e.g., $200k)
# Income of $5k becomes min of 5th percentile (e.g., $25k)
```

**Use case:** Income data (billionaires skew it), test scores (a few perfect/failed outliers), response times (some extremely slow).

**Difference from Clipping:**
- Clipping: You set hard limits (e.g., 0-120)
- Winsorizing: Limits based on data percentiles (e.g., 5th-95th)

---

### 6. Box-Cox Transformation - Fix Skewed Distributions
**Use when:**
- ✅ Your data is **highly skewed** (long right or left tail)
- ✅ Algorithm assumes **normal distribution** (Linear Regression, LDA)
- ✅ Want to make data **more normal/symmetric**
- ✅ Data has **only positive values** (Box-Cox requires x > 0)

**Don't use when:**
- ❌ Data is already **normally distributed**
- ❌ Data has **zero or negative values** (Box-Cox requires positive)
- ❌ Using **tree-based models** (they don't care about distribution)

**Example scenario:**
```python
from scipy import stats

# Right-skewed data: income, house prices, population
income_transformed, lambda_param = stats.boxcox(df['income'])

# Lambda tells you the transformation:
# λ = 1   → No transformation
# λ = 0.5 → Square root
# λ = 0   → Log
# λ = -1  → Reciprocal
```

**Use case:** House prices (most cheap, few expensive), income (long right tail), count data that's skewed.

**After Box-Cox:** Often follow with StandardScaler to center and scale the transformed data.

---

## 🎯 Decision Matrix - Quick Reference

| Your Situation | Best Choice | Why |
|----------------|-------------|-----|
| Normal data, KNN/SVM | **StandardScaler** | Default for distance-based algorithms |
| Data with outliers, KNN/SVM | **RobustScaler** | Resistant to outliers |
| Neural network, bounded range | **MinMaxScaler** | Need [0,1] for activation functions |
| Measurement errors in data | **Clipping** then MinMax | Remove invalid values |
| Real outliers, but too extreme | **Winsorizing** | Reduce impact without removing |
| Right-skewed distribution | **Box-Cox** then StandardScaler | Make normal, then scale |
| Random Forest / XGBoost | **None** | Trees don't need scaling! |
| Not sure / starting out | **StandardScaler** | Safe default choice |

---

## 🔄 Common Workflows

### Workflow 1: Standard Distance-Based Model
```python
# For KNN, SVM, Logistic Regression with clean data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Workflow 2: Data with Outliers
```python
# Option A: RobustScaler (keep outliers)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Option B: Winsorize then StandardScaler
from scipy.stats.mstats import winsorize

X_train_winsorized = winsorize(X_train, limits=[0.05, 0.05], axis=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_winsorized)
```

### Workflow 3: Skewed Data
```python
# Box-Cox then StandardScaler
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Transform skewed features
for col in skewed_columns:
    X_train[col], lambda_param = stats.boxcox(X_train[col])
    X_test[col] = stats.boxcox(X_test[col], lmbda=lambda_param)

# Then scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Workflow 4: Neural Network
```python
# MinMaxScaler for bounded range
from sklearn.preprocessing import MinMaxScaler

# First check for outliers!
# If outliers exist, handle them first

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🤖 Algorithm → Scaler Mapping

### ✅ NEED SCALING (Distance/Gradient-Based)

| Algorithm | Recommended Scaler | Why? |
|-----------|-------------------|------|
| **KNN** | StandardScaler or RobustScaler | Uses Euclidean distance |
| **SVM** | StandardScaler | Uses distance to hyperplane |
| **Logistic Regression** | StandardScaler | Gradient descent optimization |
| **Neural Networks** | MinMaxScaler or StandardScaler | Gradient descent, activation functions |
| **Linear Regression** | StandardScaler | Helps with convergence |
| **PCA** | StandardScaler | Variance-based |
| **K-Means** | StandardScaler | Uses Euclidean distance |
| **Gradient Boosting (sklearn)** | StandardScaler (optional) | Can help but not required |

### ❌ DON'T NEED SCALING (Tree-Based)

| Algorithm | Scaling Needed? | Why? |
|-----------|----------------|------|
| **Random Forest** | ❌ NO | Uses splits, not distances |
| **Decision Trees** | ❌ NO | Uses thresholds |
| **XGBoost** | ❌ NO | Tree-based |
| **LightGBM** | ❌ NO | Tree-based |
| **CatBoost** | ❌ NO | Tree-based |

---

## 📏 Detailed Scaler Guide

### 1. StandardScaler (Z-score Normalization)

**Formula:** $z = \frac{x - \mu}{\sigma}$

**When to use:**
- ✅ Data is approximately normally distributed
- ✅ Algorithm uses distance (KNN, SVM)
- ✅ Algorithm uses gradient descent (Logistic Regression, Neural Nets)
- ✅ No significant outliers

**Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

**Result:**
- Mean = 0
- Standard deviation = 1
- Range: Unbounded (typically -3 to +3)

---

### 2. MinMaxScaler

**Formula:** $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**When to use:**
- ✅ Need features in specific range [0, 1]
- ✅ Neural networks with sigmoid/tanh activation
- ✅ Image data
- ✅ Data is uniformly distributed
- ❌ DON'T use with outliers!

**Example:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Problem with outliers:**
```
Original: [1, 2, 3, 4, 100]  # 100 is outlier
Scaled:   [0.00, 0.01, 0.02, 0.03, 1.00]  # Everything squeezed near 0!
```

---

### 3. RobustScaler

**Formula:** $x_{scaled} = \frac{x - median}{IQR}$

**When to use:**
- ✅ Data has outliers
- ✅ Distance-based algorithms
- ✅ Median and IQR more robust than mean/std

**Example:**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why better for outliers:**
```
Original: [1, 2, 3, 4, 100]
- StandardScaler: Outlier affects mean and std
- RobustScaler: Uses median (3) and IQR (2), outlier has less impact
```

---

### 4. No Scaling (Tree-Based Models)

**When to use:**
- ✅ Random Forest
- ✅ Decision Trees
- ✅ XGBoost, LightGBM, CatBoost

**Why no scaling needed:**
Trees use splits like "age > 30", not distances. Whether age is 30 or 0.3 (scaled) doesn't matter—the split point adjusts accordingly.

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier

# No scaling needed!
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Use original data
```

---

## ⚠️ Critical: Train/Test Scaling

### ❌ WRONG (Data Leakage)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# WRONG: Fit on all data before split!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← Uses test data statistics!

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**Problem:** Test data statistics leak into training data!

### ✅ CORRECT
```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Then fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ← Fit on train only!
X_test_scaled = scaler.transform(X_test)        # ← Transform only!
```

**Key rule:** `fit_transform()` on train, `transform()` on test!

---

## 🔍 How to Identify Scaling Issues

### Issue 1: Poor performance on distance-based model
```python
# Symptom: KNN accuracy < 60% on easy dataset
# Check: Do features have different scales?
df.describe()  # Look at mean/std differences
```

**Solution:** Add StandardScaler or RobustScaler

### Issue 2: Model sensitive to outliers
```python
# Symptom: Performance varies wildly with different samples
# Check: Are there outliers?
df.boxplot()  # Visual check
```

**Solution:** Switch from MinMaxScaler to RobustScaler

### Issue 3: Tree model performs worse with scaling
```python
# Symptom: Random Forest accuracy drops after scaling
# Check: Are you using a tree-based model?
```

**Solution:** Remove scaling!

### Issue 4: Test accuracy much lower than train
```python
# Symptom: Train acc = 95%, Test acc = 70%
# Check: Did you fit scaler on all data?
```

**Solution:** Fix train/test scaling order

---

## 💡 Pro Tips

1. **Always check data distribution first**
   ```python
   df.hist(bins=50, figsize=(15, 10))
   ```

2. **Check for outliers**
   ```python
   df.boxplot()
   ```

3. **Compare scalers**
   ```python
   # Try multiple scalers and compare performance
   scalers = {
       'standard': StandardScaler(),
       'minmax': MinMaxScaler(),
       'robust': RobustScaler()
   }
   ```

4. **Remember the order**
   - Split data first
   - Fit scaler on train only
   - Transform both train and test

5. **Tree-based models are your friend**
   - No scaling needed
   - Handle mixed scales naturally
   - Try Random Forest first!

---

## 📝 Quick Checklist

Before training:
- [ ] Is my algorithm distance-based? → Need scaling
- [ ] Does my data have outliers? → Use RobustScaler
- [ ] Is my data normally distributed? → Use StandardScaler
- [ ] Am I using a tree model? → Don't scale
- [ ] Did I split before scaling? → Avoid leakage
- [ ] Did I fit only on training data? → Avoid leakage

---

## 🎯 Common Scenarios

| Scenario | Solution |
|----------|----------|
| KNN on mixed-scale features | StandardScaler |
| Neural network on images | MinMaxScaler (0-1) |
| SVM with salary + age features | StandardScaler |
| House prices with outliers | RobustScaler |
| Random Forest on any data | No scaling |
| Logistic regression | StandardScaler |
| Time series with different scales | StandardScaler or MinMaxScaler |
| Text features (TF-IDF) | Usually already normalized |

---

**Remember:** When in doubt, try StandardScaler first for distance-based models, and no scaling for tree-based models!
