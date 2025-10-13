# Scaling Issues Mock Exam - Complete Package

## 📚 Navigation Guide

### 🎯 Start Here
- **[scaling_exam.ipynb](scaling_exam.ipynb)** - Main exam notebook with 4 buggy scenarios

### 📖 Reference Materials
1. **[README.md](README.md)** - Overview and objectives
2. **[SCALING_GUIDE.md](SCALING_GUIDE.md)** - Comprehensive guide on when to use which scaler
3. **[hints.md](hints.md)** - Task-specific hints when you're stuck

### ✅ Solutions
1. **[solution.py](solution.py)** - Complete working solutions with output
2. **[ANSWER_KEY.md](ANSWER_KEY.md)** - Detailed explanations and grading rubric

### 📊 Datasets
- **[data/](data/)** - All generated datasets with intentional scaling issues

---

## 🎯 Exam Overview

### Four Real-World Scaling Problems

| Problem | Dataset | Algorithm | Issue | Fix |
|---------|---------|-----------|-------|-----|
| **1** | Credit Fraud | KNN | No scaling | Add StandardScaler |
| **2** | House Prices | Linear Regression | MinMax + outliers | Use RobustScaler |
| **3** | Customer Churn | Random Forest | Unnecessary scaling | Remove scaling |
| **4** | Student Performance | Linear Regression | Scaling leakage | Split first, then scale |

---

## 🚀 Quick Start

```bash
# 1. Generate datasets (if needed)
python generate_datasets.py

# 2. Open the exam notebook
# Open scaling_exam.ipynb in Jupyter or VS Code

# 3. Work through all 4 problems

# 4. Check solutions
python solution.py
```

---

## 📋 What You'll Learn

### Core Concepts

1. **When to Scale**
   - ✅ KNN, SVM, Neural Networks, Logistic Regression
   - ❌ Random Forest, XGBoost, Decision Trees

2. **Which Scaler to Use**
   - **StandardScaler**: Normal distribution, no outliers
   - **MinMaxScaler**: Bounded range [0,1], no outliers
   - **RobustScaler**: Data with outliers

3. **Proper Scaling Workflow**
   - Split data **first**
   - Fit scaler on **training** data only
   - Transform **both** train and test

4. **Data Leakage Prevention**
   - Never fit scaler on entire dataset before split
   - Test data should never influence scaler parameters

---

## 📊 Problem Details

### Problem 1: Missing Scaling (KNN) ⭐⭐
**Difficulty:** Medium  
**Key Concept:** Distance-based algorithms need scaling  
**Performance Impact:** +30-40% accuracy  
**Time:** 10-12 minutes

**Symptoms:**
- KNN accuracy ~50-60% (barely better than random)
- Features have very different scales

**Fix:** Apply StandardScaler

---

### Problem 2: Wrong Scaler (Outliers) ⭐⭐⭐
**Difficulty:** Medium-Hard  
**Key Concept:** Outliers break MinMaxScaler  
**Performance Impact:** +20-30% R²  
**Time:** 12-15 minutes

**Symptoms:**
- Poor R² score (~0.65)
- Data has outliers in some features
- MinMaxScaler squeezes normal values

**Fix:** Switch to RobustScaler

---

### Problem 3: Unnecessary Scaling (Trees) ⭐
**Difficulty:** Easy  
**Key Concept:** Tree models don't need scaling  
**Performance Impact:** Minimal (but correct approach)  
**Time:** 8-10 minutes

**Symptoms:**
- Random Forest with scaling
- Performance is okay but scaling is wasteful

**Fix:** Remove scaling entirely

---

### Problem 4: Data Leakage ⭐⭐⭐⭐
**Difficulty:** Hard (conceptually important!)  
**Key Concept:** Never fit scaler before split  
**Performance Impact:** Reveals true performance  
**Time:** 15-18 minutes

**Symptoms:**
- Test score suspiciously close to train score
- Scaler fit on entire dataset before split

**Fix:** Split first, then fit scaler on train only

---

## 🎓 Learning Path

### Before the Exam
1. Read [README.md](README.md) for context
2. Review [SCALING_GUIDE.md](SCALING_GUIDE.md) - especially the decision tree
3. Understand the 3 main scalers: Standard, MinMax, Robust

### During the Exam
1. Work through [scaling_exam.ipynb](scaling_exam.ipynb)
2. For each problem:
   - Run the buggy code
   - Investigate the issue
   - Identify the root cause
   - Apply the fix
   - Compare performance
3. Use [hints.md](hints.md) if stuck (try 5-10 min first!)

### After the Exam
1. Run [solution.py](solution.py) to see all solutions
2. Read [ANSWER_KEY.md](ANSWER_KEY.md) for explanations
3. Review any concepts you struggled with
4. Practice on your own datasets!

---

## 🔑 Key Decision Trees

### Algorithm → Scaling Decision
```
What algorithm am I using?
│
├─ KNN, SVM, Neural Network?
│  └─ ✅ YES, scale!
│
├─ Logistic Regression, Linear Regression?
│  └─ ✅ YES, scale!
│
└─ Random Forest, XGBoost, Decision Tree?
   └─ ❌ NO, don't scale!
```

### Data → Scaler Decision
```
Do I have outliers?
│
├─ YES → Use RobustScaler or Winsorizing
│
└─ NO
   │
   ├─ Normal distribution? → StandardScaler (Z-score)
   ├─ Need [0,1] range? → MinMaxScaler
   └─ Highly skewed? → Box-Cox transformation
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Scaling Everything
```python
# ❌ Don't do this
if model:
    scale()  # Wrong mindset!

# ✅ Do this
if algorithm_uses_distance_or_gradients:
    scale()
```

### Pitfall 2: Wrong Order
```python
# ❌ Wrong
scaler.fit(X)
X_train, X_test = split(X_scaled)

# ✅ Correct
X_train, X_test = split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 3: fit_transform on Test
```python
# ❌ Wrong
X_test_scaled = scaler.fit_transform(X_test)

# ✅ Correct
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 4: Ignoring Outliers
```python
# ❌ Don't blindly use MinMaxScaler
scaler = MinMaxScaler()  # Breaks with outliers!

# ✅ Check for outliers first
df.boxplot()
# If outliers exist, use RobustScaler
```

---

## 📈 Expected Performance Gains

| Problem | Before | After | Gain | Metric |
|---------|--------|-------|------|--------|
| 1: KNN | 0.55 | 0.92 | +37% | Accuracy |
| 2: Outliers | 0.67 | 0.91 | +24% | R² |
| 3: Trees | 0.83 | 0.83 | 0% | Accuracy |
| 4: Leakage | 0.91* | 0.87 | -4%* | R² |

*Problem 4 shows a "drop" but this reveals the **true** performance (the leaky version was artificially high)

---

## 🎯 Grading Rubric

Each problem worth 25 points:

### Problem 1 (25 pts)
- Identify scale differences (10 pts)
- Apply StandardScaler correctly (10 pts)
- Follow fit/transform pattern (5 pts)

### Problem 2 (25 pts)
- Visualize outliers (5 pts)
- Explain MinMax problem (10 pts)
- Use RobustScaler correctly (10 pts)

### Problem 3 (25 pts)
- Recognize tree-based model (10 pts)
- Explain why scaling unnecessary (10 pts)
- Remove scaling (5 pts)

### Problem 4 (25 pts)
- Identify leakage (10 pts)
- Explain the issue (10 pts)
- Implement correct order (5 pts)

**Total: 100 points**

---

## 💡 Pro Tips

1. **Always visualize first**
   ```python
   df.describe()
   df.hist()
   df.boxplot()
   ```

2. **Check for outliers**
   ```python
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   ```

3. **Know your algorithm**
   - Distance-based? → Scale
   - Tree-based? → Don't scale

4. **Remember the workflow**
   - Split → Fit → Transform

5. **Test both approaches**
   ```python
   # Compare scalers
   for scaler in [StandardScaler(), RobustScaler(), MinMaxScaler()]:
       # Train and evaluate
   ```

---

## 📝 Quick Reference Card

### When to Scale?
- ✅ KNN, SVM
- ✅ Neural Networks
- ✅ Logistic/Linear Regression
- ❌ Random Forest
- ❌ XGBoost, LightGBM
- ❌ Decision Trees

### Which Scaler?
- **Normal data** → StandardScaler (Z-score)
- **Outliers** → RobustScaler or Winsorizing
- **Need [0,1]** → MinMaxScaler (if no outliers)
- **Extreme outliers** → Clipping + MinMaxScaler
- **Highly skewed** → Box-Cox transformation

### Proper Workflow
1. Split data
2. Fit scaler on **train only**
3. Transform **both** train and test

---

## 🎉 Success Criteria

You've mastered scaling if you can:
- ✅ Identify which algorithms need scaling
- ✅ Choose the right scaler for your data
- ✅ Detect and handle outliers
- ✅ Avoid data leakage in scaling
- ✅ Explain why each scaling decision matters

---

## 📚 Additional Resources

- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Scaling Best Practices](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)
- Original exam: `src/mock_exam/scaling_issues/`

---

**Ready to debug some scaling issues?**  
**Open `scaling_exam.ipynb` and let's get started!** 🚀

Good luck! 🍀
