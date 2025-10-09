# Mock Exam Summary

## 📋 Complete Mock Exam Package

This mock exam simulates a real-world scenario where a data scientist achieves suspiciously high model performance due to **data leakage**. Students must identify and fix all issues.

---

## 🎯 Exam Scenario

**The Problem:**
- A junior data scientist trained a model with **99.8% accuracy**
- Manager is suspicious: "Too good to be true!"
- Task: Find the bugs, explain the issues, fix the code

**Reality Check:**
- After fixes, accuracy should drop to ~75-80%
- This is **realistic and trustworthy** performance
- Demonstrates proper ML methodology

---

## 📁 Files Created

### Core Exam Files
1. **`README.md`** - Exam overview and instructions
2. **`QUICKSTART.md`** - Step-by-step guide to get started
3. **`generate_dataset.py`** - Creates housing dataset with leakage
4. **`leaky_model.py`** - The buggy code (students must fix this)
5. **`solution.py`** - Correct implementation with explanations
6. **`data_leakage_exam.ipynb`** - Interactive Jupyter notebook version

### Support Materials
7. **`hints.md`** - Progressive hints without giving away answers
8. **`ANSWER_KEY.md`** - Detailed explanations of all 7 bugs

---

## 🐛 The 7 Data Leakage Bugs

### Bug #1: Feature Engineering on Full Dataset
**Code:**
```python
df['price_vs_mean'] = df['median_house_value'] / df['median_house_value'].mean()
```
**Issue:** Uses statistics from entire dataset including test data

### Bug #2: Target Encoding Before Split
**Code:**
```python
ocean_proximity_means = df.groupby('ocean_proximity')['high_value'].mean()
```
**Issue:** Uses target variable to encode features before splitting

### Bug #3: Imputation Before Split
**Code:**
```python
num_imputer.fit_transform(df[num_cols])
```
**Issue:** Imputer learns median/mode from entire dataset

### Bug #4: Scaling Before Split
**Code:**
```python
scaler.fit_transform(df[numeric_features])
```
**Issue:** Scaler learns mean/std from entire dataset

### Bug #5: Target Leakage in Features
**Code:**
```python
feature_cols = [..., 'median_house_value', 'price_vs_mean', ...]
```
**Issue:** Includes features that directly or indirectly reveal the target

### Bug #6: Inconsistent Categorical Encoding
**Code:**
Target encoding on full dataset, then OneHot on training data
**Issue:** Same categorical variable treated inconsistently

### Bug #7: Meaningless Cross-Validation
**Code:**
```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
```
**Issue:** CV on data preprocessed with leaked information

---

## ✅ Learning Outcomes

Students who complete this exam will:

1. **Identify** common data leakage patterns
2. **Understand** the critical importance of splitting BEFORE preprocessing
3. **Implement** sklearn Pipeline correctly
4. **Recognize** features that leak target information
5. **Distinguish** between realistic and unrealistic performance
6. **Apply** proper cross-validation methodology
7. **Debug** ML pipelines systematically

---

## 🎓 Grading Rubric

| Score | Criteria |
|-------|----------|
| **90-100%** | Found all 7 bugs, explained clearly, proper Pipeline implementation |
| **75-89%** | Found 5-6 bugs, mostly correct fixes, good understanding |
| **60-74%** | Found 3-4 bugs, partial fixes, basic understanding |
| **<60%** | Found <3 bugs, incomplete fixes, needs more practice |

---

## 📊 Expected Performance

### Before Fixes (Leaky Model)
```
Training Accuracy: 0.9980
Test Accuracy:     0.9980
CV Accuracy:       0.9980 (+/- 0.0005)

Top Features:
1. median_house_value      (0.92) ← LEAKY!
2. price_vs_mean           (0.04) ← LEAKY!
3. price_zscore            (0.02) ← LEAKY!
```

### After Fixes (Proper Model)
```
Training Accuracy: 0.8234
Test Accuracy:     0.7856
CV Accuracy:       0.7791 (+/- 0.0123)

Top Features:
1. median_income           (0.38) ✓
2. latitude                (0.15) ✓
3. housing_median_age      (0.12) ✓
```

---

## 🔑 Key Concepts Covered

### From Course Material
- **Train/Test Split** (Chapter 11)
- **Scaling Numerical** (scaling_numerical/)
- **Categorical Variables** (categorical_variables/)
- **Missing Data** (missing_data/)
- **Feature Engineering** (Multiple chapters)
- **Pipelines** (Throughout course)

### Data Leakage Patterns
1. **Temporal leakage** - Using future information
2. **Train-test contamination** - Preprocessing before split
3. **Target leakage** - Features derived from target
4. **Overfitting to CV folds** - Improper cross-validation

---

## 💻 How to Use This Exam

### For Instructors
1. **As a quiz:** 45-minute in-class exercise
2. **As homework:** Take-home assignment with written explanations
3. **As discussion:** Code review exercise in pairs/groups
4. **As practice:** Self-study with hints and solutions

### For Students
1. **Time yourself:** Try to complete in 45 minutes
2. **Don't peek:** Resist looking at solution.py immediately
3. **Use hints:** Progressive hints in hints.md
4. **Reflect:** Write down what you learned
5. **Practice:** Try creating your own leakage examples

---

## 🚀 Running the Exam

### Quick Start
```bash
cd src/mock_exam

# Generate dataset
python generate_dataset.py

# Option 1: Run buggy code
python leaky_model.py

# Option 2: Interactive notebook
jupyter notebook data_leakage_exam.ipynb

# Compare with solution
python solution.py
```

### Expected Output
```
Dataset generated: housing_with_target_leakage.csv
Shape: (20640, 10)

⚠️ Leaky model achieves 99.8% accuracy
✅ Fixed model achieves 78.6% accuracy

Achievement unlocked: Found all data leakage bugs! 🎉
```

---

## 🎯 Real-World Relevance

### Why This Matters

1. **Kaggle competitions:** Data leakage is #1 cause of failed submissions
2. **Production ML:** Leaky models fail catastrophically in production
3. **Interviews:** Common technical interview question
4. **Research:** Invalid experiments from improper methodology
5. **Business impact:** Wasted resources on unrealistic models

### Common Real-World Examples

- **Finance:** Using future stock prices to predict current prices
- **Healthcare:** Using test results that aren't available at prediction time
- **Marketing:** Using conversion data to predict conversion likelihood
- **Fraud detection:** Using investigation outcomes to train detector

---

## 📚 Additional Resources

### Recommended Reading
1. Sklearn Pipeline documentation
2. "Feature Engineering for Machine Learning" by Alice Zheng
3. Kaggle Learn: Data Leakage course
4. "Rules of Machine Learning" by Martin Zinkevich (Google)

### Related Topics to Explore
- Cross-validation strategies
- Time-series cross-validation
- Target encoding best practices
- Feature selection without leakage

---

## 🎓 Extension Activities

### For Advanced Students

1. **Add more bugs:** Create additional leakage scenarios
2. **Time series version:** Create temporal leakage examples
3. **Custom transformers:** Build safe feature engineering transformers
4. **Production pipeline:** Deploy the fixed model with proper preprocessing
5. **Automated detection:** Write tests to detect leakage

### Challenge Problems

1. Implement proper target encoding with cross-validation
2. Create a leakage detection function
3. Build a completely safe ML pipeline template
4. Explain when "leakage" might actually be intentional
5. Design tests to catch common leakage patterns

---

## ✨ Key Takeaway

> **"If your model is too good to be true, it probably is."**

Data leakage is one of the most common and dangerous mistakes in machine learning. This exam teaches students to:
- Recognize the signs
- Find the causes
- Fix the issues
- Prevent future leakage

---

## 📞 Support

If you encounter issues or have questions:
1. Check QUICKSTART.md for setup help
2. Read hints.md for progressive hints
3. Consult ANSWER_KEY.md for detailed explanations
4. Review the solution.py implementation

---

**Created:** January 2025  
**Course:** Data Preparation and Feature Engineering  
**Topic:** Data Leakage Detection and Prevention  
**Difficulty:** Intermediate  
**Time:** 45 minutes  
**Format:** Hands-on coding exercise
