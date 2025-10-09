# Mock Exam: Model Too Bad 😢

## Overview

This mock exam tests your ability to **diagnose and fix common issues that cause poor model performance**. Unlike data leakage (which gives unrealistically good results), this scenario presents a model with terrible performance due to preprocessing mistakes, poor feature engineering, and improper handling of data quality issues.

## Scenario

You've been handed a machine learning project where the model achieves only **55% accuracy** on a binary classification task - barely better than random guessing! Your job is to identify and fix all the issues causing this poor performance.

## Learning Objectives

After completing this exam, you will be able to:

1. ✅ Identify and fix missing value handling issues
2. ✅ Use appropriate encoding for categorical variables
3. ✅ Handle high-cardinality categorical features
4. ✅ Detect and treat outliers effectively
5. ✅ Choose the right scaler for your data
6. ✅ Create meaningful engineered features
7. ✅ Handle class imbalance properly
8. ✅ Use stratified splitting when appropriate
9. ✅ Evaluate models beyond just accuracy
10. ✅ Build proper preprocessing pipelines

## Files

- **`bad_model_exam.ipynb`** - Interactive notebook with the buggy code
- **`generate_dataset.py`** - Creates the housing dataset with realistic issues
- **`hints.md`** - Hints for each bug (try before looking!)
- **`ANSWER_KEY.md`** - Complete solutions and explanations
- **`README.md`** - This file

## Common Issues Covered

### Data Quality (40% of bugs)
- ❌ Dropping too much data instead of imputing
- ❌ Ignoring outliers
- ❌ Using wrong scaler with outliers

### Feature Engineering (30% of bugs)
- ❌ Wrong categorical encoding (Label vs OneHot)
- ❌ Not handling high-cardinality categoricals
- ❌ No domain-specific feature creation

### Modeling (30% of bugs)
- ❌ Ignoring class imbalance
- ❌ Not using stratified split
- ❌ No hyperparameter tuning
- ❌ Evaluating with wrong metrics

## How to Use

### 1. Generate the Dataset
```python
%run generate_dataset.py
```

### 2. Run the Bad Model
Work through `bad_model_exam.ipynb` and observe:
- ~55% test accuracy
- Poor per-class metrics
- Model biased toward one class

### 3. Identify the Bugs
For each code section marked 🐛 BUG ZONE, identify what's wrong.

### 4. Implement Fixes
In the "Your Fixed Version" section, fix all issues.

### 5. Verify Results
Your fixed model should achieve:
- **80-85% test accuracy**
- **Balanced precision/recall** for both classes
- **F1 score ~0.80-0.82**

## Bug Categories

### Critical (Must Fix)
1. Dropping all missing values
2. Label encoding nominal categories
3. Ignoring class imbalance
4. Outliers with StandardScaler

### Important (Should Fix)
5. No feature engineering
6. High-cardinality categorical
7. Not using stratified split
8. Wrong scaler choice

### Good Practice (Nice to Fix)
9. No hyperparameter tuning
10. Only checking accuracy

## Expected Performance

| Stage | Test Accuracy | Main Issue |
|-------|---------------|------------|
| Original (Bad) | ~55% | All bugs present |
| After fixing missing values | ~58% | Still many issues |
| After fixing encoding | ~65% | Getting better |
| After handling outliers | ~72% | Major improvement |
| After feature engineering | ~77% | Good features help |
| After class balancing | ~82% | Almost there! |
| After hyperparameter tuning | ~85% | Optimal! |

## Time Estimate

- **Quick review**: 30 minutes (identify main issues)
- **Full completion**: 60 minutes (fix everything properly)
- **Deep learning**: 90 minutes (understand why each fix works)

## Prerequisites

You should be familiar with:
- pandas DataFrames
- scikit-learn basics
- Classification metrics
- Common preprocessing techniques

## Difficulty Level

⭐⭐⭐⭐ **Intermediate to Advanced**

This exam is harder than the "model too good" (data leakage) exam because:
- More bugs to find (10+ vs 7)
- Requires deeper understanding of preprocessing
- Need to know multiple techniques (imputation, scaling, encoding, etc.)
- Must understand evaluation metrics

## Tips for Success

1. **Start with EDA** - Look at the data first
2. **Fix one thing at a time** - Track what improves performance
3. **Use pipelines** - Prevent data leakage and make code cleaner
4. **Check class distribution** - Both in target and predictions
5. **Look beyond accuracy** - Precision, recall, F1 matter more here
6. **Think about domain** - What features would be useful for housing prices?

## Real-World Relevance

These bugs are **extremely common** in practice:
- Dropping data is easier than imputing (but wasteful)
- Label encoding is simpler than OneHot (but wrong for nominal)
- StandardScaler is the default (but bad with outliers)
- Accuracy is easy to understand (but misleading with imbalance)

Learning to spot and fix these issues will make you a much better data scientist!

## What You'll Learn

### Technical Skills
- Proper imputation strategies
- Correct categorical encoding
- Outlier detection and treatment
- Robust scaling techniques
- Feature engineering for structured data
- Class imbalance handling

### Best Practices
- Always do EDA first
- Use pipelines to prevent leakage
- Stratify splits with imbalanced data
- Evaluate with appropriate metrics
- Tune hyperparameters systematically

### Debugging Skills
- How to diagnose poor performance
- Which fixes have biggest impact
- How to prioritize improvements
- When to use different techniques

## Next Steps

After completing this exam:

1. ✅ Try the "model_too_good" exam (data leakage)
2. ✅ Apply these techniques to your own projects
3. ✅ Create your own preprocessing pipeline template
4. ✅ Learn about SMOTE, ADASYN for class imbalance
5. ✅ Explore automated feature engineering (featuretools)

## Questions to Reflect On

1. Which bug had the biggest impact on performance?
2. How would you prioritize fixes if you had limited time?
3. What other features could you engineer for housing data?
4. When should you drop high-cardinality categoricals vs encode them?
5. How do you decide between different imputation strategies?

---

**Good luck! Remember: A model performing poorly is often easier to fix than a model with data leakage.** 🚀
