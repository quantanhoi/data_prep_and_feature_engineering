# Mock Exams Overview

This folder contains two complementary mock exams that test different aspects of machine learning best practices.

## 📂 Folder Structure

```
mock_exam/
├── model_too_good/          # Data leakage issues (99% accuracy)
│   ├── data_leakage_exam.ipynb
│   ├── generate_dataset.py
│   ├── hints.md
│   ├── ANSWER_KEY.md
│   └── README.md
│
└── model_too_bad/           # Performance issues (55% accuracy)
    ├── bad_model_exam.ipynb
    ├── generate_dataset.py
    ├── hints.md
    ├── ANSWER_KEY.md
    └── README.md
```

## 🎯 Which Exam Should You Take?

### Model Too Good (Data Leakage) 🚨
**Take this if you want to learn about:**
- Data leakage detection
- Train-test contamination
- Feature engineering timing
- Pipeline construction
- Cross-validation pitfalls

**Difficulty:** ⭐⭐⭐ Intermediate  
**Time:** 45 minutes  
**Bugs to find:** 7  
**Symptom:** Unrealistically high accuracy (99%)

### Model Too Bad (Performance Issues) 😢
**Take this if you want to learn about:**
- Data quality issues
- Preprocessing techniques
- Feature engineering
- Class imbalance handling
- Proper evaluation metrics

**Difficulty:** ⭐⭐⭐⭐ Intermediate-Advanced  
**Time:** 60 minutes  
**Bugs to find:** 10+  
**Symptom:** Terrible performance (55%)

## 📊 Comparison

| Aspect | Model Too Good | Model Too Bad |
|--------|----------------|---------------|
| **Main Issue** | Data leakage | Poor preprocessing |
| **Test Accuracy** | 99.8% (too good!) | 55% (too bad!) |
| **Number of Bugs** | 7 major issues | 10+ issues |
| **Difficulty** | Medium | Medium-Hard |
| **Focus Area** | Pipeline design | Data preprocessing |
| **Key Learning** | Prevent leakage | Improve performance |
| **Real-World** | Catches production bugs | Improves daily work |

## 🎓 Learning Path

### Recommended Order:

1. **Start with Model Too Good**
   - Easier to spot obvious mistakes
   - Critical for production systems
   - Teaches proper workflow

2. **Then do Model Too Bad**
   - Requires more technical knowledge
   - Builds on pipeline concepts
   - More diverse techniques

3. **Compare Your Solutions**
   - See which bugs were harder to find
   - Understand trade-offs
   - Build your own checklist

## 🔍 What Each Exam Teaches

### Model Too Good - Data Leakage

**Critical Bugs:**
1. ❌ Using full dataset statistics before split
2. ❌ Target encoding on full dataset
3. ❌ Imputing before split
4. ❌ Scaling before split
5. ❌ Including target-derived features
6. ❌ Including the target itself
7. ❌ Feature engineering on full dataset

**Key Lesson:** Split first, transform second!

### Model Too Bad - Performance Issues

**Critical Bugs:**
1. ❌ Dropping too much data
2. ❌ Wrong categorical encoding
3. ❌ Ignoring outliers
4. ❌ Wrong scaler choice
5. ❌ No feature engineering
6. ❌ High-cardinality categoricals
7. ❌ No stratified split
8. ❌ Ignoring class imbalance
9. ❌ No hyperparameter tuning
10. ❌ Wrong evaluation metrics

**Key Lesson:** Preprocessing matters more than model choice!

## 💡 Combined Checklist

After completing both exams, use this checklist for your projects:

### Before Modeling
- [ ] Understand your data (EDA)
- [ ] Check for missing values
- [ ] Identify outliers
- [ ] Check class distribution
- [ ] Identify feature types (numeric/categorical)
- [ ] Check for high-cardinality categoricals

### During Preprocessing
- [ ] Split data FIRST (train/test)
- [ ] Use stratified split if imbalanced
- [ ] Impute missing values (don't drop unless necessary)
- [ ] Handle outliers (clip or robust scaling)
- [ ] Encode categoricals correctly (OneHot for nominal)
- [ ] Handle high-cardinality (drop, group, or hash)
- [ ] Scale numeric features appropriately
- [ ] Create domain-specific features

### Pipeline Construction
- [ ] Use scikit-learn Pipeline
- [ ] Fit transformers on training data only
- [ ] Transform both train and test
- [ ] No full-dataset statistics
- [ ] No target information in features

### Modeling
- [ ] Handle class imbalance (class_weight, SMOTE)
- [ ] Tune hyperparameters (GridSearch, RandomizedSearch)
- [ ] Use cross-validation
- [ ] Check for overfitting (train vs test)

### Evaluation
- [ ] Don't rely on accuracy alone
- [ ] Check precision, recall, F1
- [ ] Look at confusion matrix
- [ ] Check per-class metrics
- [ ] Verify predictions make sense

### Before Production
- [ ] Test for data leakage (suspiciously high accuracy?)
- [ ] Test on fresh data
- [ ] Check feature importance makes sense
- [ ] Document preprocessing steps
- [ ] Save pipeline, not just model

## 🏆 Mastery Goals

After both exams, you should be able to:

1. **Spot data leakage** immediately
2. **Handle missing values** appropriately
3. **Encode categoricals** correctly
4. **Deal with outliers** effectively
5. **Build proper pipelines** consistently
6. **Handle class imbalance** when needed
7. **Evaluate models** comprehensively
8. **Debug poor performance** systematically
9. **Prevent common mistakes** proactively
10. **Build production-ready** ML pipelines

## 📚 Further Resources

### After Model Too Good
- Read: "Common Pitfalls in ML" (Domingos, 2012)
- Practice: Kaggle competitions (watch for leaderboard shake-up)
- Study: scikit-learn Pipeline documentation

### After Model Too Bad
- Read: "Feature Engineering for Machine Learning" (Zheng & Casari)
- Practice: Imbalanced-learn library
- Study: Different imputation strategies (KNN, Iterative)

## 🎯 Challenge Yourself

### Expert Mode
1. Fix all bugs in both exams without looking at hints
2. Add additional checks to detect issues automatically
3. Create a custom transformer for each fix
4. Build a "guardrails" library to prevent these bugs

### Create Your Own
1. Build a "model too complex" exam (overfitting)
2. Build a "model too simple" exam (underfitting)
3. Build a "data too messy" exam (real-world chaos)
4. Share with your team or community

## 🤝 Contributing

Have you found:
- Another common bug that should be included?
- A better way to explain a concept?
- Additional test cases?

Feel free to extend these exams!

## 📝 Notes

These exams are based on **real mistakes** seen in production ML systems. Every bug has caused actual problems in real projects. Learning to spot and fix them will save you countless hours of debugging and prevent embarrassing production failures.

---

**Happy Learning!** 🚀

Remember: 
- "Model too good" → Check for leakage
- "Model too bad" → Check your preprocessing
