# Mock Exam: Data Leakage Detection and Fix

## Scenario
You are a junior data scientist who just trained a model that achieves **99.8% accuracy** on both training and test sets! Your manager is suspicious and asks you to investigate why the results are "too good to be true."

## Your Task
1. **Detect** the data leakage issues in the provided code
2. **Explain** what went wrong and why the model performs unrealistically well
3. **Fix** all the issues to get realistic performance metrics

## Files in this Exam
- `leaky_model.py` - The problematic code with data leakage issues (FIND THE BUGS!)
- `solution.py` - The corrected version (DO NOT LOOK until you've tried!)
- `housing_with_target_leakage.csv` - Sample dataset with intentional issues

## Common Data Leakage Patterns You Should Look For

### 1. **Fitting Transformers Before Train/Test Split**
```python
# WRONG: Scaler sees test data!
scaler.fit(X)  
X_train, X_test = train_test_split(X)

# CORRECT: Fit only on training data
X_train, X_test = train_test_split(X)
scaler.fit(X_train)
```

### 2. **Target Leakage in Features**
Features that directly or indirectly contain information about the target that wouldn't be available at prediction time.

### 3. **Using Future Information**
Including data from the future in time-series predictions.

### 4. **Improper Cross-Validation**
Not respecting the train/test boundary during CV.

### 5. **Feature Engineering on Full Dataset**
Creating features using statistics from the entire dataset.

## Success Criteria
- You identify **at least 3 major data leakage issues**
- You fix the code to achieve realistic performance (around 70-85% accuracy)
- You can explain each issue in your own words

## Time Limit
45 minutes

## Hints
- Check where `fit()` and `fit_transform()` are called
- Look for suspicious features that seem too predictive
- Examine the order of operations carefully
- Think about what information would be available in production

Good luck! 🍀
