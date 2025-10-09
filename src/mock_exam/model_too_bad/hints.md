# Mock Exam: Model Too Bad - Hints

## If you're stuck, here are some hints for each bug:

### Bug #1: Missing Values
**Hint:** Dropping all rows with ANY missing value wastes data. Consider:
- Which columns have missing values?
- Can you impute them instead?
- Should you impute before or after splitting?

### Bug #2: Categorical Encoding
**Hint:** Label Encoding creates artificial order (0 < 1 < 2...)
- Is there a natural order in ocean_proximity? No!
- What encoding preserves independence between categories?
- What about the high-cardinality block_id?

### Bug #3: High Cardinality
**Hint:** 5000 unique blocks is too many for one-hot encoding
- Should you even use block_id as a feature?
- Could you group blocks by location?
- Or just drop it if it's not useful?

### Bug #4: Outliers
**Hint:** Population has extreme outliers (10x normal values)
- How do outliers affect StandardScaler?
- What scaler is robust to outliers?
- Should you clip or remove extreme values?

### Bug #5: Feature Engineering
**Hint:** Raw features aren't always the most informative
- avg_rooms / avg_occupancy = rooms per person?
- lat/lon: Can you create distance features?
- Polynomial features for non-linear relationships?

### Bug #6: Scaling Choice
**Hint:** StandardScaler assumes no outliers
- RobustScaler uses median and IQR (robust to outliers)
- Or handle outliers first, then use StandardScaler

### Bug #7: Train-Test Split
**Hint:** With imbalanced classes (30% vs 70%)
- Use `stratify=y` to maintain class proportions
- Otherwise test set might have different distribution

### Bug #8: Class Imbalance
**Hint:** Model predicts majority class too often
- Use `class_weight='balanced'` in model
- Or try SMOTE for oversampling
- Or use different evaluation metrics (F1, not just accuracy)

### Bug #9: Model Selection
**Hint:** Logistic Regression is simple but limited
- Try Random Forest or Gradient Boosting
- They handle non-linear relationships better
- And are less sensitive to outliers

### Bug #10: No Hyperparameter Tuning
**Hint:** Default parameters are rarely optimal
- Use GridSearchCV or RandomizedSearchCV
- Tune C (regularization) for LogisticRegression
- Tune n_estimators, max_depth for trees

## Still stuck? Try this order:

1. **First priority:** Fix data quality (imputation, outliers)
2. **Second priority:** Fix encoding (use OneHotEncoder)
3. **Third priority:** Create useful features
4. **Fourth priority:** Use proper pipeline to avoid leakage
5. **Fifth priority:** Handle class imbalance
6. **Sixth priority:** Tune hyperparameters

## Expected improvements:

- After fixing missing values: ~58% → ~62%
- After fixing encoding: ~62% → ~68%
- After fixing outliers/scaling: ~68% → ~73%
- After feature engineering: ~73% → ~77%
- After handling class imbalance: ~77% → ~82%
- After hyperparameter tuning: ~82% → ~85%

Good luck! 🍀
