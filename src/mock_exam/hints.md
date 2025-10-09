# Hints for Finding Data Leakage Issues

## Use these hints if you're stuck! (Try to find bugs yourself first)

### Hint 1: Feature Engineering Section
Look at lines where features are created using `df.mean()`, `df.std()`, or `groupby()`. 
- Are these statistics computed on the ENTIRE dataset?
- What information is leaking from the test set?

### Hint 2: The Most Obvious Leak
There's a feature in the dataset that directly contains information about the target variable.
- What does "high_value" depend on?
- Which column in the data would tell you almost exactly whether a house is high-value or not?

### Hint 3: Order of Operations
Look at the order in which operations are performed:
1. Where is `train_test_split()` called?
2. Where is `.fit()` or `.fit_transform()` called?
3. Which one should come first?

### Hint 4: Imputation
The code calls `SimpleImputer.fit_transform()` on `df` before splitting.
- What statistics does the imputer learn?
- Should the imputer see test data?

### Hint 5: Scaling
Similar to imputation, check where `StandardScaler` is fitted.
- Does it see the full dataset or just training data?
- What statistics does it learn (mean, std)?

### Hint 6: Target Encoding
Look for any encoding that uses the target variable `y` or `high_value`.
- Is this done before or after the split?
- Does it use test set labels?

### Hint 7: Cross-Validation
Even if you fix everything else, the cross-validation at the end is problematic.
- What data is it being run on?
- Has that data already been preprocessed using information from the full dataset?

## Questions to Ask Yourself

1. **For each `.fit()` or `.fit_transform()` call:**
   - What data is it being fitted on?
   - Should this include test data?
   - What statistics or patterns is it learning?

2. **For each feature:**
   - Would this feature be available in production?
   - Does it use information from the entire dataset?
   - Does it depend on the target variable?

3. **For the overall pipeline:**
   - What's the correct order: split then preprocess, or preprocess then split?
   - Are we keeping train and test completely separate?

## Common Red Flags

⚠️ Any operation on `df` (the full dataframe) before splitting
⚠️ Using `.mean()`, `.std()`, `.median()` on the full dataset
⚠️ Creating features using `groupby()` on the full dataset
⚠️ Fitting scalers/imputers before splitting
⚠️ Target encoding before splitting
⚠️ Features that directly contain the answer
⚠️ "Too good to be true" accuracy (>95% on realistic problems)

## How to Fix (General Approach)

1. **Split first!** Always split your data before any preprocessing
2. **Fit only on training data** - use `.fit()` on X_train, then `.transform()` on X_test
3. **Use Pipeline** - sklearn's Pipeline ensures correct ordering
4. **Remove leaky features** - drop any features that wouldn't exist in production
5. **Be careful with feature engineering** - compute statistics only from training data

## Expected Performance After Fixes

After fixing all leakage issues, you should see:
- Test accuracy: approximately 70-85% (much more realistic!)
- Training accuracy slightly higher than test (expected)
- Cross-validation scores similar to test accuracy
- More sensible feature importances

Good luck! 🍀
