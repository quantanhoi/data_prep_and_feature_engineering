# Scaling Issues Mock Exam

## Overview
This exam tests your ability to identify incorrect or missing scaling techniques and apply the correct transformations to improve model performance.

## Learning Objectives
- Identify when scaling is needed vs. not needed
- Recognize wrong scaling methods for specific algorithms
- Apply appropriate scaling techniques for different feature types
- Understand the impact of scaling on model performance
- Fix train/test split scaling issues (data leakage)

## Scenarios
You will encounter 4 different problems:

1. **Problem 1**: No scaling applied (Distance-based model fails)
2. **Problem 2**: Wrong scaling method used (Data distribution ignored)
3. **Problem 3**: Scaling applied to wrong algorithm (Tree-based model)
4. **Problem 4**: Data leakage in scaling (fit on all data before split)

## Files
- `scaling_exam.ipynb` - Main exam notebook with tasks
- `hints.md` - Hints for each task
- `solution.py` - Complete solution
- `ANSWER_KEY.md` - Detailed explanations
- `SCALING_GUIDE.md` - When to use which scaling method
- `data/` - Contains all generated datasets

## Time Estimate
45-60 minutes

## Getting Started
1. Open `scaling_exam.ipynb`
2. Run each problem scenario
3. Identify what's wrong with the scaling
4. Fix the scaling approach
5. Compare model performance before/after

## Key Concepts
- **StandardScaler (Z-score)**: For normally distributed data, distance-based models
- **MinMaxScaler**: For bounded data, neural networks
- **RobustScaler**: For data with outliers
- **No scaling**: For tree-based models (Random Forest, XGBoost)
- **Proper train/test scaling**: Fit on train, transform on test

Good luck! 🚀
