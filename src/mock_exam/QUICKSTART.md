# Quick Start Guide

## 🚀 Getting Started with the Mock Exam

### Option 1: Jupyter Notebook (Recommended for Beginners)

1. **Navigate to the mock_exam directory:**
   ```bash
   cd src/mock_exam
   ```

2. **Generate the dataset:**
   ```bash
   python generate_dataset.py
   ```

3. **Open the interactive notebook:**
   ```bash
   jupyter notebook data_leakage_exam.ipynb
   ```

4. **Follow the instructions in the notebook**
   - Run each cell
   - Observe the suspiciously high accuracy
   - Find and fix the bugs
   - Compare with the solution

---

### Option 2: Python Scripts (For Advanced Users)

1. **Generate the dataset:**
   ```bash
   python generate_dataset.py
   ```

2. **Run the leaky model:**
   ```bash
   python leaky_model.py
   ```
   
   You should see ~99.8% accuracy (too good to be true!)

3. **Find the bugs:**
   - Read through `leaky_model.py`
   - Use `hints.md` if stuck
   - Check `ANSWER_KEY.md` for detailed explanations

4. **Compare with the correct solution:**
   ```bash
   python solution.py
   ```
   
   You should see ~75-80% accuracy (realistic!)

---

## 📚 File Overview

| File | Purpose |
|------|---------|
| `README.md` | Exam overview and instructions |
| `generate_dataset.py` | Creates the housing dataset with intentional leakage |
| `leaky_model.py` | **THE BUGGY CODE** - Find the data leakage issues! |
| `solution.py` | The corrected version (don't peek!) |
| `data_leakage_exam.ipynb` | Interactive Jupyter notebook version |
| `ANSWER_KEY.md` | Detailed explanations of all 7 bugs |
| `hints.md` | Hints if you get stuck |
| `QUICKSTART.md` | This file |

---

## 🎯 Learning Objectives

After completing this exam, you should be able to:

1. ✅ Identify common data leakage patterns
2. ✅ Understand when to split data (BEFORE preprocessing!)
3. ✅ Use sklearn Pipeline correctly
4. ✅ Recognize features that leak target information
5. ✅ Implement proper cross-validation
6. ✅ Distinguish realistic vs. unrealistic model performance
7. ✅ Fix data leakage issues systematically

---

## 🐛 The 7 Bugs You Need to Find

Without giving away the answers, you should look for issues related to:

1. Feature engineering using dataset-wide statistics
2. Target encoding before train/test split
3. Missing value imputation timing
4. Feature scaling timing
5. Features that directly reveal the target
6. Categorical encoding inconsistencies
7. Cross-validation on preprocessed data

---

## ✅ Success Criteria

### Minimum (Pass)
- Find at least 3 major bugs
- Reduce accuracy from ~99% to ~75-85%
- Explain basic concepts of data leakage

### Target (Good)
- Find 5-6 bugs
- Fix most issues properly
- Use Pipeline for some transformations

### Excellent
- Find all 7 bugs
- Implement complete Pipeline solution
- Explain each issue thoroughly
- Achieve realistic performance metrics

---

## 💡 Tips

1. **Don't rush** - Read the code carefully
2. **Ask questions** - For each `.fit()`, ask "what data is it seeing?"
3. **Be systematic** - Go through the code step by step
4. **Use hints** - They're there to help you learn
5. **Compare results** - Check before/after accuracy
6. **Read error messages** - They often contain clues

---

## 🆘 If You're Stuck

1. **Start with hints.md** - Progressive hints that don't give everything away
2. **Check one section at a time** - Don't try to find all bugs at once
3. **Run the code** - See what happens, observe the outputs
4. **Google is your friend** - Look up "sklearn pipeline data leakage"
5. **Ask for help** - This is a learning exercise!

---

## 📊 Expected Timeline

- **Setup:** 2-3 minutes
- **Understanding the problem:** 5-10 minutes
- **Finding bugs:** 15-20 minutes
- **Implementing fixes:** 15-20 minutes
- **Verification:** 5 minutes

**Total: ~45 minutes**

---

## 🎓 After Completing

1. Compare your solution with `solution.py`
2. Read `ANSWER_KEY.md` thoroughly
3. Make notes of key lessons learned
4. Try to explain each bug to someone else
5. Think about how to avoid these issues in real projects

---

## 🔗 Related Concepts from Your Course

This exam integrates concepts from:

- **Categorical Variables:** One-hot encoding, target encoding
- **Scaling Numerical:** StandardScaler, normalization
- **Missing Data:** SimpleImputer
- **Train/Test Split:** When and how to split
- **Feature Engineering:** Creating derived features
- **Feature Selection:** Removing irrelevant/leaky features
- **Cross-Validation:** Proper CV methodology

---

## 📝 Questions to Reflect On

After completing the exam, consider:

1. Why did the leaky model achieve 99.8% accuracy?
2. Which bug had the biggest impact on performance?
3. How would you detect data leakage in a real project?
4. What would happen if you deployed the leaky model to production?
5. How can sklearn Pipeline prevent these issues?

---

## 🚀 Ready to Start?

1. ✅ Choose notebook or script approach
2. ✅ Generate the dataset
3. ✅ Set a 45-minute timer
4. ✅ Start finding bugs!

**Good luck! 🍀**

Remember: If your model is too good to be true, it probably is!
