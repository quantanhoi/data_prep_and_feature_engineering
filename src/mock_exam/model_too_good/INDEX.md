# Mock Exam: Data Leakage Detection - Complete Package

## 📦 What You've Got

Congratulations! You now have a complete mock exam package with **10 comprehensive files** covering data leakage detection and prevention.

---

## 📚 File Index

### 🎯 Start Here
1. **`README.md`** - Exam overview and learning objectives
2. **`QUICKSTART.md`** - Step-by-step guide to get started (⭐ START HERE)
3. **`VISUAL_GUIDE.md`** - Visual diagrams explaining data leakage

### 💻 Code Files
4. **`generate_dataset.py`** - Creates the housing dataset
5. **`leaky_model.py`** - The buggy code with 7 data leakage issues (YOUR TASK!)
6. **`solution.py`** - Correct implementation with detailed comments
7. **`data_leakage_exam.ipynb`** - Interactive Jupyter notebook version

### 📖 Support Materials
8. **`hints.md`** - Progressive hints (use when stuck)
9. **`ANSWER_KEY.md`** - Complete explanations of all 7 bugs
10. **`EXAM_SUMMARY.md`** - Instructor guide and exam overview

---

## 🚀 Quick Start (3 Steps)

### Step 1: Generate the Dataset
```bash
cd src/mock_exam
python generate_dataset.py
```

### Step 2: Choose Your Path

**Path A: Interactive Notebook (Recommended for Learning)**
```bash
jupyter notebook data_leakage_exam.ipynb
```

**Path B: Python Scripts (For Practice)**
```bash
python leaky_model.py  # See the bugs in action
# ... find and fix the bugs ...
python solution.py     # Compare with correct version
```

### Step 3: Learn and Verify
- Use `hints.md` if stuck
- Check `ANSWER_KEY.md` for explanations
- Read `VISUAL_GUIDE.md` for conceptual understanding

---

## 🎯 The Challenge

### The Scenario
You inherit a model with **99.8% accuracy**. Too good to be true? Yes!

### Your Mission
1. Find all **7 data leakage bugs**
2. Explain what each bug does wrong
3. Fix the code properly using sklearn Pipeline
4. Achieve realistic performance (~75-80% accuracy)

### Time Limit
**45 minutes**

---

## 🐛 The 7 Bugs (Summary)

| # | Bug Type | Impact | Difficulty |
|---|----------|--------|------------|
| 1 | Feature engineering on full dataset | High | Medium |
| 2 | Target encoding before split | Critical | Hard |
| 3 | Imputation before split | High | Easy |
| 4 | Scaling before split | High | Easy |
| 5 | Target leakage in features | Critical | Medium |
| 6 | Inconsistent categorical encoding | Medium | Medium |
| 7 | Meaningless cross-validation | Medium | Hard |

---

## 📊 Expected Results

### Before Fixes (Leaky Model)
```
Training Accuracy:  99.80%  ❌ Too good!
Test Accuracy:      99.80%  ❌ Too good!
CV Accuracy:        99.80%  ❌ Too good!

Top Features:
1. median_house_value     ← Directly leaks target!
2. price_vs_mean          ← Derived from target!
3. ocean_proximity_encoded ← Target encoded!
```

### After Fixes (Proper Model)
```
Training Accuracy:  82.34%  ✅ Realistic
Test Accuracy:      78.56%  ✅ Realistic
CV Accuracy:        77.91%  ✅ Realistic

Top Features:
1. median_income          ← Legitimate!
2. latitude               ← Legitimate!
3. housing_median_age     ← Legitimate!
```

---

## 🎓 Learning Objectives

After completing this exam, you will:

✅ Identify common data leakage patterns  
✅ Understand when to split data (BEFORE preprocessing!)  
✅ Use sklearn Pipeline correctly  
✅ Recognize features that leak target information  
✅ Implement proper cross-validation  
✅ Distinguish realistic vs. unrealistic performance  
✅ Debug ML pipelines systematically  

---

## 📖 How to Use Each File

### For Students

```
1. Read QUICKSTART.md              (5 min)
2. Read VISUAL_GUIDE.md            (10 min)
3. Run generate_dataset.py         (1 min)
4. Work through leaky_model.py     (30 min)
   - Use hints.md if stuck
5. Compare with solution.py        (5 min)
6. Read ANSWER_KEY.md              (15 min)
7. Reflect on what you learned     (5 min)

Total: ~70 minutes (including reflection)
```

### For Instructors

```
1. Read EXAM_SUMMARY.md            - Overview and grading
2. Review ANSWER_KEY.md            - Detailed solutions
3. Check solution.py               - Reference implementation
4. Modify leaky_model.py           - Customize difficulty
5. Use data_leakage_exam.ipynb     - In-class workshop
```

---

## 🎯 Success Criteria

### Minimum Pass (60%)
- Find 3-4 bugs
- Basic understanding of data leakage
- Some improvement in performance

### Target (75%)
- Find 5-6 bugs
- Good understanding of concepts
- Proper use of some Pipeline components

### Excellence (90%)
- Find all 7 bugs
- Deep understanding with clear explanations
- Complete Pipeline implementation
- Realistic performance achieved

---

## 💡 Key Concepts Covered

### Data Leakage Types
- **Temporal leakage:** Using future information
- **Train-test contamination:** Preprocessing before split
- **Target leakage:** Features derived from target
- **Overfitting to CV folds:** Improper cross-validation

### Related Course Topics
- Train/test split methodology
- Scaling numerical features
- Categorical variable encoding
- Missing data imputation
- Feature engineering
- sklearn Pipeline usage
- Cross-validation best practices

---

## 🔧 Technical Requirements

### Python Packages
```
pandas
numpy
scikit-learn
jupyter (for notebook version)
```

### Installation
```bash
pip install pandas numpy scikit-learn jupyter
# or
poetry install  # if using poetry in the workspace
```

---

## 🎨 File Descriptions

### Core Exam Files

**`leaky_model.py`** (170 lines)
- The main exam file with intentional bugs
- Simulates real-world data leakage mistakes
- Achieves 99.8% accuracy (too good to be true!)
- Contains 7 major data leakage issues

**`solution.py`** (250 lines)
- Properly implemented ML pipeline
- Uses sklearn Pipeline correctly
- Achieves realistic ~78% accuracy
- Includes detailed comments explaining each fix

**`generate_dataset.py`** (50 lines)
- Creates housing_with_target_leakage.csv
- Based on California Housing dataset
- Includes intentional leakage features
- 20,640 samples, 10 features

**`data_leakage_exam.ipynb`** (Jupyter Notebook)
- Interactive version of the exam
- Step-by-step cells for experimentation
- Fill-in-the-blank sections for students
- Great for in-class workshops

### Documentation Files

**`README.md`** - Main exam instructions  
**`QUICKSTART.md`** - Getting started guide  
**`VISUAL_GUIDE.md`** - Diagrams and visual explanations  
**`hints.md`** - Progressive hints without giving away answers  
**`ANSWER_KEY.md`** - Complete solutions with explanations  
**`EXAM_SUMMARY.md`** - Instructor guide and overview  

---

## 🌟 Best Practices Demonstrated

### In `solution.py`
1. ✅ Split data FIRST
2. ✅ Use sklearn Pipeline
3. ✅ Custom transformers for feature engineering
4. ✅ Proper cross-validation
5. ✅ Remove target leakage
6. ✅ Fit only on training data
7. ✅ Clear documentation and comments

### In `leaky_model.py` (What NOT to Do)
1. ❌ Preprocess before splitting
2. ❌ Use statistics from full dataset
3. ❌ Include target-derived features
4. ❌ Target encoding without CV
5. ❌ Fit transformers on all data
6. ❌ Meaningless cross-validation
7. ❌ Ignore realistic performance expectations

---

## 🎁 Bonus Materials

### Extension Activities
1. Create more leakage scenarios
2. Build automated leakage detection
3. Implement time-series leakage examples
4. Design production-ready pipelines
5. Write tests to prevent leakage

### Real-World Applications
- Kaggle competitions
- Production ML systems
- Research experiments
- ML interviews
- Business analytics

---

## 🆘 Getting Help

### If You're Stuck

1. **Read QUICKSTART.md** - Basic setup help
2. **Check hints.md** - Progressive hints
3. **Review VISUAL_GUIDE.md** - Conceptual diagrams
4. **Compare with solution.py** - See working code
5. **Read ANSWER_KEY.md** - Detailed explanations

### Common Issues

**"I can't find all the bugs"**
→ Check hints.md for clues

**"My fixed model still has high accuracy"**
→ You probably missed a major leakage source (check Bug #5)

**"I don't understand why X is wrong"**
→ Read the relevant section in VISUAL_GUIDE.md

**"Dataset generation fails"**
→ Make sure scikit-learn is installed: `pip install scikit-learn`

---

## 📈 What's Next?

After completing this exam:

1. **Practice:** Create your own leakage examples
2. **Review:** Study similar Kaggle competition failures
3. **Apply:** Use Pipeline in all future projects
4. **Share:** Teach someone else what you learned
5. **Explore:** Advanced topics like time-series CV

---

## 🎓 Credits

**Created for:** Data Preparation and Feature Engineering Course  
**Topic:** Data Leakage Detection and Prevention  
**Difficulty:** Intermediate  
**Time:** 45 minutes  
**Format:** Hands-on coding exercise  

Based on common patterns from:
- Kaggle competitions
- Production ML failures
- Real-world consulting experiences
- Academic research mistakes

---

## ✨ Final Words

> **"The best time to prevent data leakage is during preprocessing.  
> The second best time is when you notice 99% accuracy."**

Data leakage is one of the most common and costly mistakes in machine learning. This exam teaches you to recognize, understand, and prevent it.

**Key Takeaway:** If your model is too good to be true, it probably is. Always split first, preprocess second!

---

## 📞 Quick Reference

```bash
# Generate dataset
python generate_dataset.py

# Run buggy code (see the problem)
python leaky_model.py

# Run correct code (see the solution)
python solution.py

# Interactive learning
jupyter notebook data_leakage_exam.ipynb

# Get help
cat hints.md           # Progressive hints
cat ANSWER_KEY.md      # Full solutions
cat VISUAL_GUIDE.md    # Visual explanations
```

---

**Good luck! 🍀 Remember: Split first, preprocess second!**
