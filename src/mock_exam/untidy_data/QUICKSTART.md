# Quick Start Guide

## Getting Started with the Untidy Data Exam

### Step 1: Generate the Datasets

First, run the dataset generator:

```bash
cd /home/edward/github/data_prep_and_feature_engineering/src/mock_exam/untidy_data
python generate_datasets.py
```

This will create a `data/` folder with:
- `sales_wide.csv`
- `customer_data.csv`
- `students_courses.json`
- `orders_products.csv`

### Step 2: Open the Exam Notebook

Open `untidy_data_exam.ipynb` in Jupyter or VS Code.

### Step 3: Work Through Each Task

For each of the 4 tasks:
1. Load and examine the data
2. Identify what type of untidy data it is
3. Write transformation code
4. Run the verification cell

### Step 4: Use Resources Wisely

- **First attempt:** Try on your own
- **If stuck:** Check `hints.md` for guidance
- **Need visual help:** Review `VISUAL_GUIDE.md`
- **After completion:** Compare with `solution.py`
- **Deep dive:** Read `ANSWER_KEY.md` for explanations

---

## File Structure

```
untidy_data/
├── README.md                    # Overview
├── QUICKSTART.md               # This file
├── VISUAL_GUIDE.md             # Visual examples
├── untidy_data_exam.ipynb      # Main exam (START HERE!)
├── hints.md                    # Hints for each task
├── solution.py                 # Complete solutions
├── ANSWER_KEY.md               # Detailed explanations
├── generate_datasets.py        # Dataset generator
└── data/                       # Generated datasets
    ├── sales_wide.csv
    ├── customer_data.csv
    ├── students_courses.json
    └── orders_products.csv
```

---

## Tips for Success

### 1. Read Carefully
- Each task has specific requirements
- Pay attention to the expected column names

### 2. Check Your Work
- Verify data shape (number of rows and columns)
- Make sure no data is lost
- Ensure column names match expectations

### 3. Use the Hints Strategically
- Try for 5-10 minutes before checking hints
- Start with the general tips
- Move to specific syntax if needed

### 4. Learn from Verification Errors
- If a check fails, read the error message
- It tells you exactly what's wrong
- Fix and try again

---

## Common Pitfalls to Avoid

### Task 1 (Wide Format)
- ❌ Forgetting to split year and quarter after melting
- ❌ Not converting year to integer type
- ✅ Use `pd.melt()` then `str.split()`

### Task 2 (Multiple Variables)
- ❌ Not using `expand=True` in `str.split()`
- ❌ Keeping the original combined column
- ✅ Split, then drop the original column

### Task 3 (Nested JSON)
- ❌ Not using `record_path` parameter
- ❌ Forgetting to include `meta` fields
- ✅ Use `pd.json_normalize()` with both parameters

### Task 4 (Multiple Units)
- ❌ Not removing duplicates in orders/products tables
- ❌ Including too many columns in each table
- ✅ Keep each table focused on one entity type

---

## Time Management

- **Task 1:** 10-12 minutes
- **Task 2:** 8-10 minutes
- **Task 3:** 15-18 minutes (most challenging!)
- **Task 4:** 12-15 minutes

**Total:** 45-60 minutes

---

## After the Exam

1. Run `solution.py` to see complete solutions with output
2. Compare your approach with the provided solutions
3. Read `ANSWER_KEY.md` for conceptual understanding
4. Practice with your own untidy datasets!

---

## Questions?

If you're completely stuck:
1. Check `hints.md` for your specific task
2. Look at `VISUAL_GUIDE.md` for visual examples
3. Review the tidy data principles in `README.md`

Remember: The goal is to learn, not just to complete the exam!

---

**Ready to start? Open `untidy_data_exam.ipynb` and begin!** 🚀

Good luck! 🍀
