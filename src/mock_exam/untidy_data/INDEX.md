# Untidy Data Mock Exam - Complete Package

## 📚 Table of Contents

### 🎯 Main Exam
- **[untidy_data_exam.ipynb](untidy_data_exam.ipynb)** - The main exam notebook (START HERE!)

### 📖 Learning Resources (Read BEFORE the exam)
1. **[README.md](README.md)** - Overview and learning objectives
2. **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step getting started guide
3. **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Visual examples of each untidy data type

### 💡 Help During Exam
- **[hints.md](hints.md)** - Task-specific hints (use when stuck)

### ✅ After Completion
1. **[solution.py](solution.py)** - Complete working solutions with output
2. **[ANSWER_KEY.md](ANSWER_KEY.md)** - Detailed explanations and grading rubric

### 🔧 Utilities
- **[generate_datasets.py](generate_datasets.py)** - Regenerate datasets if needed
- **[data/](data/)** - Folder containing all generated datasets

---

## 📋 Exam Structure

### Task 1: Sales Data (25 points)
- **File:** `data/sales_wide.csv`
- **Type:** Variables in column headers (wide format)
- **Key Function:** `pd.melt()`
- **Difficulty:** ⭐⭐

### Task 2: Customer Data (20 points)
- **File:** `data/customer_data.csv`
- **Type:** Multiple variables in one column
- **Key Function:** `str.split(expand=True)`
- **Difficulty:** ⭐

### Task 3: Student Courses (30 points)
- **File:** `data/students_courses.json`
- **Type:** Nested arrays in JSON
- **Key Function:** `pd.json_normalize(record_path=...)`
- **Difficulty:** ⭐⭐⭐⭐

### Task 4: Orders & Products (25 points)
- **File:** `data/orders_products.csv`
- **Type:** Multiple observational units
- **Key Function:** `drop_duplicates()`
- **Difficulty:** ⭐⭐⭐

---

## 🚀 Quick Start

```bash
# 1. Generate datasets (if not already done)
python generate_datasets.py

# 2. Open the exam notebook
# Open untidy_data_exam.ipynb in Jupyter or VS Code

# 3. Work through all 4 tasks

# 4. Run the solution script to see answers
python solution.py
```

---

## 🎓 Learning Path

### Before the Exam:
1. Read [README.md](README.md) for context
2. Review [VISUAL_GUIDE.md](VISUAL_GUIDE.md) for visual examples
3. Check [QUICKSTART.md](QUICKSTART.md) for setup

### During the Exam:
1. Work through [untidy_data_exam.ipynb](untidy_data_exam.ipynb)
2. Use [hints.md](hints.md) if stuck (try for 5-10 min first!)
3. Run verification cells after each task

### After the Exam:
1. Run [solution.py](solution.py) to see complete solutions
2. Compare your approach with the solutions
3. Read [ANSWER_KEY.md](ANSWER_KEY.md) for deep understanding
4. Review any sections where you struggled

---

## 📊 Dataset Overview

All datasets are automatically generated and contain realistic scenarios:

| Dataset | Format | Rows | Untidy Type | Key Challenge |
|---------|--------|------|-------------|---------------|
| sales_wide.csv | CSV | 4 | Wide format | Splitting year_quarter |
| customer_data.csv | CSV | 6 | Multiple vars | City,Country parsing |
| students_courses.json | JSON | 3 students | Nested arrays | record_path parameter |
| orders_products.csv | CSV | 8 | Multiple units | Creating 3 tables |

---

## ✨ Features

- ✅ **Complete exam package** with all materials
- ✅ **Automatic verification** for each task
- ✅ **Progressive hints** - try, then get help
- ✅ **Visual examples** to understand concepts
- ✅ **Working solutions** with explanations
- ✅ **JSON dataset with nested arrays** (includes transformation hints!)
- ✅ **4 different untidy data types** covered
- ✅ **Real-world scenarios** for each case

---

## 🎯 Learning Objectives

By completing this exam, you will be able to:

1. ✓ Identify the 4 main types of untidy data
2. ✓ Apply `pd.melt()` to convert wide to long format
3. ✓ Use `str.split()` to separate multiple variables
4. ✓ Work with `pd.json_normalize()` and the `record_path` parameter
5. ✓ Split tables into proper observational units
6. ✓ Verify tidy data principles in transformed datasets

---

## 🔑 Key Transformation Syntax

### Wide to Long (Task 1)
```python
df_long = pd.melt(df, id_vars=['id_cols'], var_name='new_col', value_name='value_col')
```

### Split Multiple Variables (Task 2)
```python
df[['col1', 'col2']] = df['combined'].str.split(',', expand=True)
```

### Explode Nested Arrays (Task 3) - ⚠️ IMPORTANT!
```python
df = pd.json_normalize(
    data['parent_array'],
    record_path='child_array',  # The array to explode
    meta=['parent_field1', 'parent_field2']  # Parent info to keep
)
```

### Separate Tables (Task 4)
```python
table1 = df[['cols_for_entity1']].drop_duplicates()
table2 = df[['cols_for_entity2']].drop_duplicates()
```

---

## 💯 Grading

- **90-100 points:** Excellent understanding
- **75-89 points:** Good understanding, minor issues
- **60-74 points:** Basic understanding, needs review
- **Below 60:** Review tidy data principles

---

## 📝 Notes

- Task 3 (JSON nested arrays) is intentionally the most challenging
- The `record_path` parameter is the key to unlocking nested structures
- Real-world data often has multiple untidy issues combined
- Practice makes perfect - try creating your own untidy datasets!

---

## 🆘 Getting Help

**Stuck on syntax?**
- Check [hints.md](hints.md) for the specific task

**Not sure what's wrong?**
- Review [VISUAL_GUIDE.md](VISUAL_GUIDE.md) for visual examples

**Want to understand deeply?**
- Read [ANSWER_KEY.md](ANSWER_KEY.md) for explanations

**Need to see working code?**
- Run [solution.py](solution.py) for complete solutions

---

## 🌟 Good Luck!

Remember: The goal isn't just to pass, but to understand **why** data needs to be tidy and **how** to transform it. These skills are essential for real-world data analysis!

**Time estimate:** 45-60 minutes

**Ready?** Open `untidy_data_exam.ipynb` and let's begin! 🚀
