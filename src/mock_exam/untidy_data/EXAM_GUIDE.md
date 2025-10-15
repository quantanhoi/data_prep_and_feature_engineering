# Untidy Data Exam - Complete Guide

## 📋 Overview
This exam tests your ability to identify and fix untidy data issues across 4 different scenarios. You'll work with CSV and JSON datasets that violate tidy data principles.

**Time Estimate:** 45-60 minutes

---

## 🚀 Quick Start

### 1. Generate Datasets
```bash
python generate_datasets.py
```

This creates the `data/` folder with all required datasets.

### 2. Open the Exam
Open `untidy_data_exam.ipynb` and work through all 4 tasks.

### 3. Check Your Work
- Each task has a verification cell
- Use `hints.md` if stuck (try 5-10 min first!)
- Run `solution.py` to see complete solutions after finishing

---

## 📊 Exam Tasks

### Task 1: Sales Data (25 points) ⭐⭐
- **File:** `data/sales_wide.csv`
- **Problem:** Variables stored in column headers (wide format)
- **Key Function:** `pd.melt()`
- **Goal:** Convert from wide to long format, then split year and quarter

### Task 2: Customer Data (20 points) ⭐
- **File:** `data/customer_data.csv`
- **Problem:** Multiple variables in one column
- **Key Function:** `str.split(expand=True)`
- **Goal:** Separate city and country from combined location column

### Task 3: Student Courses (30 points) ⭐⭐⭐⭐
- **File:** `data/students_courses.json`
- **Problem:** Nested arrays in JSON structure
- **Key Function:** `pd.json_normalize(record_path=...)`
- **Goal:** Explode course arrays so each course is a separate row

### Task 4: Orders & Products (25 points) ⭐⭐⭐
- **File:** `data/orders_products.csv`
- **Problem:** Multiple observational units in same table
- **Key Function:** `drop_duplicates()`
- **Goal:** Split into 3 separate tables (orders, products, order_items)

---

## 💡 Key Concepts & Solutions

### Type 1: Variables in Column Headers

**Before (Untidy):**
```
product  | 2023_Q1 | 2023_Q2 | 2024_Q1
---------|---------|---------|--------
Laptop   | 150     | 175     | 200
```

**After (Tidy):**
```
product  | year | quarter | sales
---------|------|---------|------
Laptop   | 2023 | Q1      | 150
Laptop   | 2023 | Q2      | 175
```

**Solution:**
```python
df_long = pd.melt(df, id_vars=['product', 'category'], 
                  var_name='period', value_name='sales')
df_long[['year', 'quarter']] = df_long['period'].str.split('_', expand=True)
```

---

### Type 2: Multiple Variables in One Column

**Before (Untidy):**
```
customer_id | location
------------|------------------
101         | New York,USA
```

**After (Tidy):**
```
customer_id | city     | country
------------|----------|--------
101         | New York | USA
```

**Solution:**
```python
df[['city', 'country']] = df['location'].str.split(',', expand=True)
df = df.drop('location', axis=1)
```

---

### Type 3: Nested Arrays in JSON

**Before (Untidy):**
```json
{
  "student_id": "S001",
  "name": "John Doe",
  "courses": [
    {"course_code": "CS101", "grade": "A"},
    {"course_code": "CS102", "grade": "B+"}
  ]
}
```

**After (Tidy):**
```
student_id | name     | course_code | grade
-----------|----------|-------------|------
S001       | John Doe | CS101       | A
S001       | John Doe | CS102       | B+
```

**Solution:**
```python
df = pd.json_normalize(
    data['students'],
    record_path='courses',  # Array to explode
    meta=['student_id', 'name', 'major']  # Parent info to keep
)
```

---

### Type 4: Multiple Observational Units

**Before (Untidy):**
```
order_id | customer_name | product_id | product_name | quantity
---------|---------------|------------|--------------|----------
O001     | Alice         | P101       | Laptop       | 2
O001     | Alice         | P102       | Mouse        | 1
```

**After (Tidy - 3 Tables):**

**Orders Table:**
```
order_id | order_date | customer_name | shipping_address
```

**Products Table:**
```
product_id | product_name | product_category | unit_price
```

**Order Items Table:**
```
order_id | product_id | quantity
```

**Solution:**
```python
orders = df[['order_id', 'order_date', 'customer_name', 'shipping_address']].drop_duplicates()
products = df[['product_id', 'product_name', 'product_category', 'unit_price']].drop_duplicates()
order_items = df[['order_id', 'product_id', 'quantity']]
```

---

## 🎯 Task-Specific Hints

### Task 1 Hints
- Column headers like `2023_Q1` contain data values (year and quarter)
- Use `pd.melt()` to pivot from wide to long
- After melting, split the period column: `str.split('_', expand=True)`
- Convert year to integer type
- Final columns: product, category, year, quarter, sales

### Task 2 Hints
- The `location` column contains TWO variables separated by comma
- Use `str.split(',', expand=True)` to create two columns
- Name the new columns: city and country
- Drop the original location column after splitting

### Task 3 Hints (Most Challenging!)
- Each student has an array of courses - each course needs its own row
- **Critical:** Use `record_path='courses'` to explode the array
- Use `meta=['student_id', 'name', 'major']` to include student info in each row
- Don't forget to access `data['students']` first
- Common mistake: forgetting the `record_path` parameter

**Complete syntax:**
```python
pd.json_normalize(
    data['students'],
    record_path='courses',
    meta=['student_id', 'name', 'major']
)
```

### Task 4 Hints
- This table mixes THREE entities: orders, products, and order_items
- Create 3 separate tables, each focused on one entity
- Use `drop_duplicates()` for orders and products tables
- Order items table keeps all rows (it's the relationship table)
- Check which columns belong to which entity

---

## ⚠️ Common Pitfalls

### Task 1
- ❌ Forgetting to split year and quarter after melting
- ❌ Not converting year to integer
- ✅ Use `pd.melt()` then `str.split()`

### Task 2
- ❌ Not using `expand=True` in split
- ❌ Keeping the original combined column
- ✅ Split with expand=True, then drop original

### Task 3
- ❌ Not using `record_path` parameter
- ❌ Forgetting to include `meta` fields
- ❌ Not accessing the 'students' key first
- ✅ Use complete json_normalize syntax with both parameters

### Task 4
- ❌ Not removing duplicates from orders/products
- ❌ Including too many columns in each table
- ✅ Keep each table focused on one entity type

---

## 📚 Tidy Data Principles

**Remember these 3 rules:**

1. **Each variable must have its own column**
2. **Each observation must have its own row**
3. **Each type of observational unit must form a table**

If your data violates any of these, it's untidy!

---

## ⏱️ Time Management

- Task 1: 10-12 minutes
- Task 2: 8-10 minutes  
- Task 3: 15-18 minutes (most challenging!)
- Task 4: 12-15 minutes

**Total:** 45-60 minutes

---

## ✅ Verification Checklist

After each transformation:
- [ ] Check data shape (rows and columns)
- [ ] Verify no data was lost
- [ ] Ensure column names are descriptive
- [ ] Run the verification cell
- [ ] Fix any errors before moving on

---

## 📁 File Structure

```
untidy_data/
├── EXAM_GUIDE.md              # This file - start here!
├── TRANSFORMATION_GUIDE.md    # Reference for transformations
├── untidy_data_exam.ipynb     # Main exam notebook
├── solution.py                # Complete solutions
├── generate_datasets.py       # Dataset generator
└── data/                      # Generated datasets
    ├── sales_wide.csv
    ├── customer_data.csv
    ├── students_courses.json
    └── orders_products.csv
```

---

## 🏆 After Completion

1. Run `solution.py` to see complete solutions with output
2. Compare your approach with the provided solutions
3. Review any sections where you struggled
4. Practice with your own untidy datasets!

---

## 🎓 Learning Objectives

By completing this exam, you will:

- ✓ Identify the 4 main types of untidy data
- ✓ Apply `pd.melt()` to convert wide to long format
- ✓ Use `str.split()` to separate multiple variables
- ✓ Master `pd.json_normalize()` with `record_path`
- ✓ Split tables into proper observational units
- ✓ Verify tidy data principles in your results

---

**Ready to begin? Open `untidy_data_exam.ipynb` and start with Task 1!** 🚀

Good luck! 🍀
