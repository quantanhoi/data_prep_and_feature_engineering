# Answer Key - Untidy Data Types and Solutions

## The 4 Main Types of Untidy Data (Covered in This Exam)

### 1. Variables are Stored in Column Headers ✅

**What it is:** Column names contain data values rather than variable names.

**Example:** 
```
product    | 2023_Q1 | 2023_Q2 | 2024_Q1
-----------|---------|---------|--------
Laptop     | 150     | 175     | 200
Mouse      | 450     | 520     | 580
```

**Problem:** The years and quarters are values, not variable names.

**Solution:** Use `pd.melt()` to convert from wide to long format.

```python
df_long = pd.melt(df, 
                  id_vars=['product', 'category'],
                  var_name='period',
                  value_name='sales')
# Then split period into year and quarter
df_long[['year', 'quarter']] = df_long['period'].str.split('_', expand=True)
```

**Tidy Result:**
```
product  | category | year | quarter | sales
---------|----------|------|---------|------
Laptop   | Electronics | 2023 | Q1   | 150
Laptop   | Electronics | 2023 | Q2   | 175
```

---

### 2. Multiple Variables Stored in One Column ✅

**What it is:** A single column contains multiple pieces of information that should be separate variables.

**Example:**
```
customer_id | location
------------|------------------
101         | New York,USA
102         | London,UK
```

**Problem:** Location contains both city and country.

**Solution:** Use `str.split()` with `expand=True`.

```python
df[['city', 'country']] = df['location'].str.split(',', expand=True)
df = df.drop('location', axis=1)
```

**Tidy Result:**
```
customer_id | city     | country
------------|----------|--------
101         | New York | USA
102         | London   | UK
```

---

### 3. Nested Data Structures / Arrays ✅

**What it is:** Data contains nested arrays or objects, often in JSON format.

**Example:**
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

**Problem:** Each course should be a separate observation (row).

**Solution:** Use `pd.json_normalize()` with `record_path` parameter.

```python
df = pd.json_normalize(
    data['students'],
    record_path='courses',  # Array to explode
    meta=['student_id', 'name', 'major']  # Parent info to include
)
```

**Tidy Result:**
```
student_id | name     | major    | course_code | grade
-----------|----------|----------|-------------|------
S001       | John Doe | CS       | CS101       | A
S001       | John Doe | CS       | CS102       | B+
```

**Key Syntax to Remember:**
- `record_path`: The nested array/list to "explode"
- `meta`: Parent-level fields to repeat for each child record

---

### 4. Multiple Observational Units in the Same Table ✅

**What it is:** A table contains different types of entities that should be in separate tables.

**Example:**
```
order_id | customer_name | product_id | product_name | quantity
---------|---------------|------------|--------------|----------
1001     | Alice         | P001       | Laptop       | 1
1001     | Alice         | P002       | Mouse        | 2
1002     | Bob           | P001       | Laptop       | 1
```

**Problem:** Contains three different entities:
- Orders (order info, customer)
- Products (product info)
- Order Items (relationship)

**Solution:** Split into separate tables using `drop_duplicates()`.

```python
# Extract unique orders
orders = df[['order_id', 'customer_name', 'order_date']].drop_duplicates()

# Extract unique products
products = df[['product_id', 'product_name', 'category']].drop_duplicates()

# Keep order items (the relationship)
order_items = df[['order_id', 'product_id', 'quantity']]
```

**Tidy Result:**

**Orders Table:**
```
order_id | customer_name | order_date
---------|---------------|------------
1001     | Alice         | 2024-01-15
1002     | Bob           | 2024-01-16
```

**Products Table:**
```
product_id | product_name | category
-----------|--------------|-------------
P001       | Laptop       | Electronics
P002       | Mouse        | Accessories
```

**Order Items Table:**
```
order_id | product_id | quantity
---------|------------|----------
1001     | P001       | 1
1001     | P002       | 2
1002     | P001       | 1
```

---

## The Three Principles of Tidy Data

1. **Each variable must have its own column**
   - Don't mix multiple variables in one column
   - Don't put variables in column headers

2. **Each observation must have its own row**
   - One row = one complete observation
   - Don't spread observations across multiple rows

3. **Each type of observational unit must form a table**
   - Different entities belong in different tables
   - Use keys to link related tables

---

## Common Transformation Functions

| Untidy Data Type | Primary Solution | Key Function |
|------------------|------------------|--------------|
| Variables in headers | Wide to long | `pd.melt()` |
| Multiple vars in column | Split column | `str.split(expand=True)` |
| Nested arrays (JSON) | Explode arrays | `pd.json_normalize(record_path=...)` |
| Multiple units | Separate tables | `drop_duplicates(subset=...)` |
| Values in rows | Long to wide | `pd.pivot()` or `pd.pivot_table()` |

---

## Grading Rubric

### Task 1: Sales Wide Data (25 points)
- ✓ Correctly identified as "variables in column headers" (5 pts)
- ✓ Used `pd.melt()` appropriately (10 pts)
- ✓ Split year and quarter correctly (5 pts)
- ✓ Final tidy format with correct columns (5 pts)

### Task 2: Customer Data (20 points)
- ✓ Correctly identified as "multiple variables in one column" (5 pts)
- ✓ Used `str.split()` with expand=True (10 pts)
- ✓ Final tidy format with city and country columns (5 pts)

### Task 3: Student Courses JSON (30 points)
- ✓ Correctly identified as "nested data structure" (5 pts)
- ✓ Used `pd.json_normalize()` (10 pts)
- ✓ Correctly used `record_path` parameter (10 pts)
- ✓ Final tidy format with all variables (5 pts)

### Task 4: Orders and Products (25 points)
- ✓ Correctly identified as "multiple observational units" (5 pts)
- ✓ Created three separate tables (10 pts)
- ✓ Used `drop_duplicates()` appropriately (5 pts)
- ✓ No data loss or redundancy (5 pts)

**Total: 100 points**

---

## Additional Resources

- Hadley Wickham's "Tidy Data" paper: https://www.jstatsoft.org/article/view/v059i10
- Pandas documentation on reshaping: https://pandas.pydata.org/docs/user_guide/reshaping.html
- JSON normalization guide: https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html
