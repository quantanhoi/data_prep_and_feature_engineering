# Hints for Untidy Data Exam

## Task 1: Sales Data (sales_wide.csv)

**Type of Untidy Data:** Variables are stored in column headers

**Hints:**
- Column headers like `2023_Q1`, `2023_Q2` contain actual data values (year and quarter)
- Each year-quarter combination should be a separate row
- Use `pd.melt()` to convert from wide to long format
- You'll need to split the year and quarter after melting
- Final columns should be: product, category, year, quarter, sales

**Key Function:** `pd.melt()` with `id_vars` parameter

**Bonus Hint:** After melting, use `str.split()` to separate year and quarter

---

## Task 2: Customer Data (customer_data.csv)

**Type of Untidy Data:** Multiple variables stored in one column

**Hints:**
- The `location` column contains two variables: city and country
- They are separated by a comma
- Use `str.split()` with `expand=True` to create two columns
- Don't forget to name your new columns appropriately

**Key Function:** `df['column'].str.split(',', expand=True)`

**Bonus Hint:** You might want to remove the original `location` column after splitting

---

## Task 3: Student Courses (students_courses.json)

**Type of Untidy Data:** Nested data structure with arrays

**Hints:**
- Each student has multiple courses stored as an array
- Each course should be a separate row in the final tidy dataset
- Use `json_normalize()` to flatten the JSON
- The key parameter for handling arrays is `record_path`
- Use `meta` parameter to include student-level information in each course row

**Key Functions:** 
- `pd.json_normalize()` with `record_path='courses'`
- `meta=['student_id', 'name', 'major']` to preserve student info

**Critical Syntax:**
```python
pd.json_normalize(
    data['students'],
    record_path='courses',  # This is the array to explode
    meta=['student_id', 'name', 'major']  # These get repeated for each course
)
```

**Common Mistake:** Forgetting to access the 'students' key first

---

## Task 4: Orders and Products (orders_products.csv)

**Type of Untidy Data:** Multiple observational units in the same table

**Hints:**
- This table contains three different types of entities: orders, products, and order items
- Orders have: order_id, order_date, customer_name, shipping_address
- Products have: product_id, product_name, product_category, unit_price
- Order items (the relationship) have: order_id, product_id, quantity
- You need to create THREE separate tables
- Use `drop_duplicates()` to extract unique orders and products
- Be careful which columns to keep for each table

**Key Function:** `drop_duplicates(subset=[columns])`

**Table Structure:**
1. **orders**: order_id, order_date, customer_name, shipping_address
2. **products**: product_id, product_name, product_category, unit_price
3. **order_items**: order_id, product_id, quantity

---

## General Tips

1. Always check your data shape before and after transformation
2. Verify that you haven't lost any information
3. Check for duplicates where appropriate
4. Make sure column names are descriptive
5. Test your transformations on a small sample first

## Tidy Data Principles to Remember

1. **Each variable must have its own column**
2. **Each observation must have its own row**
3. **Each type of observational unit must form a table**

If your data violates any of these principles, it's untidy!
