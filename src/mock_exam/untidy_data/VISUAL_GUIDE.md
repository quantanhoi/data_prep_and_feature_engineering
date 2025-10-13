# Visual Guide to Untidy Data Types

This guide provides visual examples of each untidy data type you'll encounter in the exam.

---

## 🔴 Type 1: Variables in Column Headers (Wide Format)

### ❌ UNTIDY:
```
product    | category    | 2023_Q1 | 2023_Q2 | 2024_Q1 | 2024_Q2
-----------|-------------|---------|---------|---------|--------
Laptop     | Electronics | 150     | 175     | 200     | 220
Mouse      | Accessories | 450     | 520     | 580     | 630
Keyboard   | Accessories | 320     | 340     | 380     | 410
```

**Problem:** Years and quarters are values, not variable names!

### ✅ TIDY:
```
product  | category    | year | quarter | sales
---------|-------------|------|---------|------
Laptop   | Electronics | 2023 | Q1      | 150
Laptop   | Electronics | 2023 | Q2      | 175
Laptop   | Electronics | 2024 | Q1      | 200
Laptop   | Electronics | 2024 | Q2      | 220
Mouse    | Accessories | 2023 | Q1      | 450
Mouse    | Accessories | 2023 | Q2      | 520
...
```

**Transformation:** `pd.melt()` + `str.split()`

---

## 🟠 Type 2: Multiple Variables in One Column

### ❌ UNTIDY:
```
customer_id | name          | location        | total_purchases
------------|---------------|-----------------|----------------
101         | Alice Johnson | New York,USA    | 1200
102         | Bob Smith     | London,UK       | 850
103         | Carol White   | Paris,France    | 2100
```

**Problem:** `location` contains TWO variables (city AND country)!

### ✅ TIDY:
```
customer_id | name          | city     | country | total_purchases
------------|---------------|----------|---------|----------------
101         | Alice Johnson | New York | USA     | 1200
102         | Bob Smith     | London   | UK      | 850
103         | Carol White   | Paris    | France  | 2100
```

**Transformation:** `str.split(',', expand=True)`

---

## 🟡 Type 3: Nested Arrays in JSON

### ❌ UNTIDY:
```json
{
  "students": [
    {
      "student_id": "S001",
      "name": "John Doe",
      "major": "Computer Science",
      "courses": [
        {"course_code": "CS101", "course_name": "Intro to Programming", "grade": "A"},
        {"course_code": "CS102", "course_name": "Data Structures", "grade": "B+"}
      ]
    }
  ]
}
```

**Problem:** Each student has an ARRAY of courses - each course should be its own row!

### ✅ TIDY:
```
student_id | name     | major              | course_code | course_name              | grade
-----------|----------|--------------------|-------------|--------------------------|------
S001       | John Doe | Computer Science   | CS101       | Intro to Programming     | A
S001       | John Doe | Computer Science   | CS102       | Data Structures          | B+
S002       | Jane     | Mathematics        | MATH201     | Calculus II              | A
...
```

**Transformation:** 
```python
pd.json_normalize(
    data['students'],
    record_path='courses',  # ← The array to explode!
    meta=['student_id', 'name', 'major']  # ← Parent info to keep
)
```

---

## 🟢 Type 4: Multiple Observational Units in One Table

### ❌ UNTIDY:
```
order_id | order_date | customer | address     | product_id | product_name | category    | quantity | price
---------|------------|----------|-------------|------------|--------------|-------------|----------|-------
1001     | 2024-01-15 | Alice    | 123 Main St | P001       | Laptop       | Electronics | 1        | 999.99
1001     | 2024-01-15 | Alice    | 123 Main St | P002       | Mouse        | Accessories | 2        | 25.50
1002     | 2024-01-16 | Bob      | 456 Oak Ave | P001       | Laptop       | Electronics | 1        | 999.99
```

**Problem:** Contains THREE different entity types mixed together:
- 📦 Order info (id, date, customer, address)
- 🛍️ Product info (id, name, category, price)
- 🔗 Relationship (order ↔ product with quantity)

Customer "Alice" and address "123 Main St" are REPEATED unnecessarily!
Product "Laptop" details are REPEATED for every order!

### ✅ TIDY (3 separate tables):

**Table 1: ORDERS**
```
order_id | order_date | customer_name | shipping_address
---------|------------|---------------|------------------
1001     | 2024-01-15 | Alice         | 123 Main St
1002     | 2024-01-16 | Bob           | 456 Oak Ave
```

**Table 2: PRODUCTS**
```
product_id | product_name | product_category | unit_price
-----------|--------------|------------------|------------
P001       | Laptop       | Electronics      | 999.99
P002       | Mouse        | Accessories      | 25.50
```

**Table 3: ORDER_ITEMS** (the relationship)
```
order_id | product_id | quantity
---------|------------|----------
1001     | P001       | 1
1001     | P002       | 2
1002     | P001       | 1
```

**Transformation:** `drop_duplicates(subset=[...])` on different column combinations

---

## 🎯 Quick Reference Table

| Untidy Type | Problem | Main Tool | Secondary Tool |
|-------------|---------|-----------|----------------|
| Variables in Headers | Year/quarter are column names | `pd.melt()` | `str.split()` |
| Multiple Vars in Column | City,Country in one field | `str.split(expand=True)` | - |
| Nested Arrays | Courses array in JSON | `pd.json_normalize(record_path=...)` | - |
| Multiple Units | Orders + Products mixed | `drop_duplicates()` | Column selection |

---

## 💡 How to Identify Each Type

### Ask yourself:

1. **Are my column names actually data values?** → Type 1 (Wide format)
2. **Does one column contain multiple pieces of info?** → Type 2 (Multiple variables)
3. **Do I have arrays/lists nested in my data?** → Type 3 (Nested structures)
4. **Am I storing different types of entities together?** → Type 4 (Multiple units)

---

## 🚀 Transformation Strategy

1. **Load and examine** the data
2. **Identify** which tidy data principle is violated
3. **Choose** the appropriate transformation
4. **Apply** the transformation
5. **Verify** the result matches tidy data principles

Good luck! 🍀
