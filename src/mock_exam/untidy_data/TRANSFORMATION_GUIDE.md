# Quick Reference: Untidy Data → Python Syntax

## 🎯 Decision Tree: Which Transformation to Use?

### 1️⃣ Look at Your Column Headers

#### **Are column names actually DATA VALUES?** (e.g., `2023`, `Q1`, `Product_A`)
→ **WIDE FORMAT** → Use `pd.melt()` or `pd.pivot_table()`

```python
# Wide → Long (MOST COMMON)
pd.melt(df, id_vars=['id'], var_name='variable', value_name='value')

# Long → Wide (less common)
pd.pivot_table(df, values='value', index='id', columns='variable')
```

---

### 2️⃣ Look at Individual Cells

#### **Does ONE cell contain MULTIPLE values?** (e.g., `"New York, USA"`, `"2023-Q1"`)
→ **MULTIPLE VARIABLES IN ONE COLUMN** → Use `str.split()`

```python
# Split by delimiter
df[['col1', 'col2']] = df['combined'].str.split(',', expand=True)

# Extract with regex
df['year'] = df['period'].str.extract(r'(\d{4})')
```

---

### 3️⃣ Look at Data Structure

#### **Are there NESTED ARRAYS/LISTS/OBJECTS?** (JSON, nested dicts)
→ **NESTED DATA** → Use `json_normalize()` or `explode()`

```python
# JSON with nested arrays
pd.json_normalize(
    data['parent'],
    record_path='array_field',  # Array to explode
    meta=['parent_id', 'name']   # Parent fields to keep
)

# DataFrame with list columns
df.explode('list_column')
```

---

### 4️⃣ Look at What Each Row Represents

#### **Does one row mix DIFFERENT ENTITY TYPES?** (orders + products + customers)
→ **MULTIPLE OBSERVATIONAL UNITS** → Split into separate tables

```python
# Extract each entity type
orders = df[['order_id', 'date', 'customer']].drop_duplicates()
products = df[['product_id', 'name', 'price']].drop_duplicates()
items = df[['order_id', 'product_id', 'qty']]  # Relationship table
```

---

### 5️⃣ Look at Variables Across Rows

#### **Is ONE variable split ACROSS MULTIPLE ROWS?** (rare)
→ **VARIABLES IN ROWS** → Use `pivot()` or `unstack()`

```python
# Pivot specific rows into columns
pd.pivot(df, index='id', columns='variable_name', values='value')
```

---

## 📊 Complete Transformation Cheat Sheet

| **Scenario** | **Error Type** | **Shape Change** | **Primary Tool** | **Example** |
|--------------|----------------|------------------|------------------|-------------|
| Year columns: `2023`, `2024` | Variables in headers | Wide → Long | `pd.melt()` | Sales by year-quarter |
| Variable rows: `price`, `qty` | Variables in rows | Long → Wide | `pd.pivot()` | Pivot table reports |
| Cell: `"NYC,USA"` | Multiple vars in column | Same rows, more cols | `str.split()` | Parse location |
| Cell: `"2023-Q1"` | Multiple vars in column | Same rows, more cols | `str.extract()` | Parse period |
| JSON: `{courses: [...]}` | Nested structure | Fewer rows, exploded | `json_normalize()` | Student courses |
| List column: `[1,2,3]` | Nested structure | More rows | `explode()` | Tags, categories |
| Mixed: orders+products | Multiple units | Split into N tables | `drop_duplicates()` | Normalize database |
| Duplicated data | Redundancy | Fewer rows | `drop_duplicates()` | Clean dataset |
| Missing structure | No clear pattern | Custom logic | `groupby()` + `agg()` | Aggregate data |

---

## 🔄 Common Transformation Patterns

### Pattern 1: Wide → Long (Unpivot)
**When:** Columns are categories/time periods  
**Use:** `pd.melt()`

```python
# BEFORE: Wide
# product | 2023 | 2024
# Laptop  | 100  | 150

df_long = pd.melt(df, 
                  id_vars=['product'],      # Keep these
                  var_name='year',          # New column for old headers
                  value_name='sales')       # New column for values

# AFTER: Long
# product | year | sales
# Laptop  | 2023 | 100
# Laptop  | 2024 | 150
```

---

### Pattern 2: Long → Wide (Pivot)
**When:** Need summary/report format  
**Use:** `pd.pivot_table()`

```python
# BEFORE: Long
# product | year | sales
# Laptop  | 2023 | 100
# Laptop  | 2024 | 150

df_wide = pd.pivot_table(df,
                         values='sales',     # Values to fill
                         index='product',    # Row labels
                         columns='year')     # Column labels

# AFTER: Wide
# product | 2023 | 2024
# Laptop  | 100  | 150
```

---

### Pattern 3: Split Compound Values
**When:** One cell contains multiple pieces of info  
**Use:** `str.split()`, `str.extract()`

```python
# BEFORE: location = "New York,USA"
df[['city', 'country']] = df['location'].str.split(',', expand=True)

# BEFORE: period = "2023-Q1"
df[['year', 'quarter']] = df['period'].str.split('-', expand=True)

# With regex for complex patterns
df['year'] = df['period'].str.extract(r'(\d{4})')  # Extract 4 digits
df['code'] = df['id'].str.extract(r'([A-Z]+)')     # Extract letters
```

---

### Pattern 4: Explode Nested Arrays (JSON)
**When:** Each parent has multiple child records  
**Use:** `pd.json_normalize(record_path=...)`

```python
# BEFORE: JSON
# {
#   "student_id": "S001",
#   "courses": [{"code": "CS101"}, {"code": "CS102"}]
# }

df = pd.json_normalize(
    data['students'],           # Parent array
    record_path='courses',      # Child array to explode ← KEY!
    meta=['student_id', 'name'] # Parent fields to repeat
)

# AFTER: DataFrame
# student_id | code
# S001       | CS101
# S001       | CS102
```

---

### Pattern 5: Explode List Columns
**When:** DataFrame column contains lists  
**Use:** `explode()`

```python
# BEFORE:
# student_id | courses
# S001       | [CS101, CS102]

df = df.explode('courses')

# AFTER:
# student_id | courses
# S001       | CS101
# S001       | CS102
```

---

### Pattern 6: Split Mixed Entity Tables
**When:** One table contains multiple entity types  
**Use:** Column selection + `drop_duplicates()`

```python
# BEFORE: Everything mixed
# order_id | customer | product_id | product_name | qty
# 1001     | Alice    | P001       | Laptop       | 1

# AFTER: Separate tables
orders = df[['order_id', 'customer', 'date']].drop_duplicates()
products = df[['product_id', 'product_name', 'price']].drop_duplicates()
order_items = df[['order_id', 'product_id', 'qty']]  # Junction table
```

---

## 🚦 Quick Decision Guide

### START HERE: What's wrong with your data?

```
┌─────────────────────────────────────┐
│ Is it WIDE format?                  │
│ (Years/categories as columns)       │ → pd.melt()
└─────────────────────────────────────┘
           │ NO
           ↓
┌─────────────────────────────────────┐
│ One cell = multiple values?         │
│ ("NYC,USA" or "2023-Q1")           │ → str.split() / str.extract()
└─────────────────────────────────────┘
           │ NO
           ↓
┌─────────────────────────────────────┐
│ Nested arrays/objects?              │
│ (JSON, lists in cells)              │ → json_normalize() / explode()
└─────────────────────────────────────┘
           │ NO
           ↓
┌─────────────────────────────────────┐
│ Multiple entity types mixed?        │
│ (Orders + Products together)        │ → Split tables + drop_duplicates()
└─────────────────────────────────────┘
           │ NO
           ↓
┌─────────────────────────────────────┐
│ Need aggregation/summary?           │
│ (Pivot to wide for reporting)      │ → pivot_table() / groupby()
└─────────────────────────────────────┘
```

---

## 💡 Pro Tips

### When to use `melt()` vs `pivot()`
- **`melt()`**: Wide → Long (unpivot) - **Data preparation**
- **`pivot()`**: Long → Wide (pivot) - **Data presentation**

### When to use `json_normalize()` vs `explode()`
- **`json_normalize()`**: For JSON/dict with nested arrays → DataFrame
- **`explode()`**: For DataFrame columns that contain lists

### When to use `str.split()` vs `str.extract()`
- **`str.split()`**: Simple delimiter (`,` `/` `-`)
- **`str.extract()`**: Complex patterns (regex needed)

---

## ⚠️ Common Mistakes

| ❌ Wrong | ✅ Right | Why |
|----------|----------|-----|
| `str.split(',')` | `str.split(',', expand=True)` | Need `expand=True` for multiple columns |
| `json_normalize(data)` | `json_normalize(data, record_path='field')` | Missing `record_path` for arrays |
| `melt()` without `id_vars` | `melt(id_vars=['id'])` | Need to specify which columns to keep |
| `pivot()` with duplicates | `pivot_table()` with aggfunc | Use `pivot_table` when duplicates exist |

---

## 📝 Summary: 5 Main Transformations

| # | Transformation | Function | When to Use |
|---|----------------|----------|-------------|
| 1 | **Unpivot** (Wide→Long) | `pd.melt()` | Column names are data values |
| 2 | **Pivot** (Long→Wide) | `pd.pivot_table()` | Need cross-tabulation/summary |
| 3 | **Split** | `str.split(expand=True)` | Multiple vars in one column |
| 4 | **Explode** | `json_normalize()` / `explode()` | Nested arrays/lists |
| 5 | **Normalize** | `drop_duplicates()` | Multiple entity types mixed |

---

**🎯 Remember:** Tidy data = 1 variable per column, 1 observation per row, 1 entity type per table!
