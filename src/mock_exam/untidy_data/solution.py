"""
Complete Solutions for Untidy Data Mock Exam
"""
import pandas as pd
import json

print("=" * 80)
print("UNTIDY DATA EXAM - COMPLETE SOLUTIONS")
print("=" * 80)

# =============================================================================
# TASK 1: Sales Wide Data - Variables in Column Headers
# =============================================================================
print("\n" + "=" * 80)
print("TASK 1: Sales Data (Wide Format)")
print("=" * 80)

# Load the data
sales_wide = pd.read_csv('data/sales_wide.csv')
print("\n📋 ORIGINAL DATA (UNTIDY):")
print(sales_wide)
print(f"\nShape: {sales_wide.shape}")

print("\n🔍 UNTIDY DATA TYPE: Variables are stored in column headers")
print("   Problem: Year and quarter are in column names (2023_Q1, 2023_Q2, etc.)")
print("   instead of being separate variables")

# Solution
print("\n✅ SOLUTION:")
print("Step 1: Melt the dataframe to convert from wide to long format")

sales_long = pd.melt(
    sales_wide,
    id_vars=['product', 'category'],
    var_name='period',
    value_name='sales'
)

print("\nAfter melting:")
print(sales_long.head(10))

print("\nStep 2: Split the period column into year and quarter")
sales_long[['year', 'quarter']] = sales_long['period'].str.split('_', expand=True)

print("\nStep 3: Drop the original period column and clean up")
sales_tidy = sales_long.drop('period', axis=1)
sales_tidy['year'] = sales_tidy['year'].astype(int)

print("\n📊 FINAL TIDY DATA:")
print(sales_tidy)
print(f"\nShape: {sales_tidy.shape}")
print("\n✓ Each variable has its own column")
print("✓ Each observation (product-quarter-year combination) has its own row")

# Save
sales_tidy.to_csv('data/sales_tidy.csv', index=False)
print("\n💾 Saved to: data/sales_tidy.csv")

# =============================================================================
# TASK 2: Customer Data - Multiple Variables in One Column
# =============================================================================
print("\n\n" + "=" * 80)
print("TASK 2: Customer Data (Multiple Variables in One Column)")
print("=" * 80)

# Load the data
customer_data = pd.read_csv('data/customer_data.csv')
print("\n📋 ORIGINAL DATA (UNTIDY):")
print(customer_data)
print(f"\nShape: {customer_data.shape}")

print("\n🔍 UNTIDY DATA TYPE: Multiple variables stored in one column")
print("   Problem: 'location' column contains both city and country")

# Solution
print("\n✅ SOLUTION:")
print("Split the location column into city and country")

customer_data[['city', 'country']] = customer_data['location'].str.split(',', expand=True)
customer_tidy = customer_data.drop('location', axis=1)

print("\n📊 FINAL TIDY DATA:")
print(customer_tidy)
print(f"\nShape: {customer_tidy.shape}")
print("\n✓ Each variable (city and country) has its own column")

# Save
customer_tidy.to_csv('data/customer_tidy.csv', index=False)
print("\n💾 Saved to: data/customer_tidy.csv")

# =============================================================================
# TASK 3: Student Courses JSON - Nested Arrays
# =============================================================================
print("\n\n" + "=" * 80)
print("TASK 3: Student Courses (Nested JSON with Arrays)")
print("=" * 80)

# Load the JSON data
with open('data/students_courses.json', 'r') as f:
    students_data = json.load(f)

print("\n📋 ORIGINAL DATA (UNTIDY):")
print("JSON structure with nested arrays:")
print(json.dumps(students_data, indent=2)[:800] + "...")

print("\n🔍 UNTIDY DATA TYPE: Nested data structure with arrays")
print("   Problem: Each student has an array of courses")
print("   Each course should be a separate observation (row)")

# Solution
print("\n✅ SOLUTION:")
print("Use pd.json_normalize() with record_path to explode the nested array")

students_tidy = pd.json_normalize(
    students_data['students'],
    record_path='courses',
    meta=['student_id', 'name', 'major']
)

print("\n📊 FINAL TIDY DATA:")
print(students_tidy)
print(f"\nShape: {students_tidy.shape}")
print("\n✓ Each course enrollment is now a separate row")
print("✓ Student information is repeated for each of their courses")
print("✓ Each variable has its own column")

# Save
students_tidy.to_csv('data/students_courses_tidy.csv', index=False)
print("\n💾 Saved to: data/students_courses_tidy.csv")

print("\n📝 KEY SYNTAX FOR NESTED ARRAYS:")
print("   pd.json_normalize(")
print("       data['students'],              # The array to iterate")
print("       record_path='courses',         # The nested array to explode")
print("       meta=['student_id', 'name']    # Parent fields to include")
print("   )")

# =============================================================================
# TASK 4: Orders and Products - Multiple Observational Units
# =============================================================================
print("\n\n" + "=" * 80)
print("TASK 4: Orders and Products (Multiple Observational Units)")
print("=" * 80)

# Load the data
orders_products = pd.read_csv('data/orders_products.csv')
print("\n📋 ORIGINAL DATA (UNTIDY):")
print(orders_products)
print(f"\nShape: {orders_products.shape}")

print("\n🔍 UNTIDY DATA TYPE: Multiple observational units in the same table")
print("   Problem: Table contains information about:")
print("   - Orders (order_id, order_date, customer, address)")
print("   - Products (product_id, name, category, price)")
print("   - Order Items (the relationship between orders and products)")

# Solution
print("\n✅ SOLUTION:")
print("Separate into three tables based on observational units")

# Table 1: Orders
print("\nStep 1: Extract unique ORDERS")
orders = orders_products[['order_id', 'order_date', 'customer_name', 'shipping_address']].drop_duplicates()
print("\n📊 ORDERS TABLE:")
print(orders)
print(f"Shape: {orders.shape}")

# Table 2: Products
print("\nStep 2: Extract unique PRODUCTS")
products = orders_products[['product_id', 'product_name', 'product_category', 'unit_price']].drop_duplicates()
print("\n📊 PRODUCTS TABLE:")
print(products)
print(f"Shape: {products.shape}")

# Table 3: Order Items (the many-to-many relationship)
print("\nStep 3: Extract ORDER ITEMS (relationships)")
order_items = orders_products[['order_id', 'product_id', 'quantity']].copy()
print("\n📊 ORDER_ITEMS TABLE:")
print(order_items)
print(f"Shape: {order_items.shape}")

print("\n✓ Each type of observational unit now has its own table")
print("✓ Tables can be joined using order_id and product_id as keys")
print("✓ No redundant information (customer name not repeated for each item)")

# Save all three tables
orders.to_csv('data/orders.csv', index=False)
products.to_csv('data/products.csv', index=False)
order_items.to_csv('data/order_items.csv', index=False)

print("\n💾 Saved to:")
print("   - data/orders.csv")
print("   - data/products.csv")
print("   - data/order_items.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "=" * 80)
print("EXAM SUMMARY - UNTIDY DATA TYPES COVERED")
print("=" * 80)

print("""
1. ✅ Variables in Column Headers (Wide Format)
   - Solution: pd.melt() to convert to long format
   
2. ✅ Multiple Variables in One Column
   - Solution: str.split() with expand=True
   
3. ✅ Nested Data Structures (Arrays in JSON)
   - Solution: pd.json_normalize() with record_path parameter
   
4. ✅ Multiple Observational Units in Same Table
   - Solution: Split into separate tables using drop_duplicates()

🎉 All tasks completed! Data is now tidy and ready for analysis.
""")

print("=" * 80)
