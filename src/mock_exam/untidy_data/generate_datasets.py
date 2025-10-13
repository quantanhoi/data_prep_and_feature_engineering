"""
Generate untidy datasets for the mock exam
"""
import pandas as pd
import json
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Dataset 1: Wide format - Variables in column headers (Year and Quarter mixed)
# Untidy: Column headers contain values (years) instead of variable names
sales_wide = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics'],
    '2023_Q1': [150, 450, 320, 89],
    '2023_Q2': [175, 520, 340, 95],
    '2024_Q1': [200, 580, 380, 110],
    '2024_Q2': [220, 630, 410, 125]
})
sales_wide.to_csv('data/sales_wide.csv', index=False)
print("✓ Generated sales_wide.csv")

# Dataset 2: Multiple variables in one column
# Untidy: location column contains both city and country separated by comma
customer_data = pd.DataFrame({
    'customer_id': [101, 102, 103, 104, 105, 106],
    'name': ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Eve Davis', 'Frank Miller'],
    'location': ['New York,USA', 'London,UK', 'Paris,France', 'Berlin,Germany', 'Tokyo,Japan', 'Sydney,Australia'],
    'total_purchases': [1200, 850, 2100, 1500, 950, 1750]
})
customer_data.to_csv('data/customer_data.csv', index=False)
print("✓ Generated customer_data.csv")

# Dataset 3: JSON with nested arrays - Student courses
# Untidy: courses is an array within each student record
students_data = {
    'students': [
        {
            'student_id': 'S001',
            'name': 'John Doe',
            'major': 'Computer Science',
            'courses': [
                {'course_code': 'CS101', 'course_name': 'Intro to Programming', 'grade': 'A'},
                {'course_code': 'CS102', 'course_name': 'Data Structures', 'grade': 'B+'},
                {'course_code': 'MATH201', 'course_name': 'Calculus II', 'grade': 'A-'}
            ]
        },
        {
            'student_id': 'S002',
            'name': 'Jane Smith',
            'major': 'Mathematics',
            'courses': [
                {'course_code': 'MATH201', 'course_name': 'Calculus II', 'grade': 'A'},
                {'course_code': 'MATH301', 'course_name': 'Linear Algebra', 'grade': 'A+'}
            ]
        },
        {
            'student_id': 'S003',
            'name': 'Mike Johnson',
            'major': 'Physics',
            'courses': [
                {'course_code': 'PHYS101', 'course_name': 'Mechanics', 'grade': 'B'},
                {'course_code': 'PHYS201', 'course_name': 'Electromagnetism', 'grade': 'B+'},
                {'course_code': 'MATH201', 'course_name': 'Calculus II', 'grade': 'B+'},
                {'course_code': 'CS101', 'course_name': 'Intro to Programming', 'grade': 'A'}
            ]
        }
    ]
}

with open('data/students_courses.json', 'w') as f:
    json.dump(students_data, f, indent=2)
print("✓ Generated students_courses.json")

# Dataset 4: Multiple observational units in one table
# Untidy: Contains both order information AND product information in same table
orders_products = pd.DataFrame({
    'order_id': [1001, 1001, 1002, 1002, 1003, 1004, 1004, 1004],
    'order_date': ['2024-01-15', '2024-01-15', '2024-01-16', '2024-01-16', 
                   '2024-01-17', '2024-01-18', '2024-01-18', '2024-01-18'],
    'customer_name': ['Alice', 'Alice', 'Bob', 'Bob', 'Carol', 'David', 'David', 'David'],
    'shipping_address': ['123 Main St', '123 Main St', '456 Oak Ave', '456 Oak Ave',
                         '789 Pine Rd', '321 Elm St', '321 Elm St', '321 Elm St'],
    'product_id': ['P001', 'P002', 'P001', 'P003', 'P002', 'P001', 'P002', 'P004'],
    'product_name': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse', 'Laptop', 'Mouse', 'Monitor'],
    'product_category': ['Electronics', 'Accessories', 'Electronics', 'Accessories',
                         'Accessories', 'Electronics', 'Accessories', 'Electronics'],
    'quantity': [1, 2, 1, 1, 3, 2, 1, 1],
    'unit_price': [999.99, 25.50, 999.99, 45.00, 25.50, 999.99, 25.50, 299.99]
})
orders_products.to_csv('data/orders_products.csv', index=False)
print("✓ Generated orders_products.csv")

print("\n✅ All datasets generated successfully!")
print("\nDatasets summary:")
print("1. sales_wide.csv - Variables in column headers (wide format)")
print("2. customer_data.csv - Multiple variables in one column")
print("3. students_courses.json - Nested arrays in JSON")
print("4. orders_products.csv - Multiple observational units in one table")
