"""
Generate a synthetic dataset with intentional data leakage for the mock exam.
This creates housing_with_target_leakage.csv
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Set random seed
np.random.seed(42)

print("Generating dataset with intentional leakage issues...")

# Load California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Rename the target for clarity
df = df.rename(columns={'MedHouseVal': 'median_house_value'})

print(f"Original dataset shape: {df.shape}")

# Create binary target: high_value (1) if house value > median, else 0
threshold = df['median_house_value'].median()
df['high_value'] = (df['median_house_value'] > threshold).astype(int)

print(f"Target distribution:\n{df['high_value'].value_counts()}")

# Add some categorical features
np.random.seed(42)
ocean_proximity_options = ['NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'INLAND', 'ISLAND']
df['ocean_proximity'] = np.random.choice(
    ocean_proximity_options, 
    size=len(df),
    p=[0.25, 0.20, 0.25, 0.29, 0.01]  # weighted probabilities
)

# Introduce some missing values (realistic scenario)
missing_mask = np.random.random(len(df)) < 0.05  # 5% missing
df.loc[missing_mask, 'total_bedrooms'] = np.nan

# Introduce a few more missing values in other columns
for col in ['median_income', 'housing_median_age']:
    missing_mask = np.random.random(len(df)) < 0.02  # 2% missing
    df.loc[missing_mask, col] = np.nan

# Reorder columns to make target less obvious
cols = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
        'population', 'households', 'ocean_proximity', 'median_house_value', 'high_value']
df = df[cols]

# Save to CSV
output_path = "housing_with_target_leakage.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print("\n⚠️ WARNING: This dataset contains 'median_house_value' which leaks target information!")
print("This is intentional for the mock exam.")
