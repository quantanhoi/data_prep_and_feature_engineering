"""
Generate a synthetic dataset for the 'model too bad' mock exam.
This creates housing_for_bad_model.csv
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Set random seed
np.random.seed(42)

print("Generating dataset for 'model too bad' exam...")

# Load California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Rename columns to use snake_case
df = df.rename(columns={
    'MedInc': 'median_income',
    'HouseAge': 'housing_median_age',
    'AveRooms': 'avg_rooms',
    'AveBedrms': 'avg_bedrooms',
    'Population': 'population',
    'AveOccup': 'avg_occupancy',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'MedHouseVal': 'median_house_value'
})

print(f"Original dataset shape: {df.shape}")

# Create binary target: expensive (1) if house value > 2.5, else 0
threshold = 2.5  # median is around 2.0
df['expensive'] = (df['median_house_value'] > threshold).astype(int)

print(f"Target distribution:\n{df['expensive'].value_counts()}")

# Add some categorical features
np.random.seed(42)
ocean_proximity_options = ['NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'INLAND', 'ISLAND']
df['ocean_proximity'] = np.random.choice(
    ocean_proximity_options, 
    size=len(df),
    p=[0.25, 0.20, 0.25, 0.29, 0.01]
)

# Add neighborhood type (correlated with house value)
neighborhood_options = ['Urban', 'Suburban', 'Rural']
# Make it somewhat correlated with target
neighborhood_probs = []
for val in df['expensive']:
    if val == 1:  # expensive houses more likely in Urban/Suburban
        neighborhood_probs.append([0.5, 0.4, 0.1])
    else:
        neighborhood_probs.append([0.2, 0.3, 0.5])

df['neighborhood'] = [np.random.choice(neighborhood_options, p=probs) 
                      for probs in neighborhood_probs]

# Introduce realistic missing values
missing_mask = np.random.random(len(df)) < 0.15  # 15% missing in avg_bedrooms
df.loc[missing_mask, 'avg_bedrooms'] = np.nan

missing_mask = np.random.random(len(df)) < 0.10  # 10% missing in median_income
df.loc[missing_mask, 'median_income'] = np.nan

missing_mask = np.random.random(len(df)) < 0.20  # 20% missing in ocean_proximity
df.loc[missing_mask, 'ocean_proximity'] = np.nan

# Add some outliers that will affect scaling
outlier_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[outlier_indices, 'population'] = df.loc[outlier_indices, 'population'] * 10

# Add a high-cardinality categorical (will cause issues if not handled)
df['block_id'] = [f"BLOCK_{i%5000}" for i in range(len(df))]

# Remove the actual target variable to avoid leakage
df = df.drop(columns=['median_house_value'])

# Reorder columns
cols = ['median_income', 'housing_median_age', 'avg_rooms', 'avg_bedrooms',
        'population', 'avg_occupancy', 'latitude', 'longitude', 
        'ocean_proximity', 'neighborhood', 'block_id', 'expensive']
df = df[cols]

# Save to CSV
output_path = "housing_for_bad_model.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\n⚠️ This dataset has several issues that will cause poor model performance!")
print("Issues include: missing values, outliers, high-cardinality categorical, class imbalance, etc.")
