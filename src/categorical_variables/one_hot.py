import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.read_parquet('okcupid_profiles.parquet')

# Create OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform body_type column
encoded_data = encoder.fit_transform(df[['body_type']])

# Bonus 1: Get feature names
feature_names = encoder.get_feature_names_out(['body_type'])
print(f"Feature names: {feature_names}")

# Bonus 2: Missing values are handled automatically as a separate category
print(f"Missing values in body_type: {df['body_type'].isna().sum()}")
print("Note: OneHotEncoder treats missing values as a separate category by default")

# Create DataFrame with encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
print(encoded_df.head())


"""
Problems with one hot encoding

- Incomplete vocabulary: Due to random sampling, some values just do not make it
into your training set.
- Model size due to cardinality: What if you're categorical variable has
millions of values, like e.g. a device id?
- Cold start: What to do with values that are not yet there during data
collection of your training data, but appear later during the serving period?
"""