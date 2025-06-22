import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the parquet file
df = pd.read_parquet('okcupid_profiles.parquet')

# Check the location column
print(f"Total unique locations: {df['location'].nunique()}")
print(f"Total profiles: {len(df)}")
print(f"Sample locations: {df['location'].value_counts().head()}")

# Create train-test split BEFORE encoding (this is crucial!)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Identify unique locations in each set
train_locations = set(train_df['location'].dropna().unique())
test_locations = set(test_df['location'].dropna().unique())

# THE PROBLEM: Find locations in test but NOT in train
missing_locations = test_locations - train_locations
print(f"\n⚠️ PROBLEM IDENTIFIED:")
print(f"Locations in training set: {len(train_locations)}")
print(f"Locations in test set: {len(test_locations)}")
print(f"Locations in test but NOT in train: {len(missing_locations)}")
print(f"Examples of missing locations: {list(missing_locations)[:5]}")

# Show what happens with standard OneHotEncoder
print("\n❌ Standard OneHotEncoder will fail:")
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(train_df[['location']].dropna())

try:
    # This will fail for unseen locations
    test_encoded = encoder.transform(test_df[['location']].dropna())
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# SOLUTION 1: Use handle_unknown='ignore'
print("\n✅ SOLUTION 1: OneHotEncoder with handle_unknown='ignore'")
encoder_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_safe.fit(train_df[['location']].dropna())

# Now it works
train_encoded = encoder_safe.transform(train_df[['location']].dropna())
test_encoded = encoder_safe.transform(test_df[['location']].dropna())

print(f"Train encoded shape: {train_encoded.shape}")
print(f"Test encoded shape: {test_encoded.shape}")
print(f"Number of location features: {len(encoder_safe.get_feature_names_out())}")

# SOLUTION 2: Create 'other' category for rare locations (as mentioned in PDF)
print("\n✅ SOLUTION 2: Group rare locations as 'other'")
location_counts = df['location'].value_counts()
threshold = 5  # locations with less than 5 occurrences become 'other'
rare_locations = location_counts[location_counts < threshold].index

# Create new column with 'other' category
df['location_grouped'] = df['location'].copy()
df.loc[df['location'].isin(rare_locations), 'location_grouped'] = 'other'

# Now split and encode
train_grouped, test_grouped = train_test_split(df, test_size=0.2, random_state=42)
train_locs_grouped = set(train_grouped['location_grouped'].dropna().unique())
test_locs_grouped = set(test_grouped['location_grouped'].dropna().unique())
missing_grouped = test_locs_grouped - train_locs_grouped

print(f"After grouping rare locations:")
print(f"Missing locations reduced from {len(missing_locations)} to {len(missing_grouped)}")




# SOLUTION 3: Feature Hashing (the hashing trick)
print("\n✅ SOLUTION 3: Feature Hashing for handling high cardinality")

from sklearn.feature_extraction import FeatureHasher

# First, prepare the location data in the format FeatureHasher expects
# FeatureHasher expects input as list of dicts
def prepare_for_hasher(df, column):
    """Convert location column to list of dicts for FeatureHasher"""
    return [{f'location={loc}': 1} for loc in df[column].fillna('missing')]

# Set number of hash features (buckets)
# Rule of thumb: sqrt(n_unique_values) to 2*sqrt(n_unique_values)
n_features = 100  # Much smaller than actual unique locations!

# Create hasher - alternate_sign=True uses ±1 to reduce collision impact
hasher = FeatureHasher(n_features=n_features, input_type='dict', alternate_sign=True)

# Prepare data
train_hasher_input = prepare_for_hasher(train_df, 'location')
test_hasher_input = prepare_for_hasher(test_df, 'location')

# Transform to hashed features
train_hashed = hasher.transform(train_hasher_input).toarray()
test_hashed = hasher.transform(test_hasher_input).toarray()

print(f"Original unique locations: {df['location'].nunique()}")
print(f"Hashed feature dimensions: {n_features}")
print(f"Compression ratio: {df['location'].nunique() / n_features:.1f}x")
print(f"Train hashed shape: {train_hashed.shape}")
print(f"Test hashed shape: {test_hashed.shape}")

# Check for hash collisions
print("\n--- Analyzing Hash Collisions ---")
# Count how many locations hash to each bucket
location_to_hash = {}
for loc in df['location'].dropna().unique():
    # Hash the location to see which bucket it goes to
    hashed = hasher.transform([{f'location={loc}': 1}])
    # Find non-zero indices (where this location hashes to)
    indices = hashed.nonzero()[1]
    if len(indices) > 0:
        location_to_hash[loc] = indices[0]

# Count collisions
from collections import Counter
hash_counts = Counter(location_to_hash.values())
collisions = sum(1 for count in hash_counts.values() if count > 1)
max_collision = max(hash_counts.values())

print(f"Buckets with collisions: {collisions}")
print(f"Maximum locations in one bucket: {max_collision}")
print(f"Average locations per bucket: {len(location_to_hash) / n_features:.2f}")

# BONUS: Compare all solutions
print("\n📊 COMPARISON OF ALL SOLUTIONS:")
print("-" * 60)
print(f"Original problem: {len(missing_locations)} unseen locations")
print(f"\nSolution 1 (ignore): Zero vector for unseen")
print(f"  Dimensions: {len(encoder_safe.get_feature_names_out())}")
print(f"  Memory: Sparse representation possible")



print(f"\nSolution 2 (other category): Groups rare locations")
print(f"  Unseen reduced to: {len(missing_grouped)}")
print(f"  Information loss: High (all rare → 'other')")



print(f"\nSolution 3 (hashing): Fixed dimensions")
print(f"  Dimensions: {n_features} (fixed!)")
print(f"  Handles ANY new location automatically")
print(f"  Trade-off: Hash collisions")
