import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet('okcupid_profiles.parquet')

# Apply one-hot encoding to location column
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Create train-test split
"""
You split your data into:
Training set (X_train): 80% of the data (47,956 rows)
Test set (X_test): 20% of the data (11,990 rows)
"""
X_train, X_test = train_test_split(df[['location']], test_size=0.2, random_state=42)

# Fit encoder on training data only
encoder.fit(X_train)

# Identify variables not appearing in the training set
train_locations = set(X_train['location'].dropna().unique())
test_locations = set(X_test['location'].dropna().unique())
missing_in_train = test_locations - train_locations

print(f"Locations in test but not in train: {len(missing_in_train)}")
print(f"Examples: {list(missing_in_train)[:10]}")  # Show first 10

# Transform both sets
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

print(f"\nTrain shape: {X_train_encoded.shape}")
print(f"Test shape: {X_test_encoded.shape}")
print(f"Number of unique locations in train: {len(train_locations)}")

"""
What happens with each set:
1. Training Set (X_train):
This is where the model "learns"
The OneHotEncoder looks at this data and says: "OK, I see 174 different locations here"
It creates 174 columns, one for each location it found
This becomes the model's "vocabulary" - all the locations it knows about

2. Test Set (X_test):
This is "new" data the model hasn't seen before
Used to evaluate how well the model performs on unseen data
Contains 25 locations that weren't in the training set
"""


"""
Real-world analogy:
- Imagine teaching someone to recognize US cities:
- Training set: You show them photos of 174 cities
- Test set: You quiz them with new photos, including 25 cities they've never seen
Problem: They can't recognize cities they were never taught!

Why split the data?
- You need to test your model on data it hasn't seen during training to:
- Check if it actually learned patterns (not just memorized)
- See how it handles new situations
- Simulate real-world performance

The "incomplete vocabulary" problem occurs because the random split accidentally put all instances of some locations (like 'ashland, california') into the test set, so the training set never got to learn about them."""


"""
Solution with 'other' category
Compute frequency per column.
Categories below frequency cutoff become "other".
Values not seen yet by the algorithm also go in "other"
"""