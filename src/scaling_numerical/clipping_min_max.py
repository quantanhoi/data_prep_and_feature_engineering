import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the housing data
housing_data = pd.read_csv('housing.csv')

# 1. Plot histogram of total_rooms to visualize distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(housing_data['total_rooms'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Original total_rooms Distribution')
plt.xlabel('Total Rooms')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Estimate reasonable bounds visually
# From the histogram, we can see most values are between 1000-8000
# Let's use percentiles to set reasonable bounds
lower_bound = np.percentile(housing_data['total_rooms'], 5)   # 5th percentile
upper_bound = np.percentile(housing_data['total_rooms'], 95)  # 95th percentile

print(f"5th percentile: {lower_bound:.0f}")
print(f"95th percentile: {upper_bound:.0f}")
print(f"Min value: {housing_data['total_rooms'].min()}")
print(f"Max value: {housing_data['total_rooms'].max()}")

# 3. Clip the column using reasonable bounds
total_rooms_clipped = np.clip(housing_data['total_rooms'], lower_bound, upper_bound)

# 4. Apply Min-Max scaling to [-1, 1]
# FIX: Convert to numpy array first, then reshape
scaler = MinMaxScaler(feature_range=(-1, 1))
total_rooms_scaled = scaler.fit_transform(total_rooms_clipped.values.reshape(-1, 1)).flatten()

# 5. Verify the transformed column is in range [-1, 1]
print(f"\nAfter clipping and scaling:")
print(f"Min value: {total_rooms_scaled.min():.6f}")
print(f"Max value: {total_rooms_scaled.max():.6f}")
print(f"Mean value: {total_rooms_scaled.mean():.6f}")

# 6. Plot histogram of the transformed column
plt.subplot(1, 3, 2)
plt.hist(total_rooms_clipped, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('After Clipping')
plt.xlabel('Total Rooms (Clipped)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(total_rooms_scaled, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('After Clipping + Min-Max Scaling [-1,1]')
plt.xlabel('Scaled Total Rooms')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./images/clipping_min_max.png', bbox_inches='tight')
plt.show()



# 7. Compare with sklearn's MinMaxScaler(clip=True)
print("\n" + "="*60)
print("COMPARISON: Manual Clipping vs sklearn clip=True")
print("="*60)

# Method 1: Our manual approach (clip first, then scale)
manual_result = total_rooms_scaled

# Method 2: sklearn's clip=True (different behavior)
scaler_with_clip = MinMaxScaler(feature_range=(-1, 1), clip=True)
sklearn_result = scaler_with_clip.fit_transform(housing_data['total_rooms'].values.reshape(-1, 1)).flatten()

print(f"Manual approach (clip then scale):")
print(f"  Min: {manual_result.min():.6f}, Max: {manual_result.max():.6f}")
print(f"  Values at bounds: {np.sum(manual_result == -1)} at -1, {np.sum(manual_result == 1)} at 1")

print(f"\nsklearn clip=True:")
print(f"  Min: {sklearn_result.min():.6f}, Max: {sklearn_result.max():.6f}")
print(f"  Values at bounds: {np.sum(sklearn_result == -1)} at -1, {np.sum(sklearn_result == 1)} at 1")

# Show the difference
print(f"\nKey Difference:")
print(f"- Manual approach: Clips outliers BEFORE scaling, treating them as the boundary values")
print(f"- sklearn clip=True: Scales first, then clips any values outside [-1,1] range")
print(f"- This results in different distributions and outlier handling")

# Additional analysis
print(f"\nOutlier Analysis:")
original_outliers_low = np.sum(housing_data['total_rooms'] < lower_bound)
original_outliers_high = np.sum(housing_data['total_rooms'] > upper_bound)
print(f"Original outliers below {lower_bound:.0f}: {original_outliers_low}")
print(f"Original outliers above {upper_bound:.0f}: {original_outliers_high}")
print(f"Total outliers clipped: {original_outliers_low + original_outliers_high}")
