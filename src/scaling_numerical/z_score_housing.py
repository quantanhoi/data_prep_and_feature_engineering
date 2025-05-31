import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the housing data
housing_data = pd.read_csv('housing.csv')

# Extract the total_rooms column
total_rooms = housing_data['total_rooms'].values

# Method 1: Manual Z-score normalization
mean_rooms = np.mean(total_rooms)
std_rooms = np.std(total_rooms)
z_scores_manual = (total_rooms - mean_rooms) / std_rooms

print(f"Original data statistics:")
print(f"Mean: {mean_rooms:.2f}")
print(f"Standard Deviation: {std_rooms:.2f}")
print(f"Min: {np.min(total_rooms):.2f}")
print(f"Max: {np.max(total_rooms):.2f}")

# Method 2: Using sklearn StandardScaler
scaler = StandardScaler()
z_scores_sklearn = scaler.fit_transform(total_rooms.reshape(-1, 1)).flatten()

# Verify the transformation
print(f"\nAfter Z-score normalization:")
print(f"Mean: {np.mean(z_scores_manual):.6f}")
print(f"Standard Deviation: {np.std(z_scores_manual):.6f}")
print(f"Min: {np.min(z_scores_manual):.2f}")
print(f"Max: {np.max(z_scores_manual):.2f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original distribution
ax1.hist(total_rooms, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(mean_rooms, color='red', linestyle='--', label=f'Mean: {mean_rooms:.0f}')
ax1.set_title('Original Total Rooms Distribution')
ax1.set_xlabel('Total Rooms')
ax1.set_ylabel('Frequency')
ax1.legend()

# Z-score normalized distribution
ax2.hist(z_scores_manual, bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', label='Mean: 0')
ax2.axvline(-2, color='orange', linestyle=':', label='±2 std')
ax2.axvline(2, color='orange', linestyle=':')
ax2.set_title('Z-Score Normalized Distribution')
ax2.set_xlabel('Z-Score')
ax2.set_ylabel('Frequency')
ax2.legend()

plt.tight_layout()
plt.savefig('./images/z_score_housing.png', bbox_inches='tight')
plt.show()