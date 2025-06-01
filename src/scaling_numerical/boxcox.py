import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the housing data
df = pd.read_csv('housing.csv')

# Check for any missing values in total_rooms column
print(f"Missing values in total_rooms: {df['total_rooms'].isna().sum()}")

# Remove any rows with missing total_rooms values
df_clean = df.dropna(subset=['total_rooms'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram before transformation
ax1.hist(df_clean['total_rooms'], bins=50, color='blue', alpha=0.7, edgecolor='black')
ax1.set_title('Distribution of Total Rooms (Before Box-Cox)', fontsize=14)
ax1.set_xlabel('Total Rooms')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Apply Box-Cox transformation
# Box-Cox requires positive values, and total_rooms should already be positive
# The boxcox function returns both transformed data and the lambda parameter
total_rooms_transformed, lambda_param = stats.boxcox(df_clean['total_rooms'])

# Plot histogram after transformation
ax2.hist(total_rooms_transformed, bins=50, color='green', alpha=0.7, edgecolor='black')
ax2.set_title('Distribution of Total Rooms (After Box-Cox)', fontsize=14)
ax2.set_xlabel('Transformed Total Rooms')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

# Add lambda value to the plot
ax2.text(0.02, 0.98, f'λ = {lambda_param:.4f}', 
         transform=ax2.transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('./images/boxcox.png', bbox_inches='tight')
plt.show()

# Print some statistics
print(f"\nBox-Cox Lambda parameter: {lambda_param:.4f}")
print(f"\nOriginal data statistics:")
print(f"  Mean: {df_clean['total_rooms'].mean():.2f}")
print(f"  Std: {df_clean['total_rooms'].std():.2f}")
print(f"  Skewness: {stats.skew(df_clean['total_rooms']):.4f}")

print(f"\nTransformed data statistics:")
print(f"  Mean: {np.mean(total_rooms_transformed):.2f}")
print(f"  Std: {np.std(total_rooms_transformed):.2f}")
print(f"  Skewness: {stats.skew(total_rooms_transformed):.4f}")

# Optional: Add the transformed data back to the dataframe
df_clean['total_rooms_boxcox'] = total_rooms_transformed
