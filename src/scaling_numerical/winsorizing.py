import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the housing data
housing_data = pd.read_csv('housing.csv')

def winsorize_column(data, column_name, lower_percentile=5, upper_percentile=95):
    """
    Apply winsorizing to a specific column
    
    Parameters:
    - data: DataFrame
    - column_name: name of column to winsorize
    - lower_percentile: lower bound percentile (default 5%)
    - upper_percentile: upper bound percentile (default 95%)
    """
    # Calculate percentile bounds
    lower_bound = np.percentile(data[column_name], lower_percentile)
    upper_bound = np.percentile(data[column_name], upper_percentile)
    
    # Apply winsorizing
    winsorized_data = data[column_name].copy()
    winsorized_data = np.where(winsorized_data < lower_bound, lower_bound, winsorized_data)
    winsorized_data = np.where(winsorized_data > upper_bound, upper_bound, winsorized_data)
    
    return winsorized_data, lower_bound, upper_bound

# Example: Winsorize total_rooms column
column_to_winsorize = 'total_rooms'
winsorized_rooms, lower_bound, upper_bound = winsorize_column(
    housing_data, 
    column_to_winsorize, 
    lower_percentile=5, 
    upper_percentile=95
)

print(f"Original {column_to_winsorize} statistics:")
print(f"Min: {housing_data[column_to_winsorize].min()}")
print(f"Max: {housing_data[column_to_winsorize].max()}")
print(f"Mean: {housing_data[column_to_winsorize].mean():.2f}")
print(f"Std: {housing_data[column_to_winsorize].std():.2f}")

print(f"\nWinsorizing bounds:")
print(f"Lower bound (5th percentile): {lower_bound}")
print(f"Upper bound (95th percentile): {upper_bound}")

print(f"\nWinsorized {column_to_winsorize} statistics:")
print(f"Min: {winsorized_rooms.min()}")
print(f"Max: {winsorized_rooms.max()}")
print(f"Mean: {winsorized_rooms.mean():.2f}")
print(f"Std: {winsorized_rooms.std():.2f}")

# Count how many values were winsorized
original_values = housing_data[column_to_winsorize]
outliers_low = np.sum(original_values < lower_bound)
outliers_high = np.sum(original_values > upper_bound)

print(f"\nOutliers handled:")
print(f"Values below {lower_bound}: {outliers_low}")
print(f"Values above {upper_bound}: {outliers_high}")
print(f"Total outliers winsorized: {outliers_low + outliers_high}")
print(f"Percentage of data winsorized: {((outliers_low + outliers_high) / len(original_values)) * 100:.2f}%")



# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Winsorizing Effect on Total Rooms Distribution', fontsize=16)

# Original distribution
axes[0, 0].hist(housing_data['total_rooms'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('Original Distribution')
axes[0, 0].set_xlabel('Total Rooms')
axes[0, 0].set_ylabel('Frequency')

# Winsorized distribution
axes[0, 1].hist(winsorized_rooms, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title('Winsorized Distribution (5%-95%)')
axes[0, 1].set_xlabel('Total Rooms')
axes[0, 1].set_ylabel('Frequency')

# Box plots comparison
box_data = [housing_data['total_rooms'], winsorized_rooms]
box_labels = ['Original', 'Winsorized']
axes[1, 0].boxplot(box_data, labels=box_labels)
axes[1, 0].set_title('Box Plot Comparison')
axes[1, 0].set_ylabel('Total Rooms')

# Scatter plot showing winsorizing effect
axes[1, 1].scatter(range(len(housing_data)), housing_data['total_rooms'], 
                   alpha=0.5, s=1, label='Original', color='blue')
axes[1, 1].scatter(range(len(winsorized_rooms)), winsorized_rooms, 
                   alpha=0.5, s=1, label='Winsorized', color='red')
axes[1, 1].axhline(y=lower_bound, color='orange', linestyle='--', label=f'Lower bound: {lower_bound}')
axes[1, 1].axhline(y=upper_bound, color='orange', linestyle='--', label=f'Upper bound: {upper_bound}')
axes[1, 1].set_title('Original vs Winsorized Values')
axes[1, 1].set_xlabel('Data Point Index')
axes[1, 1].set_ylabel('Total Rooms')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('./images/winsorizing.png', bbox_inches='tight')
plt.show()

