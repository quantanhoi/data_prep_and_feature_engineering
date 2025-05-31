import numpy as np
import matplotlib.pyplot as plt

# Your data
test_scores = np.array([1, 72, 78, 85, 90, 95, 88, 76, 82, 91])

# Calculate Z-scores
mean_score = np.mean(test_scores)
# Standard deviation measures how spread out data points a re from the mean, 
# it tells you whether your data points are clustered tightlz around the average or scattered widely across the range
std_score = np.std(test_scores)
z_scores = (test_scores - mean_score) / std_score

print(f"Mean: {mean_score:.2f}")
print(f"Standard Deviation: {std_score:.2f}")

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Z-Score Normalization Visualization', fontsize=16, fontweight='bold')

# 1. Original Data Distribution
axes[0, 0].hist(test_scores, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
axes[0, 0].set_title('Original Test Scores Distribution')
axes[0, 0].set_xlabel('Test Scores')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Z-Scores Distribution
axes[0, 1].hist(z_scores, bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean: 0')
axes[0, 1].axvline(-2, color='orange', linestyle=':', label='±2 std')
axes[0, 1].axvline(2, color='orange', linestyle=':')
axes[0, 1].set_title('Z-Score Normalized Distribution')
axes[0, 1].set_xlabel('Z-Scores')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Before vs After Comparison (Scatter Plot)
student_numbers = np.arange(1, len(test_scores) + 1)
axes[1, 0].scatter(student_numbers, test_scores, color='blue', s=100, alpha=0.7, label='Original Scores')
axes[1, 0].scatter(student_numbers, z_scores * 20 + mean_score, color='red', s=100, alpha=0.7, label='Z-Scores (scaled for comparison)')
axes[1, 0].set_title('Original vs Z-Score Normalized (Scaled)')
axes[1, 0].set_xlabel('Student Number')
axes[1, 0].set_ylabel('Score Value')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Z-Score Values Bar Chart
colors = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1 else 'green' for z in z_scores]
bars = axes[1, 1].bar(student_numbers, z_scores, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].axhline(2, color='red', linestyle='--', alpha=0.5, label='Outlier threshold (±2)')
axes[1, 1].axhline(-2, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Z-Scores by Student (Color-coded by Magnitude)')
axes[1, 1].set_xlabel('Student Number')
axes[1, 1].set_ylabel('Z-Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, z_score) in enumerate(zip(bars, z_scores)):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05 if height > 0 else height - 0.15,
                    f'{z_score:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig("./images/z_score.png", bbox_inches='tight')
plt.show()
