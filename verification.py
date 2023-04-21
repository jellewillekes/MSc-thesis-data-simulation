import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('synthetic_dataset.csv')

# Assuming 'dataset' is a pandas DataFrame with columns 'Y' and features
target_col = 'Y'
significance_level = 0.05

# Separate numerical and categorical columns
numerical_cols = dataset.select_dtypes(include=np.number).columns.tolist()
categorical_cols = dataset.select_dtypes(include='object').columns.tolist()

# Remove the target column from numerical columns
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Group the numerical columns by 2
grouped_cols = [numerical_cols[i:i + 2] for i in range(0, len(numerical_cols), 2)]

# Determine the number of rows and columns for the subplots
nrows = (len(grouped_cols) + 1) // 3 + ((len(grouped_cols) + 1) % 3 > 0)
ncols = 3

# Create the subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 4), sharey=False)
axes = axes.ravel()

for i, group in enumerate(grouped_cols):
    ax = axes[i]
    for j, col in enumerate(group):
        dataset[col].value_counts(normalize=True).sort_index().plot(ax=ax, label=col)

    # Set the y-axis range based on the columns
    min_value = min(dataset[group].min())
    max_value = max(dataset[group].max())
    ax.set_ylim(min_value, max_value)

    ax.set_title(f'Histogram of Columns Group {i + 1}')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Value')
    ax.legend()

# Plot the target variable in the last subplot
ax = axes[-1]
dataset[target_col].value_counts(normalize=True).sort_index().plot(ax=ax, label=target_col)
ax.set_title('Histogram of Target Variable')
ax.set_xlabel('Frequency')
ax.set_ylabel('Value')
ax.legend()

# Remove empty subplots if there's an odd number of groups
if (len(grouped_cols) + 1) % ncols != 0:
    for i in range((len(grouped_cols) + 1) % ncols, ncols):
        fig.delaxes(axes[-i])

plt.tight_layout()
plt.show()
