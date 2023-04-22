import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('synthetic_dataset_1.csv')

# Assuming 'dataset' is a pandas DataFrame with columns 'Y' and features
target_col = 'Y'

numerical_columns = dataset.select_dtypes(include=np.number).columns.tolist()
categorical_columns = dataset.select_dtypes(include='object').columns.tolist()

if target_col in numerical_columns:
    numerical_columns.remove(target_col)

num_pairs = len(numerical_columns) // 4
num_rows = int(np.ceil(num_pairs / 4))

# Adjust the figsize to fit on a 13.3-inch (2560 × 1600) screen
fig, axes = plt.subplots(num_rows, 4, figsize=(15, 2 * num_rows))

for i, (ax, pair) in enumerate(zip(axes.flatten(), range(0, len(numerical_columns), 4))):
    col1 = numerical_columns[pair]
    col2 = numerical_columns[pair + 1]
    col3 = numerical_columns[pair + 2]
    col4 = numerical_columns[pair + 3]

    sns.histplot(dataset, x=col1, color="blue", alpha=0.5, ax=ax, kde=True, label=col1)
    sns.histplot(dataset, x=col2, color="orange", alpha=0.5, ax=ax, kde=True, label=col2)
    sns.histplot(dataset, x=col3, color="green", alpha=0.5, ax=ax, kde=True, label=col3)
    sns.histplot(dataset, x=col4, color="red", alpha=0.5, ax=ax, kde=True, label=col4)

    ax.set_title(f'Distribution of {col1}, {col2}, {col3}, and {col4}')
    ax.legend()

# Remove unused subplots
for j in range(i + 1, num_rows * 4):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()
