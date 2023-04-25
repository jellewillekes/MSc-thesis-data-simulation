import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load dataset
dataset = pd.read_csv('synthetic_dataset.csv')

# Define constants
TARGET_COL = 'Y'
NUM_SUBPLOTS_PER_ROW = 4

# Extract numerical and categorical columns
numerical_columns = dataset.select_dtypes(include=np.number).columns.tolist()
categorical_columns = dataset.select_dtypes(include='object').columns.tolist()

# Remove the target column if it's numerical
if TARGET_COL in numerical_columns:
    numerical_columns.remove(TARGET_COL)

# Combine numerical and categorical feature columns
feature_columns = numerical_columns + categorical_columns
num_feature_groups = len(feature_columns) // NUM_SUBPLOTS_PER_ROW
num_rows = int(np.ceil(num_feature_groups / NUM_SUBPLOTS_PER_ROW))

# Define distribution dictionary
distribution_dict = {
    'F1': 'C.Normal',
    'F5': 'C.Exponential',
    'F9': 'C.Lognormal',
    'F13': 'C.Uniform',
    'F17': 'C.Weibull',
    'F21': 'D.Bernoulli',
    'F25': 'D.Bionomial',
    'F29': 'D.Geometrical',
    'F33': 'D.Poisson',
    'F37': 'D.Laplace'
}


def calculate_bin_width(data):
    """
    Calculate the bin width for a histogram using the Freedman-Diaconis rule.

    Parameters:
    data (array-like): The data to calculate the bin width for.

    Returns:
    float: The bin width for the histogram.
    """
    num_data_points = len(data)
    data_range = np.max(data) - np.min(data)
    interquartile_range = stats.iqr(data)
    bin_width = 2 * interquartile_range / (num_data_points ** (1 / 3))

    return bin_width


def plot_target_distribution(ax):
    """Plot the distribution of the target variable."""
    sns.histplot(dataset, bins=1000, x=TARGET_COL, color="purple", alpha=0.5, ax=ax, kde=False, label=TARGET_COL)
    ax.set_title(f'Distribution of {TARGET_COL}')
    ax.legend()
    ax.set_xlim(dataset[TARGET_COL].min(), 80)
    ax.set_xlabel(f'Value of {TARGET_COL}', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)


def plot_feature_distributions(axes):
    """Plot the distributions of feature variables."""
    for index, (ax, pair) in enumerate(zip(axes.flatten()[1:], range(0, len(feature_columns), NUM_SUBPLOTS_PER_ROW))):
        cols_to_plot = feature_columns[pair:pair + NUM_SUBPLOTS_PER_ROW]
        colors = ["blue", "orange", "green", "red"]

        for col, color in zip(cols_to_plot, colors):
            if col in numerical_columns:
                bin_width = calculate_bin_width(dataset[col])
                sns.histplot(dataset, binwidth=bin_width, x=col, color=color, alpha=0.15, ax=ax, kde=False, label=col)
            elif col in categorical_columns:
                sns.countplot(x=col, data=dataset, color=color, alpha=0.15, ax=ax, label=col)

            if col == 'F5':
                ax.set_xlim(dataset[col].min(), 10)

            if col == 'F9':
                ax.set_xlim(dataset[col].min(), 15)

        ax.set_xlabel('Value of Feature', size=8)
        ax.set_ylabel('Frequency', size=8)
        ax.legend(fontsize=8)

        if cols_to_plot[0] in distribution_dict.keys():
            distribution_name = distribution_dict[cols_to_plot[0]]
            ax.set_title(f'{distribution_name} Features', size=10)

    fig.delaxes(axes.flatten()[-1])


# Create subplots
fig, axes = plt.subplots(num_rows, NUM_SUBPLOTS_PER_ROW, figsize=(15, 2 * num_rows))

# Adjust subplots spacing
fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)

# Plot target variable distribution
ax_target = axes.flatten()[0]
plot_target_distribution(ax_target)

# Plot feature variable distributions
plot_feature_distributions(axes)
plt.savefig('plots/features_histograms.png')
plt.show()
