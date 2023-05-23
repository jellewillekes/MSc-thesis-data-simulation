import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Define constants
np.random.seed(99)  # For reproducibility
n_samples = 100000
n_cat = 5
n_num = 5

TARGET_COL = 'Y'
NUM_SUBPLOTS_PER_ROW = 2
COLOR_MAP = {'target': 'red', 'numerical': 'green', 'categorical': 'blue'}


def load_dataset(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def calculate_bin_width(data):
    """Calculate the bin width for a histogram using the Freedman-Diaconis rule."""
    num_data_points = len(data)
    np.max(data) - np.min(data)
    interquartile_range = stats.iqr(data)
    bin_width = 2 * interquartile_range / (num_data_points ** (1 / 3))

    return bin_width


def int_to_letter(n):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    quot, rem = divmod(n, len(alphabet))
    return alphabet[rem] * (quot + 1)


def letter_to_int(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    number = 0
    for char in letter:
        number = 26 * number + (alphabet.index(char) + 1)
    return number - 1


def set_categorical_labels(dataset):
    # Columns for the categorical variables
    category_columns = [f'cat_{i}' for i in range(1, n_cat + 1)]

    for col in category_columns:
        dataset[col] = dataset[col].apply(lambda x: int_to_letter(int(x)))

    return dataset


def set_numerical_labels(dataset):
    # Columns for the categorical variables
    category_columns = [f'cat_{i}' for i in range(1, n_cat + 1)]

    # Replace the values in the categorical columns
    for col in category_columns:
        dataset[col] = dataset[col].apply(lambda x: letter_to_int(x))

    return dataset


def plot_target_distribution(dataset, ax_target):
    """Plot the distribution of the target variable."""
    bw = calculate_bin_width(dataset[TARGET_COL])
    sns.histplot(dataset, binwidth=bw, x=TARGET_COL, color=COLOR_MAP['target'], alpha=0.5, ax=ax_target, kde=False,
                 label=TARGET_COL)
    ax_target.set_title(f'Distribution of {TARGET_COL}')
    ax_target.legend()
    ax_target.set_xlim(dataset[TARGET_COL].min(), 10)
    ax_target.set_xlabel(f'Value of {TARGET_COL}', fontsize=10)
    ax_target.set_ylabel('Frequency', fontsize=10)


def plot_feature_distributions(dataset, numerical_columns, categorical_columns, distribution_dict, axes):
    """Plot the distributions of feature variables."""
    feature_columns = numerical_columns + categorical_columns
    last_used_index = 0
    for index in range(0, len(feature_columns)):
        row, col = divmod(index, NUM_SUBPLOTS_PER_ROW)
        ax = axes[row][col]
        feature = feature_columns[index]

        if feature in numerical_columns:
            bin_width = calculate_bin_width(dataset[feature])
            sns.histplot(dataset, binwidth=bin_width, x=feature, color='green', alpha=0.15, ax=ax,
                         kde=False, label=feature)
        elif feature in categorical_columns:
            counts = dataset[feature].value_counts()
            counts = counts.reset_index()
            counts.index = counts.iloc[:, 0].apply(lambda x: int_to_letter(int(x)))
            counts = counts.drop(feature, axis=1)
            ax.bar(counts.index, counts.iloc[:, 0], color='blue', alpha=0.15, label=feature)

        distribution_name = distribution_dict.get(feature, "Unknown")
        ax.set_title(f'{distribution_name} Feature: {feature}', size=10)
        ax.set_ylabel('X', color='white', size=1)
        ax.set_xlabel(f'X', color='white', size=1)
        ax.legend(fontsize=8)
        last_used_index = index

    return last_used_index  # Return the last index used


def plot_distributions(dataset, numerical_columns, categorical_columns, distribution_dict):
    """Create plots for the target and feature variables."""
    # Create a separate plot for the target variable
    # fig_target, ax_target = plt.subplots(figsize=(15, 5))
    # plot_target_distribution(dataset, ax_target)
    # plt.show()
    # plt.close()

    # Create subplots for the features
    num_rows = int(np.ceil(len(numerical_columns + categorical_columns) / NUM_SUBPLOTS_PER_ROW))
    fig, axes = plt.subplots(num_rows, NUM_SUBPLOTS_PER_ROW, figsize=(15, 1.5 * num_rows))

    # Always ensure axes is a 2D array, even when num_rows or NUM_SUBPLOTS_PER_ROW is 1
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if NUM_SUBPLOTS_PER_ROW == 1:
        axes = axes.reshape(-1, 1)

    # Adjust subplots spacing
    fig.subplots_adjust(left=0.1, right=0.9, top=0.90, bottom=0.10, wspace=0.2, hspace=0.6)

    # Plot feature variable distributions
    last_used_index = plot_feature_distributions(dataset, numerical_columns, categorical_columns, distribution_dict,
                                                 axes)

    # Remove unused subplots
    for unused_subplot_index in range(last_used_index + 1, num_rows * NUM_SUBPLOTS_PER_ROW):
        fig.delaxes(axes.flatten()[unused_subplot_index])

    # Add an overall title and labels
    # fig.suptitle('Feature Distributions', fontsize=15, fontweight='bold', y=1.02)
    fig.text(0.5, 0.04, 'Value', ha='center', va='center', fontsize=12)
    fig.text(0.04, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)

    plt.show()


# Load dataset
dataset = load_dataset('synthetic_dataset.csv')

# Extract numerical and categorical columns
# numerical_columns = dataset.select_dtypes(include=np.number).columns.tolist()
# categorical_columns = dataset.select_dtypes(include='object').columns.tolist()
numerical_columns = ['num_1', 'num_2', 'num_3', 'num_4', 'num_5']
categorical_columns = ['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5']

# Remove the target column if it's numerical
if TARGET_COL in numerical_columns:
    numerical_columns.remove(TARGET_COL)

# Define the expected distributions
distribution_dict = {
    'num_1': 'Normal',
    'num_2': 'Exponential',
    'num_3': 'Lognormal',
    'num_4': 'Uniform',
    'num_5': 'Weibull',
    'cat_1': 'Bernoulli',
    'cat_2': 'Binomial',
    'cat_3': 'Geometrical',
    'cat_4': 'Poisson',
    'cat_5': 'Neg. Binomial'
}

# Create the plots
plot_distributions(dataset, numerical_columns, categorical_columns, distribution_dict)
