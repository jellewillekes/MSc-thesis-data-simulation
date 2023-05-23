import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr, uniform, norm, expon, bernoulli, binom, geom, poisson, lognorm, nbinom, weibull_min
from sklearn.preprocessing import MinMaxScaler

np.random.seed(99)  # For reproducibility
n_samples = 100000
n_cat = 5
n_num = 5

distribution_dict = {
    'num_1': (norm, {}),
    'num_2': (expon, {'scale': 1}),
    'num_3': (lognorm, {'s': 1}),
    'num_4': (uniform, {'loc': 0, 'scale': 100}),
    'num_5': (weibull_min, {'c': 2}),
    'cat_1': (bernoulli, {'p': 0.5}),
    'cat_2': (binom, {'n': 5, 'p': 0.5}),
    'cat_3': (geom, {'p': 0.5}),
    'cat_4': (poisson, {'mu': 5}),
    'cat_5': (nbinom, {'n': 5, 'p': 0.8})
}


def generate_continuous_data(n_samples, n_num):
    constants = [10, 0, 10, 0, 5]
    data = [dist.rvs(size=n_samples, **params) + constants[i] for i in range(n_num) for dist, params in
            [distribution_dict[f'num_{i + 1}']]]
    return np.column_stack(data)


def generate_categorical_data(n_samples, n_cat):
    data = [dist.rvs(size=n_samples, **params) for i in range(n_cat) for dist, params in
            [distribution_dict[f'cat_{i + 1}']]]
    return np.column_stack(data)


def create_dataset(n_samples, n_cat, n_num):
    continuous_data = generate_continuous_data(n_samples, n_num)
    categorical_data = generate_categorical_data(n_samples, n_cat)
    data = np.hstack([continuous_data, categorical_data])
    columns = [f'num_{i + 1}' for i in range(n_num)] + [f'cat_{i + 1}' for i in range(n_cat)]
    return pd.DataFrame(data, columns=columns)


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


def calculate_bin_width(data):
    """Calculate the bin width for a histogram using the Freedman-Diaconis rule."""
    num_data_points = len(data)
    np.max(data) - np.min(data)
    interquartile_range = iqr(data)
    bin_width = 2 * interquartile_range / (num_data_points ** (1 / 3))

    return bin_width


def generate_target_variable(df):
    # Generate the target using a complex, non-linear function of the features
    Y = df['num_1']**2 + np.sin(df['num_2']) + df['num_3'] * df['num_4'] + np.exp(-df['num_5'])

    # Introduce interactions between some of the categorical features as well
    Y += df['cat_1'] * df['cat_2'] + df['cat_3']**2 * df['cat_4'] + np.sqrt(df['cat_5'])

    # Finally, introduce interactions between some of the numerical and categorical features
    Y += df['num_1'] * df['cat_1'] + df['num_2'] * df['cat_2']**2 + df['num_3'] * np.sqrt(df['cat_3']) + df['num_5']

    return Y


def scale_and_randomize(df):
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1, 1))
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def plot_histogram(df, n_bins):
    plt.hist(df['Y'], n_bins)
    plt.xlabel('Y values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Y column with {n_bins} bins')
    plt.show()


dataset = create_dataset(n_samples, n_cat, n_num)
Y = generate_target_variable(dataset)
dataset['Y'] = Y
dataset = scale_and_randomize(dataset)

# Plotting histogram
bw = calculate_bin_width(Y)
n_bins = int((max(Y) - min(Y)) / bw)
plot_histogram(dataset, n_bins)

# Saving the dataset to a CSV file
dataset.to_csv('synthetic_dataset.csv', index=False)
