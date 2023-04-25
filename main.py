import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, expon, lognorm, weibull_min, bernoulli, binom, geom, poisson, dlaplace
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

np.random.seed(99)  # For reproducibility
n_samples = 1000000
n_cat = 20
n_num = 20


def generate_continuous_data(n_samples):
    normal_data = norm.rvs(loc=0, scale=1, size=(n_samples, 4))
    exponential_data = expon.rvs(scale=1, size=(n_samples, 4))
    lognormal_data = lognorm.rvs(s=1, size=(n_samples, 4))
    uniform_data = uniform.rvs(loc=0, scale=10, size=(n_samples, 4))
    weibull_data = weibull_min.rvs(c=2, scale=1, size=(n_samples, 4))
    return np.hstack([normal_data, exponential_data, lognormal_data, uniform_data, weibull_data])


def generate_categorical_data(n_samples):
    discrete_distributions = [
        bernoulli.rvs(p=0.5, size=(n_samples, 4)),
        binom.rvs(n=10, p=0.5, size=(n_samples, 4)),
        geom.rvs(p=0.5, size=(n_samples, 4)),
        poisson.rvs(mu=3, size=(n_samples, 4)),
        dlaplace.rvs(loc=0, a=1, size=(n_samples, 4))
    ]
    alphabet = list('ABCDEFGHIJ')
    categorical_data = []

    for dist_data in discrete_distributions:
        shifted_data = dist_data - np.min(dist_data)
        max_value = np.max(shifted_data)
        categorical_columns = np.array([alphabet[i % len(alphabet)] for i in range(max_value + 1)])
        categorical_data.append(categorical_columns[shifted_data])

    return np.hstack(categorical_data)


def create_dataset(n_samples, n_cat, n_num):
    continuous_data = generate_continuous_data(n_samples)
    categorical_data = generate_categorical_data(n_samples)
    data = np.hstack([continuous_data, categorical_data])
    columns = [f'F{i + 1}' for i in range(n_num + n_cat)]
    return pd.DataFrame(data, columns=columns)


def preprocess_dataset(dataset, n_cat, n_num):
    df = dataset.copy()
    category_columns = [f'F{i + n_cat}' for i in range(1, n_cat + 1)]
    numerical_columns = [f'F{i}' for i in range(1, n_num + 1)]

    ordinal_encoder = OrdinalEncoder()
    df[category_columns] = ordinal_encoder.fit_transform(df[category_columns])
    df[numerical_columns] = df[numerical_columns].astype('float64')

    return df, category_columns, numerical_columns


def updated_generate_target_variable(df, numerical_columns, category_columns):
    # Apply simpler transformations to numerical features
    transformed_num_features = np.hstack([
        df[numerical_columns[:4]].values,
        df[numerical_columns[4:8]].values,
        df[numerical_columns[8:12]].values,
        df[numerical_columns[12:16]].values,
        df[numerical_columns[16:20]].values
    ])

    # Create interactions between a smaller number of selected features (pairwise interactions)
    selected_interactions = [
        (numerical_columns[0], category_columns[0]),
        (numerical_columns[4], category_columns[4]),
        (numerical_columns[8], category_columns[8]),
        (numerical_columns[12], category_columns[12]),
        (numerical_columns[16], category_columns[16]),
    ]

    interaction_features = np.zeros((n_samples, len(selected_interactions)))

    for i, (num_col, cat_col) in enumerate(selected_interactions):
        interaction_features[:, i] = df[num_col] * df[cat_col]

    # Add interactions between numerical variables from different distributions
    numerical_interactions = np.zeros((n_samples, 5))

    for i in range(5):
        numerical_interactions[:, i] = df[numerical_columns[i]] * df[numerical_columns[i + 4]]

    # Add interactions between numerical variables from different distributions
    categorical_interactions = np.zeros((n_samples, 5))

    for i in range(5):
        categorical_interactions[:, i] = df[category_columns[i]] * df[category_columns[i + 4]]

    # Combine transformed numerical features, selected interactions, and numerical interactions
    X_transformed = np.hstack(
        [transformed_num_features, interaction_features, numerical_interactions, categorical_interactions])

    # Generate the target variable Y as a simple linear combination of the transformed features
    Y = np.sum(X_transformed, axis=1)
    Y = np.sqrt(Y)
    df['Y'] = Y

    return df


def generate_target_variable(df, numerical_columns, category_columns):
    transformed_num_features = np.hstack([
        np.sqrt(df[numerical_columns[:4]].values),
        np.log1p(df[numerical_columns[4:8]].values),
        np.power(df[numerical_columns[8:12]].values, 0.5),
        df[numerical_columns[12:16]].values,
        df[numerical_columns[16:20]].values
    ])

    interaction_features = np.zeros((n_samples, 2 * (n_num + n_cat)))

    # Interactions between numerical and categorical features
    for i, num_col in enumerate(numerical_columns):
        interaction_features[:, i] = df[num_col] * df[category_columns[i % n_cat]]

    for i, cat_col in enumerate(category_columns):
        interaction_features[:, n_num + i] = df[cat_col] * df[numerical_columns[i % n_num]]

    # Interactions between numerical features
    for i, num_col1 in enumerate(numerical_columns[:4]):
        num_col2 = numerical_columns[i + 4]
        interaction_features[:, 2 * n_num + i] = df[num_col1] * df[num_col2]

    # Interactions between categorical features
    for i, cat_col1 in enumerate(category_columns[:4]):
        cat_col2 = category_columns[i + 4]
        interaction_features[:, 2 * (n_num + n_cat) - 4 + i] = df[cat_col1] * df[cat_col2]

    X_transformed = np.hstack([transformed_num_features, interaction_features])
    Y = np.sum(X_transformed, axis=1)
    df['Y'] = Y

    return df


def scale_and_randomize(df):
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1, 1))
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_bins(df, n_bins):
    bins = pd.cut(df['Y'], bins=n_bins)
    y_counts = bins.value_counts().sort_index()
    df_counts = pd.DataFrame({'bins': y_counts.index, 'counts': y_counts.values})
    return df_counts


def plot_histogram(df, n_bins):
    plt.hist(df['Y'], bins=n_bins)
    plt.xlabel('Y values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Y column with {n_bins} bins')
    plt.show()


dataset = create_dataset(n_samples, n_cat, n_num)
df, category_columns, numerical_columns = preprocess_dataset(dataset, n_cat, n_num)
df = updated_generate_target_variable(df, numerical_columns, category_columns)
dataset['Y'] = df['Y']
dataset = scale_and_randomize(dataset)
df_counts = create_bins(dataset, 100)
print(df_counts)
plot_histogram(dataset, 10)

dataset.to_csv('synthetic_dataset.csv', index=False)
