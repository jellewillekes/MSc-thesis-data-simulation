import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, expon, lognorm, gamma, beta, weibull_min, pareto, cauchy, chi2, bernoulli, binom, \
    geom, hypergeom, nbinom, poisson, randint, zipf, logser, dlaplace
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

np.random.seed(99)  # For reproducibility
n_samples = 100000
n_cat = 20
n_num = 20

# Continuous Distributions
normal_data = norm.rvs(loc=0, scale=1, size=(n_samples, 4))
exponential_data = expon.rvs(scale=1, size=(n_samples, 4))
lognormal_data = lognorm.rvs(s=1, size=(n_samples, 4))
uniform_data = uniform.rvs(loc=0, scale=10, size=(n_samples, 4))
weibull_data = weibull_min.rvs(c=2, scale=1, size=(n_samples, 4))

# Discrete Distributions
discrete_distributions = [
    bernoulli.rvs(p=0.5, size=(n_samples, 4)),
    binom.rvs(n=10, p=0.5, size=(n_samples, 4)),
    geom.rvs(p=0.5, size=(n_samples, 4)),
    poisson.rvs(mu=3, size=(n_samples, 4)),
    dlaplace.rvs(loc=0, a=1, size=(n_samples, 4))
]

# Shift and map discrete samples to alphabetical categories
alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
categorical_data = []

for dist_data in discrete_distributions:
    shifted_data = dist_data - np.min(dist_data)
    max_value = np.max(shifted_data)
    categorical_columns = np.array([alphabet[i % len(alphabet)] for i in range(max_value + 1)])
    categorical_data.append(categorical_columns[shifted_data])

# Combine numerical and categorical data
data = np.hstack(
    [normal_data, exponential_data, lognormal_data, uniform_data, weibull_data] + categorical_data)

# Create a DataFrame
columns = [f'F{i + 1}' for i in range(n_num + n_cat)]
dataset = pd.DataFrame(data, columns=columns)

df = dataset.copy()

# Convert categorical features to numerical values using ordinal encoding
ordinal_encoder = OrdinalEncoder()
category_columns = [f'F{i + n_cat}' for i in range(1, n_cat + 1)]
df[category_columns] = ordinal_encoder.fit_transform(df[category_columns])

# Convert numerical columns to float64
numerical_columns = [f'F{i}' for i in range(1, n_num + 1)]
df[numerical_columns] = df[numerical_columns].astype('float64')


# Define a nonlinear transformation to generate the target variable Y
def nonlinear_transformation(row):
    result = 0

    # Numerical features
    for i in range(1, n_num + 1, n_num // 2):
        num_product_term = row[f'F{i}'] * row[f'F{(i % n_num) + 1}']
        num_square_term = row[f'F{i + 1}'] ** 2
        num_sqrt_term = np.sqrt(np.abs(row[f'F{i + 2}']))
        result += num_product_term + num_square_term + num_sqrt_term

    # Categorical features
    for i in range(n_num + 1, n_num + n_cat, n_cat // 2):
        cat_product_term = row[f'F{i}'] * row[f'F{(i % n_cat) + n_num + 1}']
        cat_difference_term = np.abs(row[f'F{i + 1}'] - row[f'F{(i % n_cat) + n_num + 2}'])
        result += cat_product_term + cat_difference_term

    return result

# Apply the nonlinear transformation to obtain Y
df['Y'] = df.apply(nonlinear_transformation, axis=1)

# Scale Y to a range of [0, 100] using min-max scaling
scaler = MinMaxScaler(feature_range=(0, 100))
df['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1, 1))

# Randomize the dataset
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('synthetic_dataset_1.csv', index=False)

