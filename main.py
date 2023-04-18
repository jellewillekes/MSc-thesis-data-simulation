import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, expon, gamma, beta, binom, poisson, lognorm, geom, weibull_min
import string
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

import matplotlib.pyplot as plt

np.random.seed(99)  # For reproducibility
n_samples = 100000
n_cat = 20
n_num = 20

# Generate data for each distribution
uniform_data = uniform.rvs(loc=0, scale=10, size=(n_samples, 2))
normal_data = norm.rvs(loc=0, scale=1, size=(n_samples, 2))
exponential_data = expon.rvs(scale=1, size=(n_samples, 2))
gamma_data = gamma.rvs(a=2, scale=1, size=(n_samples, 2))
beta_data = beta.rvs(a=2, b=5, size=(n_samples, 2))
binomial_data = binom.rvs(n=10, p=0.5, size=(n_samples, 2))
poisson_data = poisson.rvs(mu=3, size=(n_samples, 2))
lognormal_data = lognorm.rvs(s=0.5, scale=np.exp(0.5), size=(n_samples, 2))
geometric_data = geom.rvs(p=0.3, size=(n_samples, 2))
weibull_data = weibull_min.rvs(c=1.5, scale=2, size=(n_samples, 2))

# Combine data into a single dataset
data = np.hstack(
    [uniform_data, normal_data, exponential_data, gamma_data, beta_data, binomial_data, poisson_data, lognormal_data,
     geometric_data, weibull_data])

# Create a DataFrame
columns = [f'F{i + 1}' for i in range(n_num)]
df = pd.DataFrame(data, columns=columns)

# Alphabet letters to use as labels
alphabet = list(string.ascii_uppercase)[:10]

# Add 25 categorical variables with random number of labels between 2 and 10
for i in range(1, n_cat+1):
    num_labels = np.random.randint(2, 11)
    num_digits = np.random.randint(1, 4)
    letter = np.random.choice(alphabet, num_labels, replace=False)
    labels = [s.strip() * num_digits for s in letter]
    df[f'F{i + n_num}'] = np.random.choice(labels, n_samples)

dataset = df.copy()

# Convert categorical features to numerical values using ordinal encoding
ordinal_encoder = OrdinalEncoder()
category_columns = [f'F{i + n_cat}' for i in range(1, n_cat + 1)]
df[category_columns] = ordinal_encoder.fit_transform(df[category_columns])


# Define a nonlinear transformation to generate the target variable Y
def nonlinear_transformation(row):
    result = 0

    # Numerical features
    for i in range(1, n_num + 1, n_num//2):
        num_product_term = row[f'F{i}'] * row[f'F{(i % n_num) + 1}']
        num_square_term = row[f'F{i + 1}'] ** 2
        num_sqrt_term = np.sqrt(np.abs(row[f'F{i + 2}']))
        result += num_product_term + num_square_term + num_sqrt_term

    # Categorical features
    for i in range(n_num + 1, n_num + n_cat, n_cat//2):
        cat_product_term = row[f'F{i}'] * row[f'F{(i % n_cat) + n_num + 1}']
        cat_difference_term = np.abs(row[f'F{i + 1}'] - row[f'F{(i % n_cat) + n_num + 2}'])
        result += cat_product_term + cat_difference_term

    return result


# Apply the nonlinear transformation to obtain Y
df['Y'] = df.apply(nonlinear_transformation, axis=1)

# Scale Y to a range of [0, 100] using min-max scaling
scaler = MinMaxScaler(feature_range=(0, 100))
df['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1, 1))

dataset['Y'] = df['Y']

# Randomize the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset.to_csv('synthetic_dataset.csv', index=False)
