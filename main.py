import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, expon, gamma, beta
import string
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

import matplotlib.pyplot as plt

np.random.seed(99)  # For reproducibility
n_samples = 100000

# Generate data for each distribution
uniform_data = uniform.rvs(loc=0, scale=10, size=(n_samples, 5))
normal_data = norm.rvs(loc=0, scale=1, size=(n_samples, 5))
exponential_data = expon.rvs(scale=1, size=(n_samples, 5))
gamma_data = gamma.rvs(a=2, scale=1, size=(n_samples, 5))
beta_data = beta.rvs(a=2, b=5, size=(n_samples, 5))

# Combine data into a single dataset
data = np.hstack([uniform_data, normal_data, exponential_data, gamma_data, beta_data])

# Create a DataFrame
columns = [f'F{i + 1}' for i in range(25)]
df = pd.DataFrame(data, columns=columns)

# Alphabet letters to use as labels
alphabet = list(string.ascii_uppercase)[:10]

# Add 25 categorical variables with random number of labels between 5 and 10
for i in range(1, 26):
    num_labels = np.random.randint(2, 11)
    num_digits = np.random.randint(1, 4)
    letter = np.random.choice(alphabet, num_labels, replace=False)
    labels = [s.strip() * num_digits for s in letter]
    df[f'F{i + 25}'] = np.random.choice(labels, n_samples)

dataset = df.copy()

# Convert categorical features to numerical values using ordinal encoding
ordinal_encoder = OrdinalEncoder()
category_columns = [f'F{i + 25}' for i in range(1, 26)]
df[category_columns] = ordinal_encoder.fit_transform(df[category_columns])


# Define a nonlinear transformation to generate the target variable Y
def nonlinear_transformation(row):
    result = 0
    for i in range(1, 51, 5):
        log_term = np.log(row[f'F{i}']) if row[f'F{i}'] > 0 else 0
        product_term = row[f'F{i + 1}'] * row[f'F{(i % 25) + 1}']
        sin_term = np.sin(row[f'F{i + 2}'])
        result +=  row[f'F{i + 3}'] ** 2 + row[f'F{i + 4}'] ** 3 + log_term + product_term + sin_term
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
