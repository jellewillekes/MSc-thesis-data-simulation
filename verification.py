import pandas as pd
import numpy as np
from scipy import stats

dataset = pd.read_csv('synthetic_dataset.csv')

# Assuming 'dataset' is a pandas DataFrame with columns 'Y' and features
target_col = 'Y'
significance_level = 0.05

# Separate numerical and categorical columns
numerical_cols = dataset.select_dtypes(include=np.number).columns.tolist()
categorical_cols = dataset.select_dtypes(include='object').columns.tolist()

# Remove target column 'Y' from numerical columns
numerical_cols.remove(target_col)

# Perform ANOVA for numerical features
print("Numerical features with no heterogeneity:")
for feature in numerical_cols:
    # Discretize the continuous feature (you can choose the number of bins based on your domain knowledge)
    binned_feature = pd.cut(dataset[feature], bins=3, labels=['low', 'medium', 'high'])

    # Group the target variable 'Y' by the binned feature levels
    groups = dataset.groupby(binned_feature)[target_col].apply(list)

    # Perform one-way ANOVA test
    f_stat, p_value = stats.f_oneway(*groups)

    if p_value >= significance_level:
        print(f"{feature}: F-statistic = {f_stat}, p-value = {p_value}")

# Perform Chi-square test for categorical features
print("\nCategorical features with no heterogeneity:")
for feature in categorical_cols:
    # Create a contingency table for the categorical feature and target variable 'Y'
    contingency_table = pd.crosstab(dataset[feature], dataset[target_col])

    # Perform Chi-square test of independence
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    if p_value >= significance_level:
        print(f"{feature}: Chi2-statistic = {chi2_stat}, p-value = {p_value}")






