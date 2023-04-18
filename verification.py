import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('synthetic_dataset.csv')

# Assuming 'df' is your dataset with both numerical and categorical features
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Heterogeneity: Check the distribution of the target variable
sns.histplot(df['Y'], kde=True)
plt.title('Distribution of Target Variable Y')
plt.show()

df_trans = df.copy()
# Convert categorical features to numerical values using ordinal encoding
ordinal_encoder = OrdinalEncoder()
category_columns = [f'F{i + 25}' for i in range(1, 26)]
df_trans[category_columns] = ordinal_encoder.fit_transform(df_trans[category_columns])

# Balanced Classes: Check the distribution of each categorical variable

# Low Multicollinearity: Check the correlation between numerical features
corr_matrix = numerical_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, fmt='.2f', mask=mask, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Another way to check multicollinearity is to calculate the Variance Inflation Factor (VIF)
vif = pd.DataFrame()
vif["Feature"] = numerical_df.columns
vif["VIF"] = [variance_inflation_factor(numerical_df.values, i) for i in range(numerical_df.shape[1])]
print("Variance Inflation Factors (VIF):")
print(vif)

# Minimal Missing Data: Check the percentage of missing data in each column
missing_data = df.isnull().mean() * 100
print("Percentage of Missing Data per Column:")
print(missing_data)

# Absence of Endogeneity: PCA can be used to verify if there is a hidden structure in the data
scaler = StandardScaler()
scaled_numerical_df = scaler.fit_transform(numerical_df)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_numerical_df)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
sns.scatterplot(data=principal_df, x='PC1', y='PC2')
plt.title('PCA Scatterplot')
plt.show()

# Randomness: Visualize the distribution of each numerical feature
for col in numerical_df.columns:
    sns.histplot(numerical_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()