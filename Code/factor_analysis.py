import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the FRED data
fred_data = pd.read_csv('/Users/frank/Desktop/Project/Data/fred_data.csv')
# Load the Yahoo Finance data
yfinance_data = pd.read_csv('/Users/frank/Desktop/Project/Data/yfinance_data.csv')

# Convert date columns to datetime
fred_data['Date'] = pd.to_datetime(fred_data['Unnamed: 0'])
yfinance_data['Date'] = pd.to_datetime(yfinance_data['Unnamed: 0'])

# Merge the datasets on the Date column
merged_data = pd.merge(fred_data, yfinance_data, on='Date', how='inner')

# Select relevant columns for factor analysis
factor_data = merged_data[['Consumer Price Index (CPI)', 'Producer Price Index (PPI)', 'Unemployment Rate', 'Close']]

# Standardize the data
factor_data_standardized = (factor_data - factor_data.mean()) / factor_data.std()

# Perform PCA
pca = PCA(n_components=3)
pca.fit(factor_data_standardized)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, 4), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 4), pca.explained_variance_ratio_.cumsum(), where='mid', label='cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.title('PCA Explained Variance')

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/pca_explained_variance.png')
plt.close()

