import pandas as pd
import seaborn as sns
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

# Calculate correlation matrix
corr_matrix = merged_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Economic Indicators and Stock Market Performance')

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/correlation_matrix.png')
plt.close()
