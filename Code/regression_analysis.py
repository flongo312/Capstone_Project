import pandas as pd
import statsmodels.api as sm

# Load the FRED data
fred_data = pd.read_csv('/Users/frank/Desktop/Project/Data/fred_data.csv')
# Load the Yahoo Finance data
yfinance_data = pd.read_csv('/Users/frank/Desktop/Project/Data/yfinance_data.csv')

# Convert date columns to datetime
fred_data['Date'] = pd.to_datetime(fred_data['Unnamed: 0'])
yfinance_data['Date'] = pd.to_datetime(yfinance_data['Unnamed: 0'])

# Merge the datasets on the Date column
merged_data = pd.merge(fred_data, yfinance_data, on='Date', how='inner')

# Select relevant columns for regression
X = merged_data[['Consumer Price Index (CPI)', 'Producer Price Index (PPI)', 'Unemployment Rate']]
y = merged_data['Close']

# Add constant to predictor variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# Save the summary to a text file
with open('/Users/frank/Desktop/Project/Figures/regression_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
