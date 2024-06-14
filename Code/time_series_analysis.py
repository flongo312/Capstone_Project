import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the Yahoo Finance data
yfinance_data = pd.read_csv('/Users/frank/Desktop/Project/Data/yfinance_data.csv')

# Convert date column to datetime
yfinance_data['Date'] = pd.to_datetime(yfinance_data['Unnamed: 0'])

# Set date as index
yfinance_data.set_index('Date', inplace=True)

# Perform seasonal decomposition
result = seasonal_decompose(yfinance_data['Close'], model='multiplicative', period=12)

# Plot the decomposition results
result.plot()
plt.suptitle('Seasonal Decomposition of Stock Prices')

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/time_series_decomposition.png')
plt.close()
