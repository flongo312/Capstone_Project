import pandas as pd
import matplotlib.pyplot as plt

#ARIMA Model (AutoRegressive Integrated Moving Average)

# Load the data
fred_data = pd.read_csv('/mnt/data/fred_data.csv')
yfinance_data = pd.read_csv('/mnt/data/yfinance_data.csv')

# Rename 'Unnamed: 0' to 'Date'
fred_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
yfinance_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Merge the data on 'Date'
data = pd.merge(fred_data, yfinance_data, on='Date')

# Plot time series for CPI and stock market closing prices
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.set_xlabel('Date')
ax1.set_ylabel('Consumer Price Index (CPI)', color='tab:blue')
ax1.plot(data['Date'], data['Consumer Price Index (CPI)'], color='tab:blue', label='CPI')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Stock Market Closing Prices', color='tab:red')
ax2.plot(data['Date'], data['Close'], color='tab:red', label='Close')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Time Series Analysis: CPI and Stock Market Closing Prices')
plt.show()
