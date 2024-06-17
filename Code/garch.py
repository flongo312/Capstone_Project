import pandas as pd
import matplotlib.pyplot as plt

#GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model

# Load the data
fred_data = pd.read_csv('/mnt/data/fred_data.csv')
yfinance_data = pd.read_csv('/mnt/data/yfinance_data.csv')

# Rename 'Unnamed: 0' to 'Date'
fred_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
yfinance_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Merge the data on 'Date'
data = pd.merge(fred_data, yfinance_data, on='Date')

# Calculate rolling volatility (standard deviation of stock returns over a rolling window)
data['Returns'] = data['Close'].pct_change()
data['Rolling Volatility'] = data['Returns'].rolling(window=30).std()

# Plot the rolling volatility
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Rolling Volatility'], color='purple')
plt.xlabel('Date')
plt.ylabel('Rolling Volatility')
plt.title('Volatility Analysis: Rolling Volatility of Stock Returns')
plt.show()
