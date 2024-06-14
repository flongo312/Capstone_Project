import pandas as pd
import matplotlib.pyplot as plt

# Load the Yahoo Finance data
yfinance_data = pd.read_csv('/Users/frank/Desktop/Project/Data/yfinance_data.csv')

# Convert date column to datetime
yfinance_data['Date'] = pd.to_datetime(yfinance_data['Unnamed: 0'])

# Calculate rolling volatility
yfinance_data['Rolling_Volatility'] = yfinance_data['Close'].rolling(window=30).std()

# Plot the rolling volatility
plt.figure(figsize=(14, 7))
plt.plot(yfinance_data['Date'], yfinance_data['Rolling_Volatility'])
plt.title('Rolling Volatility of Stock Returns')
plt.xlabel('Date')
plt.ylabel('Volatility')

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/rolling_volatility.png')
plt.close()

