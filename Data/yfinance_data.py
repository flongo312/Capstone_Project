import yfinance as yf
import pandas as pd
import numpy as np

# Define the list of securities
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'IBM', 'GE', 'TSLA', 'NVDA', 'NFLX',
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'VTI', 'IWM', 'EFA', 'EEM', 'GLD',
    'VFIAX', 'SWPPX', 'FXAIX', 'VWELX', 'VFINX', 'AGTHX', 'AEPGX', 'ANWPX', 'FCNTX', 'DODGX',
    'BND', 'AGG', 'LQD', 'TLT', 'IEF', 'SHY', 'TIP', 'HYG', 'JNK', 'MUB',
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'BNB-USD', 'DOT-USD', 'LINK-USD', 'DOGE-USD'
]

# Fetch data
def fetch_data(ticker):
    data = yf.download(ticker, start="2015-01-01")
    data['Ticker'] = ticker
    return data

# Fetch data for all tickers
data_frames = [fetch_data(ticker) for ticker in tickers]
combined_data = pd.concat(data_frames)

# Drop rows with missing values
combined_data = combined_data.dropna()

# Reset index
combined_data.reset_index(inplace=True)
combined_data.set_index(['Date', 'Ticker'], inplace=True)

# Calculate additional metrics
def calculate_metrics(df):
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod() - 1
    df['Volatility'] = df['Daily Return'].rolling(window=252).std() * np.sqrt(252)
    df['Sharpe Ratio'] = (df['Daily Return'].mean() / df['Volatility']) * np.sqrt(252)
    df['Max Drawdown'] = df['Adj Close'].rolling(window=252, min_periods=1).apply(lambda x: ((pd.Series(x) / pd.Series(x).cummax()) - 1).min(), raw=True)
    df['Beta'] = df['Daily Return'].rolling(window=252).cov(df['Daily Return']) / df['Daily Return'].rolling(window=252).var()
    df['Alpha'] = df['Adj Close'] - df['Adj Close'].rolling(window=252).mean()
    return df

# Apply metrics calculation
combined_data = combined_data.groupby('Ticker').apply(calculate_metrics)

# Reset index to ensure date-based indexing
combined_data.reset_index(inplace=True)
combined_data.set_index('Date', inplace=True)

# Sort the data for better readability
combined_data.sort_index(inplace=True)

# Save to CSV
output_file = "/Users/frank/Desktop/Project/Data/yfinance_data.csv"
combined_data.to_csv(output_file)

print(f"Data saved to {output_file}")

# Summary of selected tickers
selected_tickers = combined_data['Ticker'].unique()
print("Selected tickers for the common date range:")
for ticker in selected_tickers:
    print(ticker)
