import yfinance as yf
import pandas as pd

# Define the list of securities organized by type
securities = {
    'Stocks': ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META','AMD', 'BAC',],
    'ETFs': ['SPY', 'QQQ', 'IWM', 'GLD', 'EEM', 'VOO', 'XLF', 'TQQQ', 'DIA', 'LQD'],
    'Mutual Funds': ['VFIAX', 'VTSAX', 'FXAIX', 'FCNTX', 'AGTHX', 'TRBCX', 'VBTLX'],
    'Cryptocurrencies': ['BTC-USD', 'LTC-USD']
}

# Function to fetch the earliest available date for each ticker
def get_earliest_date(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="max")
    return hist.index.min()

# Get the earliest common start date
earliest_dates = []
for category, tickers in securities.items():
    for ticker in tickers:
        earliest_dates.append(get_earliest_date(ticker))

common_start_date = max(earliest_dates).tz_localize(None)  # Remove timezone information

print(f"Earliest common start date: {common_start_date}")

# Function to fetch data starting from the earliest common date
def fetch_data(ticker, start_date, security_type):
    data = yf.download(ticker, start=start_date)
    data['Ticker'] = ticker
    data['Type'] = security_type
    return data

# Fetch data for all tickers from the common start date
data_frames = []
for category, tickers in securities.items():
    for ticker in tickers:
        data_frames.append(fetch_data(ticker, common_start_date, category))

combined_data = pd.concat(data_frames)

# Drop rows with missing values
combined_data = combined_data.dropna()

# Ensure all tickers have data aligned by common dates
combined_data.reset_index(inplace=True)
combined_data.set_index(['Date', 'Ticker'], inplace=True)

common_dates = pd.date_range(start=common_start_date, end=combined_data.index.get_level_values('Date').max())
combined_data = combined_data.unstack(level='Ticker').reindex(common_dates).stack(level='Ticker').sort_index()

# Save to CSV
output_file = "/Users/frank/Desktop/Project/Data/yfinance_data.csv"
combined_data.to_csv(output_file)

print(f"Data saved to {output_file}")

# Summary of selected tickers
selected_tickers = combined_data.index.get_level_values('Ticker').unique()
print("Selected tickers for the common date range:")
for ticker in selected_tickers:
    print(ticker)