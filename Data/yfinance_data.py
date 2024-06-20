import yfinance as yf
import pandas as pd

# Define the list of securities organized by type, including major indices and relevant cryptocurrencies
securities = {
    'Stocks': [
        'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'AMD', 'BAC', 
        'MSFT', 'NFLX', 'BABA', 'JPM', 'V', 'PYPL', 'DIS', 'CSCO', 
        'XOM', 'PFE', 'KO', 'PEP', 'INTC', 'CRM', 'T', 'MRK', 'WMT', 
        'NKE', 'GE', 'COST', 'MCD', 'SBUX', 'IBM', 'WFC', 'UNH', 'ORCL', 
        'QCOM', 'AVGO', 'TM', 'ADBE', 'TXN', 'HON', 'MO', 'MDT', 'SPG', 
        'AXP', 'GS', 'LMT', 'BMY', 'BLK'
    ],
    'ETFs': [
        'SPY', 'QQQ', 'IWM', 'GLD', 'EEM', 'VOO', 'XLF', 'TQQQ', 'DIA', 
        'LQD', 'IVV', 'VTI', 'XLK', 'VNQ', 'HYG', 'XLV', 'XLE', 'XLY', 
        'XLP', 'XLU', 'GDX', 'XLI', 'IWF', 'IWB', 'IJH', 'IYR', 'VUG', 
        'IJR', 'IWN', 'EFA', 'VWO', 'SHY', 'USO', 'TLT', 'VGK', 'XOP', 
        'IEFA', 'IEMG', 'PFF', 'EWT', 'EWJ', 'IWC', 'VB', 'SCHX', 'VIG', 
        'SCHF', 'EZU', 'IGV', 'VCSH', 'SPDW'
    ],
    'Mutual Funds': [
        'VFIAX', 'VTSAX', 'FXAIX', 'FCNTX', 'AGTHX', 'TRBCX', 'VBTLX', 
        'SWPPX', 'VWELX', 'DODGX', 'VIGAX', 'PRGFX', 'VWIAX', 'DFELX', 
        'VIMAX', 'VPCCX', 'VEIPX', 'VWINX', 'VTIAX', 'PTTRX', 'POAGX', 
        'RPMGX', 'VWUAX', 'FAGIX', 'VGHAX', 'FFIDX', 'FBGRX', 'FSPTX', 
        'FDGRX', 'VWNDX', 'JATTX', 'SGENX', 'RWMFX', 'PRHSX', 'FLPSX', 
        'FBALX', 'FAIRX', 'VGSLX', 'VDIGX', 'MGVAX', 'PRWCX', 'PARNX', 
        'PRNHX', 'RYVPX', 'SWHGX', 'VHCAX', 'VSEQX'
    ],
    'Indices': [
        '^GSPC',  # S&P 500
        '^DJI',   # Dow Jones Industrial Average
        '^IXIC'   # NASDAQ Composite
    ],
    'Cryptocurrencies': [
        'BTC-USD',  # Bitcoin
        'ETH-USD',  # Ethereum
        'XRP-USD',  # Ripple
        'LTC-USD',  # Litecoin
        'XMR-USD'   # Monero
    ]
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
        data = fetch_data(ticker, common_start_date, category)
        data_frames.append(data)

# Combine all data into a single DataFrame
combined_data = pd.concat(data_frames)

# Ensure all tickers have data aligned by common dates
combined_data.reset_index(inplace=True)
combined_data.set_index(['Date', 'Ticker'], inplace=True)

# Create a complete date range from the common start date to the latest date in the data
common_dates = pd.date_range(start=common_start_date, end=combined_data.index.get_level_values('Date').max())
combined_data = combined_data.unstack(level='Ticker').reindex(common_dates).stack(level='Ticker').sort_index()

# Ensure the Date column is labeled correctly
combined_data.reset_index(inplace=True)
combined_data.rename(columns={'level_0': 'Date'}, inplace=True)

# Save to CSV
output_file = "/Users/frank/Desktop/Project/Data/yfinance_data.csv"
combined_data.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")
