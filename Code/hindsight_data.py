import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define the list of securities organized by type, excluding those that IPO'd after 2009
securities = {
    'Stocks': [
        'AAPL', 'AMZN', 'GOOGL', 'NVDA', 'BAC', 'MSFT', 'JPM', 'DIS', 
        'CSCO', 'XOM', 'PFE', 'KO', 'PEP', 'INTC', 'T', 'MRK', 'WMT', 
        'NKE', 'GE', 'MCD', 'IBM', 'WFC', 'UNH', 'ORCL', 'QCOM', 'TM', 
        'TXN', 'HON', 'MO', 'MDT', 'SPG', 'AXP', 'GS', 'LMT', 'BMY', 'BLK'
    ],
    'ETFs': [
        'SPY', 'QQQ', 'IWM', 'GLD', 'EEM', 'XLF', 'DIA', 'LQD', 
        'IVV', 'XLK', 'VNQ', 'HYG', 'XLV', 'XLE', 'XLY', 
        'XLP', 'XLU', 'GDX', 'XLI', 'IWF', 'IWB', 'IJH', 'IYR', 
        'IJR', 'IWN', 'EFA', 'VWO', 'SHY', 'USO', 'TLT', 'VGK', 
        'XOP', 'EWT', 'EWJ', 'VB'
    ],
    'Mutual Funds': [
        'VFIAX', 'VTSAX', 'FXAIX', 'FCNTX', 'AGTHX', 'TRBCX', 
        'VWELX', 'DODGX', 'PRGFX', 'VWIAX', 
        'VEIPX', 'VWINX', 'PTTRX', 'POAGX', 'RPMGX', 'VWUAX', 
        'FAGIX', 'VGHAX', 'FFIDX', 'FBGRX', 'FSPTX', 'FDGRX', 
        'VWNDX', 'JATTX', 'SGENX', 'RWMFX', 'PRHSX', 'FLPSX', 
        'FBALX', 'FAIRX', 'VGSLX', 'VDIGX', 'MGVAX', 'PRWCX', 
        'PARNX', 'PRNHX', 'RYVPX', 'SWHGX'
    ],
    'Indices': [
        '^GSPC',  # S&P 500
        '^DJI',   # Dow Jones Industrial Average
        '^IXIC'   # NASDAQ Composite
    ]
}

# Define the start and end dates
start_date = '2011-05-04'
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

# Function to fetch data starting from the start date up to the end date
def fetch_data(ticker, start_date, end_date, security_type):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker
    data['Type'] = security_type
    return data

# Fetch data for all tickers from the start date up to the end date
data_frames = []
for category, tickers in tqdm(securities.items(), desc='Categories'):
    for ticker in tqdm(tickers, desc=f'Tickers in {category}', leave=False):
        data = fetch_data(ticker, start_date, end_date, category)
        data_frames.append(data)

# Combine all data into a single DataFrame
combined_data = pd.concat(data_frames)

# Ensure all tickers have data aligned by common dates
combined_data.reset_index(inplace=True)
combined_data.set_index(['Date', 'Ticker'], inplace=True)

# Create a complete date range from the start date to the end date
common_dates = pd.date_range(start=start_date, end=end_date)
combined_data = combined_data.unstack(level='Ticker').reindex(common_dates).stack(level='Ticker').sort_index()

# Ensure the Date column is labeled correctly
combined_data.reset_index(inplace=True)
combined_data.rename(columns={'level_0': 'Date'}, inplace=True)

# Save to CSV
output_file = "/Users/frank/Desktop/Project/Data/hindsight_data.csv"
combined_data.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")
