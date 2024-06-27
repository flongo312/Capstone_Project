import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the list of securities organized by type, including major indices and relevant cryptocurrencies
securities = {
    'Stocks': [
        'AAPL', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META', 'AMD', 'BAC', 
        'MSFT', 'NFLX', 'JPM', 'V', 'DIS', 'CSCO', 
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

# Data Exploration and Visualization
# Set up the output directory for figures
figure_directory = '/Users/frank/Desktop/Project/Figures'
os.makedirs(figure_directory, exist_ok=True)

# Filter out the 'Indices' type
filtered_data = combined_data[combined_data['Type'] != 'Indices']

# Histogram: Count of Securities by Type in the dataset
plt.figure(figsize=(10, 6))
sns.countplot(x='Type', data=filtered_data.drop_duplicates(subset=['Ticker']), palette='viridis')
plt.title('Count of Securities by Type')
plt.xlabel('Security Type')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)
for p in plt.gca().patches:
    plt.gca().annotate(f'\n{p.get_height()}', (p.get_x() + 0.3, p.get_height()), ha='center', va='top', color='white', size=10)
plt.savefig(os.path.join(figure_directory, 'histogram_security_count.png'))
plt.close()

# Cumulative Returns by Security Type
pivot_data = combined_data.pivot(index='Date', columns='Ticker', values='Adj Close')
returns_data = pivot_data.pct_change().dropna()

cumulative_returns = (1 + returns_data).cumprod()

# Equal-weighted portfolio for each type
combined_data['Daily Return'] = combined_data.groupby('Ticker')['Adj Close'].pct_change()
cumulative_returns_by_type = combined_data.groupby(['Date', 'Type'])['Daily Return'].mean().unstack().fillna(0)
cumulative_returns_by_type = (1 + cumulative_returns_by_type).cumprod()

plt.figure(figsize=(14, 8))
for column in cumulative_returns_by_type.columns:
    plt.plot(cumulative_returns_by_type.index, cumulative_returns_by_type[column], label=column)
plt.title('Cumulative Returns by Security Type')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(figure_directory, 'cumulative_returns_by_type.png'))
plt.close()

print(f"Visualizations saved in {figure_directory}")
