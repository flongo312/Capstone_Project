import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "yfinance", "pandas", "numpy", "matplotlib", "seaborn", "tqdm"
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the list of securities organized by type, excluding those with data starting after 2010
securities = {
    'Stocks': [
        'AAPL', 'AMZN', 'GOOGL', 'NVDA', 'AMD', 'BAC', 
        'MSFT', 'NFLX', 'JPM', 'DIS', 'CSCO', 
        'XOM', 'PFE', 'KO', 'PEP', 'INTC', 'CRM', 'T', 'MRK', 'WMT', 
        'NKE', 'GE', 'COST', 'MCD', 'SBUX', 'IBM', 'WFC', 'UNH', 'ORCL', 
        'QCOM', 'AVGO', 'TM', 'ADBE', 'TXN', 'HON', 'MO', 'MDT', 'SPG', 
        'AXP', 'GS', 'LMT', 'BMY', 'BLK', 'CAT', 'BA', 'ABT', 
        'JCI', 'F', 'DE', 'LLY', 'CVX', 'PG', 'SO', 'D', 'DUK', 
        'NEE', 'CL', 'JNJ', 'EMR', 'KMB', 'USB', 'FDX', 'GD', 'LHX',
        'BK', 'STT', 'SCHW', 'TROW', 'AMP', 'BEN', 'TGT', 
        'BBY', 'DG', 'ROST', 'TJX', 'M', 'JWN', 'BKNG', 'MAR', 
        'DAL', 'UAL', 'AAL', 'ALK', 
        'PENN', 'MGM', 'WYNN', 'LVS', 'BYD', 'MLCO', 
        'CCL', 'RCL', 'AIG', 'TRV', 'ALL', 'CB', 'MET', 'PRU', 'LNC', 
        'UNM', 'AFL', 'PFG', 'PNW', 'ETR', 
        'PPL', 'NRG', 'AES', 'XEL', 'WEC', 'CNP', 'CMS', 'NI', 'DTE', 
        'EVRG', 'ATO', 'SRE', 'PCG', 'AEE', 'LNT', 'AEP', 'ED'
    ],
    'ETFs': [
        'SPY', 'QQQ', 'IWM', 'GLD', 'EEM', 'XLF', 'DIA', 
        'IVV', 'XLK', 'XLV', 'XLE', 'XLY', 
        'XLU', 'XLI', 'IWF', 'IWB', 'IYR', 'VUG', 
        'IJR', 'IWN', 'SHY', 'TLT', 'XOP', 
        'PFF', 'EWT', 'EWJ', 'IWC', 'VB', 
        'EZU', 'SPDW', 'VTI', 'VEU', 'BND', 'VWO', 
        'VNQ', 'VIG', 'VYM', 'IJH', 'IVW', 'EFG'
    ],
    'Mutual Funds': [
        'VEIPX', 'VWUAX', 'FAGIX', 'FFIDX', 'FBGRX', 'FDGRX', 
        'JATTX', 'RWMFX', 'PRHSX', 'VGSLX', 'VDIGX', 'MGVAX', 
        'PARNX', 'PRNHX', 'VHCAX', 'AEPGX', 'NEWFX'
    ],
    'Indices': [
        '^GSPC'  # S&P 500
    ]
}

# Define the end date
end_date = '2014-11-10'

# Function to fetch the earliest available date for each ticker
def get_earliest_date(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="max")
    return hist.index.min()

# Get the earliest common start date and print each ticker's earliest date
earliest_dates = []
for category, tickers in tqdm(securities.items(), desc='Categories'):
    for ticker in tqdm(tickers, desc=f'Tickers in {category}', leave=False):
        date = get_earliest_date(ticker)
        if pd.isna(date):
            print(f"{ticker}: No data available")
        else:
            print(f"{ticker}: {date}")  # Print the earliest date for each ticker
            earliest_dates.append(date)

# Filter out NaN values from earliest_dates
earliest_dates = [date for date in earliest_dates if pd.notna(date)]
common_start_date = max(earliest_dates).tz_localize(None)  # Remove timezone information

print(f"Earliest common start date: {common_start_date}")

# Function to fetch data starting from the earliest common date up to a given end date
def fetch_data(ticker, start_date, end_date, security_type):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker
    data['Type'] = security_type
    return data

# Fetch data for all tickers from the common start date up to the end date
data_frames = []
for category, tickers in tqdm(securities.items(), desc='Categories'):
    for ticker in tqdm(tickers, desc=f'Tickers in {category}', leave=False):
        data = fetch_data(ticker, common_start_date, end_date, category)
        data_frames.append(data)

# Combine all data into a single DataFrame
combined_data = pd.concat(data_frames)

# Ensure all tickers have data aligned by common dates
combined_data.reset_index(inplace=True)
combined_data.set_index(['Date', 'Ticker'], inplace=True)

# Create a complete date range from the common start date to the end date
common_dates = pd.date_range(start=common_start_date, end=end_date)
combined_data = combined_data.unstack(level='Ticker').reindex(common_dates).stack(level='Ticker').sort_index()

# Ensure the Date column is labeled correctly
combined_data.reset_index(inplace=True)
combined_data.rename(columns={'level_0': 'Date'}, inplace=True)

# Ensure the Data directory exists
data_directory = os.path.join(script_dir, '../Data')
os.makedirs(data_directory, exist_ok=True)

# Save to CSV
output_file = os.path.join(data_directory, "yfinance_data.csv")
combined_data.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")

# Data Exploration and Visualization

# Set up the output directory for figures
figure_directory = os.path.join(script_dir, '../Figures')
os.makedirs(figure_directory, exist_ok=True)

# Filter out the 'Indices' type
filtered_data = combined_data[combined_data['Type'] != 'Indices']

# Use a colorblind-friendly palette
colorblind_palette = sns.color_palette("colorblind")

# Set a consistent style
sns.set(style="whitegrid", palette=colorblind_palette, font_scale=1.4)

# Histogram: Count of Securities by Type in the dataset
plt.figure(figsize=(14, 8))
ax = sns.countplot(x='Type', data=filtered_data.drop_duplicates(subset=['Ticker']), palette=colorblind_palette)
plt.title('Distribution of Securities by Type', fontsize=24, weight='bold')
plt.xlabel('Type of Security', fontsize=20, weight='bold')
plt.ylabel('Number of Securities', fontsize=20, weight='bold')
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', color='black', fontsize=22, weight='bold')
plt.tight_layout()
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

plt.figure(figsize=(16, 9))
for column in cumulative_returns_by_type.columns:
    plt.plot(cumulative_returns_by_type.index, cumulative_returns_by_type[column] * 100 - 100, label=column, linewidth=2.5)
plt.title('Cumulative Returns Over Time by Security Type', fontsize=24, weight='bold')
plt.xlabel('Date', fontsize=20, weight='bold')
plt.ylabel('Cumulative Return (%)', fontsize=20, weight='bold')
plt.legend(title='Security Type', fontsize=16, title_fontsize=18, loc='upper left')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figure_directory, 'cumulative_returns_by_type.png'))
plt.close()

print(f"Visualizations saved in {figure_directory}")
