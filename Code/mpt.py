import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "pandas", "numpy", "matplotlib", "squarify", "scipy"
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
from scipy.optimize import minimize
import os
from textwrap import fill

# Comprehensive dictionary for ticker names with shortened names for specific securities
ticker_info = {
    'AAPL': 'Apple', 'AMZN': 'Amazon', 'GOOGL': 'Google', 'NVDA': 'NVIDIA', 'AMD': 'AMD', 'BAC': 'Bank of America', 
    'MSFT': 'Microsoft', 'NFLX': 'Netflix', 'JPM': 'JPMorgan', 'DIS': 'Disney', 'CSCO': 'Cisco', 
    'XOM': 'Exxon Mobil', 'PFE': 'Pfizer', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'INTC': 'Intel', 'CRM': 'Salesforce', 
    'T': 'AT&T', 'MRK': 'Merck', 'WMT': 'Walmart', 'NKE': 'Nike', 'GE': 'GE', 'COST': 'Costco', 'MCD': 'McDonald\'s', 
    'SBUX': 'Starbucks', 'IBM': 'IBM', 'WFC': 'Wells Fargo', 'UNH': 'UnitedHealth', 'ORCL': 'Oracle', 
    'QCOM': 'Qualcomm', 'AVGO': 'Broadcom', 'TM': 'Toyota', 'ADBE': 'Adobe', 'TXN': 'Texas Instruments', 
    'HON': 'Honeywell', 'MO': 'Altria', 'MDT': 'Medtronic', 'SPG': 'Simon Property Group', 'AXP': 'American Express', 
    'GS': 'Goldman Sachs', 'LMT': 'Lockheed Martin', 'BMY': 'Bristol-Myers Squibb', 'BLK': 'BlackRock', 
    'CAT': 'Caterpillar', 'BA': 'Boeing', 'ABT': 'Abbott Laboratories', 'JCI': 'Johnson Controls', 'F': 'Ford', 
    'DE': 'Deere & Company', 'LLY': 'Eli Lilly', 'CVX': 'Chevron', 'PG': 'Procter & Gamble', 'SO': 'Southern Company', 
    'D': 'Dominion Energy', 'DUK': 'Duke Energy', 'NEE': 'NextEra Energy', 'CL': 'Colgate-Palmolive', 
    'JNJ': 'Johnson & Johnson', 'EMR': 'Emerson Electric', 'KMB': 'Kimberly-Clark', 'USB': 'U.S. Bancorp', 
    'FDX': 'FedEx', 'GD': 'General Dynamics', 'LHX': 'L3Harris Technologies', 'BK': 'Bank of New York Mellon', 
    'STT': 'State Street', 'SCHW': 'Charles Schwab', 'TROW': 'T. Rowe Price', 'AMP': 'Ameriprise Financial', 
    'BEN': 'Franklin Resources', 'TGT': 'Target', 'BBY': 'Best Buy', 'DG': 'Dollar General', 'ROST': 'Ross Stores', 
    'TJX': 'TJX Companies', 'M': 'Macy\'s', 'JWN': 'Nordstrom', 'BKNG': 'Booking Holdings', 'MAR': 'Marriott International', 
    'DAL': 'Delta Air Lines', 'UAL': 'United Airlines', 'AAL': 'American Airlines', 'ALK': 'Alaska Air', 
    'PENN': 'Penn National Gaming', 'MGM': 'MGM Resorts', 'WYNN': 'Wynn Resorts', 'LVS': 'Las Vegas Sands', 
    'BYD': 'Boyd Gaming', 'MLCO': 'Melco Resorts & Entertainment', 'CCL': 'Carnival', 'RCL': 'Royal Caribbean', 
    'AIG': 'American International Group', 'TRV': 'Travelers', 'ALL': 'Allstate', 'CB': 'Chubb', 'MET': 'MetLife', 
    'PRU': 'Prudential', 'LNC': 'Lincoln National', 'UNM': 'Unum', 'AFL': 'Aflac', 'PFG': 'Principal Financial Group', 
    'PNW': 'Pinnacle West', 'ETR': 'Entergy', 'PPL': 'PPL Corporation', 'NRG': 'NRG Energy', 'AES': 'AES Corporation', 
    'XEL': 'Xcel Energy', 'WEC': 'WEC Energy Group', 'CNP': 'CenterPoint Energy', 'CMS': 'CMS Energy', 
    'NI': 'NiSource', 'DTE': 'DTE Energy', 'EVRG': 'Evergy', 'ATO': 'Atmos Energy', 'SRE': 'Sempra Energy', 
    'PCG': 'PG&E', 'AEE': 'Ameren', 'LNT': 'Alliant Energy', 'AEP': 'American Electric Power', 'ED': 'Consolidated Edison',
    'SPY': 'SPDR S&P 500 ETF', 'QQQ': 'QQQ ETF', 'IWM': 'Russell 2000 ETF', 'GLD': 'Gold ETF', 
    'EEM': 'Emerging Markets ETF', 'XLF': 'Financials ETF', 'DIA': 'Dow Jones ETF', 'IVV': 'iShares S&P 500', 
    'XLK': 'Tech Sector SPDR ETF', 'XLV': 'Health Care XLV ETF', 'XLE': 'Energy ETF', 'XLY': 'Consumer Discretionary ETF', 
    'XLU': 'Utilities ETF', 'XLI': 'Industrial ETF', 'IWF': 'Russell 1000 Growth ETF', 'IWB': 'Russell 1000 ETF', 
    'IYR': 'U.S. Real Estate ETF', 'VUG': 'Vanguard Growth ETF', 'IJR': 'Small-Cap ETF', 'IWN': 'Russell 2000 Value ETF', 
    'SHY': '1-3 Year Treasury Bond ETF', 'TLT': '20+ Year Treasury Bond ETF', 'XOP': 'Oil & Gas ETF', 'PFF': 'Preferred and Income Securities ETF', 
    'EWT': 'Taiwan ETF', 'EWJ': 'Japan ETF', 'IWC': 'Micro-Cap ETF', 'VB': 'Small-Cap ETF', 'EZU': 'Eurozone ETF', 
    'SPDW': 'Developed World ex-US ETF', 'VTI': 'Vanguard Total Stock Market ETF', 'VEU': 'FTSE All-World ex-US ETF', 
    'BND': 'Vanguard Total Bond Market ETF', 'VWO': 'FTSE Emerging Markets ETF', 'VNQ': 'Vanguard Real Estate ETF', 
    'VIG': 'Dividend Appreciation ETF', 'VYM': 'Vanguard High Dividend Yield ETF', 'IJH': 'iShares Core S&P Mid-Cap ETF', 
    'IVW': 'iShares S&P 500 Growth ETF', 'EFG': 'iShares MSCI EAFE Growth ETF', 'VEIPX': 'Vanguard Equity-Income Fund', 
    'VWUAX': 'Vanguard U.S. Growth Fund', 'FAGIX': 'Fidelity Capital & Income Fund', 'FFIDX': 'Fidelity Fund', 
    'FBGRX': 'Fidelity Blue Chip Growth Fund', 'FDGRX': 'Fidelity Growth Company Fund', 'JATTX': 'Janus Henderson Triton Fund', 
    'SGENX': 'First Eagle Global Fund', 'RWMFX': 'Washington Mutual Investors Fund', 'PRHSX': 'T. Rowe Price Health Sciences Fund', 
    'VGSLX': 'Vanguard Real Estate Index Fund', 'VDIGX': 'Vanguard Dividend Growth Fund', 'MGVAX': 'MFS Value Fund', 
    'PARNX': 'Parnassus Mid Cap Fund', 'PRNHX': 'T. Rowe Price New Horizons Fund', 'VHCAX': 'Vanguard Health Care Fund', 
    'AEPGX': 'American Funds EuroPacific Growth Fund', 'NEWFX': 'American Funds New Economy Fund', '^GSPC': 'S&P 500'
}


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the top assets based on composite score from the saved CSV files
filtered_top_securities_5_years = pd.read_csv(os.path.join(script_dir, '../Data/top_assets_composite_score_5_years.csv'))
filtered_top_securities_7_5_years = pd.read_csv(os.path.join(script_dir, '../Data/top_assets_composite_score_7_5_years.csv'))
filtered_top_securities_10_years = pd.read_csv(os.path.join(script_dir, '../Data/top_assets_composite_score_10_years.csv'))

# Load the returns data
returns_data_path = os.path.join(script_dir, '../Data/yfinance_data.csv')
returns_data = pd.read_csv(returns_data_path)
returns_data['Date'] = pd.to_datetime(returns_data['Date'])
returns_data.set_index('Date', inplace=True)

# Use all available data within the specified range
data = returns_data[(returns_data.index >= '2011-05-04') & (returns_data.index <= '2014-11-10')]

# Pivot the returns data to get adjusted close prices for each ticker
historical_data = data.pivot(columns='Ticker', values='Adj Close')

# Calculate daily returns
historical_returns = historical_data.pct_change().dropna()

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

# Function to calculate negative Sharpe ratio (for minimization)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# Function to estimate expected returns using CAPM
def estimate_expected_returns(tickers, returns_data, market_returns, risk_free_rate=0.03):
    market_excess_return = market_returns.mean() * 252 - risk_free_rate
    expected_returns = {}
    for ticker in tickers:
        ticker_returns = returns_data[ticker]
        covariance = np.cov(ticker_returns, market_returns)[0, 1]
        beta = covariance / market_returns.var()
        expected_return = risk_free_rate + beta * market_excess_return
        expected_returns[ticker] = expected_return
    return pd.Series(expected_returns)

# Function to perform portfolio optimization
def optimize_portfolio(filtered_data, returns_data, market_returns, risk_free_rate=0.03, min_securities=10):
    tickers = filtered_data['Ticker']
    
    # Verify tickers are in returns data
    missing_tickers = [ticker for ticker in tickers if ticker not in returns_data.columns]
    if missing_tickers:
        print(f"Warning: The following tickers are missing in the returns data and will be excluded: {missing_tickers}")
        tickers = tickers[tickers.isin(returns_data.columns)]
    
    if tickers.empty:
        print("Error: No tickers available for optimization after excluding missing tickers.")
        return pd.DataFrame(columns=['Ticker', 'Weight'])
    
    # Select the returns for the available tickers
    returns = returns_data[tickers]

    # Estimate expected returns using CAPM
    mean_returns = estimate_expected_returns(tickers, returns, market_returns, risk_free_rate)
    cov_matrix = returns.cov()

    num_assets = len(tickers)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1

    # Add constraints to ensure a minimum number of securities are included
    constraints += [{'type': 'ineq', 'fun': lambda x: x[i]} for i in range(num_assets)]
    constraints += [{'type': 'ineq', 'fun': lambda x: min_securities - np.sum(x > 1e-5)}]  # Minimum number of securities

    bound = (0.0, 1.0)  # No upper limit constraint on weights
    bounds = tuple(bound for _ in range(num_assets))

    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return pd.DataFrame({
        'Ticker': tickers,
        'Weight': result.x
    })

# Use historical market returns
market_returns = historical_returns['^GSPC']

# Optimize portfolios using all available data
optimal_weights_5_years = optimize_portfolio(filtered_top_securities_5_years, historical_returns, market_returns)
optimal_weights_7_5_years = optimize_portfolio(filtered_top_securities_7_5_years, historical_returns, market_returns)
optimal_weights_10_years = optimize_portfolio(filtered_top_securities_10_years, historical_returns, market_returns)

# Save the optimal weights to CSV files
optimal_weights_path_5_years = os.path.join(script_dir, '../Data/optimal_weights_5_years.csv')
optimal_weights_path_7_5_years = os.path.join(script_dir, '../Data/optimal_weights_7_5_years.csv')
optimal_weights_path_10_years = os.path.join(script_dir, '../Data/optimal_weights_10_years.csv')

if not optimal_weights_5_years.empty:
    optimal_weights_5_years.to_csv(optimal_weights_path_5_years, index=False)
else:
    print(f"Skipping saving for 5 years as no optimal weights were calculated.")

if not optimal_weights_7_5_years.empty:
    optimal_weights_7_5_years.to_csv(optimal_weights_path_7_5_years, index=False)
else:
    print(f"Skipping saving for 7.5 years as no optimal weights were calculated.")

if not optimal_weights_10_years.empty:
    optimal_weights_10_years.to_csv(optimal_weights_path_10_years, index=False)
else:
    print(f"Skipping saving for 10 years as no optimal weights were calculated.")

# Function to merge and display the combined data
def display_combined_data(filtered_data, optimal_weights, title):
    if optimal_weights.empty:
        print(f"No optimal weights to display for {title}")
        return pd.DataFrame()
    
    combined_data = filtered_data.merge(optimal_weights, on='Ticker')
    
    # Add the Name information from the dictionary
    combined_data['Name'] = combined_data['Ticker'].map(lambda x: ticker_info.get(x, 'Unknown'))
    
    combined_data_sorted = combined_data.sort_values(by='Weight', ascending=False)
    combined_data_sorted['Weight'] = combined_data_sorted['Weight'].round(4)
    
    print(title)
    print(combined_data_sorted)
    
    return combined_data_sorted

# 5-Year Horizon
combined_5_years = display_combined_data(filtered_top_securities_5_years, optimal_weights_5_years, '5-Year Horizon')

# 7.5-Year Horizon
combined_7_5_years = display_combined_data(filtered_top_securities_7_5_years, optimal_weights_7_5_years, '7.5-Year Horizon')

# 10-Year Horizon
combined_10_years = display_combined_data(filtered_top_securities_10_years, optimal_weights_10_years, '10-Year Horizon')

# Improved plotting function for treemap-like visualization using squarify with text wrapping
def plot_treemap_like(data, title, filename):
    if data.empty:
        print(f"No data to plot for {title}")
        return
    
    plt.figure(figsize=(14, 10))
    plt.title(title, fontsize=24, fontweight='bold')
    
    # Filter out zero weights
    data = data[data['Weight'] > 0]
    
    sizes = data['Weight'] * 100  # Adjust sizes for better visual representation
    labels = [f"{fill(name, width=20)}\n{weight:.2%}" for name, weight in zip(data['Name'], data['Weight'])]
    
    # Create a color map for types
    unique_types = data['Type'].unique()
    type_colors = {type_: plt.cm.tab20(i / len(unique_types)) for i, type_ in enumerate(unique_types)}
    colors = data['Type'].map(type_colors)
    
    # Plot using squarify with black borders
    squarify.plot(sizes=sizes, label=labels, alpha=.8, color=colors, text_kwargs={'fontsize': 18, 'weight': 'bold'}, edgecolor='black', linewidth=2)
    
    plt.gca().set_facecolor('lightgrey')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=type_colors[type_]) for type_ in unique_types]
    plt.legend(handles, unique_types, title="Security Type", loc="upper left", fontsize=16, title_fontsize=18)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"{title} plot saved as: {filename}")

# Define the directory to save figures
figure_directory = os.path.join(script_dir, '../Figures')
os.makedirs(figure_directory, exist_ok=True)

# Plotting treemap-like visualizations
plot_treemap_like(combined_5_years, 'Optimal Portfolio for 5 Years', os.path.join(figure_directory, 'optimal_portfolio_5_years.png'))
plot_treemap_like(combined_7_5_years, 'Optimal Portfolio for 7.5 Years', os.path.join(figure_directory, 'optimal_portfolio_7_5_years.png'))
plot_treemap_like(combined_10_years, 'Optimal Portfolio for 10 Years', os.path.join(figure_directory, 'optimal_portfolio_10_years.png'))