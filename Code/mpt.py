import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
from scipy.optimize import minimize
import os

# Comprehensive dictionary for ticker names
ticker_info = {
    'AAPL': 'Apple Inc.', 'TSLA': 'Tesla Inc.', 'AMZN': 'Amazon.com Inc.',
    'GOOGL': 'Alphabet Inc.', 'NVDA': 'NVIDIA Corporation', 'META': 'Meta Platforms Inc.',
    'AMD': 'Advanced Micro Devices Inc.', 'BAC': 'Bank of America Corporation',
    'MSFT': 'Microsoft Corp.', 'NFLX': 'Netflix Inc.', 'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.', 'DIS': 'Walt Disney Co.', 'CSCO': 'Cisco Systems Inc.',
    'XOM': 'Exxon Mobil Corporation', 'PFE': 'Pfizer Inc.', 'KO': 'Coca-Cola Co.',
    'PEP': 'PepsiCo Inc.', 'INTC': 'Intel Corporation', 'CRM': 'Salesforce Inc.',
    'T': 'AT&T Inc.', 'MRK': 'Merck & Co. Inc.', 'WMT': 'Walmart Inc.',
    'NKE': 'Nike Inc.', 'GE': 'General Electric Co.', 'COST': 'Costco Wholesale Corporation',
    'MCD': 'McDonald\'s Corporation', 'SBUX': 'Starbucks Corporation', 'IBM': 'International Business Machines Corporation',
    'WFC': 'Wells Fargo & Company', 'UNH': 'UnitedHealth Group Incorporated', 'ORCL': 'Oracle Corporation',
    'QCOM': 'QUALCOMM Incorporated', 'AVGO': 'Broadcom Inc.', 'TM': 'Toyota Motor Corporation',
    'ADBE': 'Adobe Inc.', 'TXN': 'Texas Instruments Incorporated', 'HON': 'Honeywell International Inc.',
    'MO': 'Altria Group Inc.', 'MDT': 'Medtronic plc', 'SPG': 'Simon Property Group Inc.',
    'AXP': 'American Express Company', 'GS': 'The Goldman Sachs Group Inc.', 'LMT': 'Lockheed Martin Corporation',
    'BMY': 'Bristol-Myers Squibb Company', 'BLK': 'BlackRock Inc.', 'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust', 'IWM': 'iShares Russell 2000 ETF', 'GLD': 'SPDR Gold Shares',
    'EEM': 'iShares MSCI Emerging Markets ETF', 'VOO': 'Vanguard S&P 500 ETF', 'XLF': 'Financial Select Sector SPDR Fund',
    'TQQQ': 'ProShares UltraPro QQQ', 'DIA': 'SPDR Dow Jones Industrial Average ETF Trust',
    'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF', 'IVV': 'iShares Core S&P 500 ETF',
    'VTI': 'Vanguard Total Stock Market ETF', 'XLK': 'Technology Select Sector SPDR Fund',
    'VNQ': 'Vanguard Real Estate ETF', 'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
    'XLV': 'Health Care Select Sector SPDR Fund', 'XLE': 'Energy Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR Fund', 'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLU': 'Utilities Select Sector SPDR Fund', 'GDX': 'VanEck Vectors Gold Miners ETF',
    'XLI': 'Industrial Select Sector SPDR Fund', 'IWF': 'iShares Russell 1000 Growth ETF',
    'IWB': 'iShares Russell 1000 ETF', 'IJH': 'iShares Core S&P Mid-Cap ETF', 'IYR': 'iShares U.S. Real Estate ETF',
    'VUG': 'Vanguard Growth ETF', 'IJR': 'iShares Core S&P Small-Cap ETF', 'IWN': 'iShares Russell 2000 Value ETF',
    'EFA': 'iShares MSCI EAFE ETF', 'VWO': 'Vanguard FTSE Emerging Markets ETF', 'SHY': 'iShares 1-3 Year Treasury Bond ETF',
    'USO': 'United States Oil Fund LP', 'TLT': 'iShares 20+ Year Treasury Bond ETF', 'VGK': 'Vanguard FTSE Europe ETF',
    'XOP': 'SPDR S&P Oil & Gas Exploration & Production ETF', 'IEFA': 'iShares Core MSCI EAFE ETF',
    'IEMG': 'iShares Core MSCI Emerging Markets ETF', 'PFF': 'iShares Preferred and Income Securities ETF',
    'EWT': 'iShares MSCI Taiwan ETF', 'EWJ': 'iShares MSCI Japan ETF', 'IWC': 'iShares Micro-Cap ETF',
    'VB': 'Vanguard Small-Cap ETF', 'SCHX': 'Schwab U.S. Large-Cap ETF', 'VIG': 'Vanguard Dividend Appreciation ETF',
    'SCHF': 'Schwab International Equity ETF', 'EZU': 'iShares MSCI Eurozone ETF', 'IGV': 'iShares Expanded Tech-Software Sector ETF',
    'VCSH': 'Vanguard Short-Term Corporate Bond ETF', 'SPDW': 'SPDR Portfolio Developed World ex-US ETF',
    'VFIAX': 'Vanguard 500 Index Fund Admiral Shares', 'VTSAX': 'Vanguard Total Stock Market Index Fund Admiral Shares',
    'FXAIX': 'Fidelity 500 Index Fund', 'FCNTX': 'Fidelity Contrafund', 'AGTHX': 'American Funds Growth Fund of America A',
    'TRBCX': 'T. Rowe Price Blue Chip Growth Fund', 'VBTLX': 'Vanguard Total Bond Market Index Fund Admiral Shares',
    'SWPPX': 'Schwab S&P 500 Index Fund', 'VWELX': 'Vanguard Wellington Fund Investor Shares', 'DODGX': 'Dodge & Cox Stock Fund',
    'VIGAX': 'Vanguard Growth Index Fund Admiral Shares', 'PRGFX': 'T. Rowe Price Growth Stock Fund', 'VWIAX': 'Vanguard Wellesley Income Fund Admiral Shares',
    'DFELX': 'Dimensional U.S. Core Equity 2 ETF', 'VIMAX': 'Vanguard Mid Cap Index Fund Admiral Shares', 'VPCCX': 'Vanguard PRIMECAP Core Fund',
    'VEIPX': 'Vanguard Equity-Income Fund Investor Shares', 'VWINX': 'Vanguard Wellesley Income Fund Investor Shares', 'VTIAX': 'Vanguard Total International Stock Index Fund Admiral Shares',
    'PTTRX': 'PIMCO Total Return Fund Institutional Class', 'POAGX': 'PRIMECAP Odyssey Aggressive Growth Fund', 'RPMGX': 'T. Rowe Price Mid-Cap Growth Fund',
    'VWUAX': 'Vanguard U.S. Growth Fund Admiral Shares', 'FAGIX': 'Fidelity Capital & Income Fund', 'VGHAX': 'Vanguard Health Care Fund Admiral Shares',
    'FFIDX': 'Fidelity Fund', 'FBGRX': 'Fidelity Blue Chip Growth Fund', 'FSPTX': 'Fidelity Select Technology Portfolio',
    'FDGRX': 'Fidelity Growth Company Fund', 'VWNDX': 'Vanguard Windsor Fund Investor Shares', 'JATTX': 'Janus Henderson Triton Fund Class T',
    'SGENX': 'First Eagle Global Fund Class A', 'RWMFX': 'American Funds Washington Mutual Investors Fund Class F-1', 'PRHSX': 'T. Rowe Price Health Sciences Fund',
    'FLPSX': 'Fidelity Low-Priced Stock Fund', 'FBALX': 'Fidelity Balanced Fund', 'FAIRX': 'Fairholme Fund',
    'VGSLX': 'Vanguard Real Estate Index Fund Admiral Shares', 'VDIGX': 'Vanguard Dividend Growth Fund Investor Shares', 'MGVAX': 'MFS Value Fund Class A',
    'PRWCX': 'T. Rowe Price Capital Appreciation Fund', 'PARNX': 'Parnassus Mid Cap Fund Investor Shares', 'PRNHX': 'T. Rowe Price New Horizons Fund',
    'RYVPX': 'Royce Value Plus Fund Investment Class', 'SWHGX': 'Schwab Health Care Fund', 'VHCAX': 'Vanguard Health Care Fund Admiral Shares',
    'VSEQX': 'Vanguard Strategic Equity Fund Investor Shares', '^GSPC': 'S&P 500', '^DJI': 'Dow Jones Industrial Average', '^IXIC': 'NASDAQ Composite'
}

# Load the top assets based on composite score from the saved CSV files
filtered_top_securities_5_years = pd.read_csv('/Users/frank/Desktop/Project/Data/top_assets_composite_score_5_years.csv')
filtered_top_securities_7_5_years = pd.read_csv('/Users/frank/Desktop/Project/Data/top_assets_composite_score_7_5_years.csv')
filtered_top_securities_10_years = pd.read_csv('/Users/frank/Desktop/Project/Data/top_assets_composite_score_10_years.csv')

# Load the returns data
returns_data_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
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
optimal_weights_path_5_years = '/Users/frank/Desktop/Project/Data/optimal_weights_5_years.csv'
optimal_weights_path_7_5_years = '/Users/frank/Desktop/Project/Data/optimal_weights_7_5_years.csv'
optimal_weights_path_10_years = '/Users/frank/Desktop/Project/Data/optimal_weights_10_years.csv'

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

# Improved plotting function for treemap-like visualization using squarify
def plot_treemap_like(data, title, filename):
    if data.empty:
        print(f"No data to plot for {title}")
        return
    
    plt.figure(figsize=(14, 10))
    plt.title(title, fontsize=24, fontweight='bold')
    
    # Filter out zero weights
    data = data[data['Weight'] > 0]
    
    sizes = np.log(data['Weight'] + 1)  # Apply logarithmic scaling
    labels = [f"{name}\n{weight:.2%}" for name, weight in zip(data['Name'], data['Weight'])]
    
    # Create a color map for types
    unique_types = data['Type'].unique()
    type_colors = {type_: plt.cm.tab20(i / len(unique_types)) for i, type_ in enumerate(unique_types)}
    colors = data['Type'].map(type_colors)
    
    squarify.plot(sizes=sizes, label=labels, alpha=.8, color=colors, text_kwargs={'fontsize': 14, 'weight': 'bold'})
    
    plt.gca().set_facecolor('lightgrey')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"{title} plot saved as: {filename}")

# Define the directory to save figures
figure_directory = '/Users/frank/Desktop/Project/Figures'
os.makedirs(figure_directory, exist_ok=True)

# Plotting treemap-like visualizations
plot_treemap_like(combined_5_years, 'Optimal Portfolio for 5 Years', os.path.join(figure_directory, 'optimal_portfolio_5_years.png'))
plot_treemap_like(combined_7_5_years, 'Optimal Portfolio for 7.5 Years', os.path.join(figure_directory, 'optimal_portfolio_7_5_years.png'))
plot_treemap_like(combined_10_years, 'Optimal Portfolio for 10 Years', os.path.join(figure_directory, 'optimal_portfolio_10_years.png'))

