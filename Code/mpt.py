import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
from scipy.optimize import minimize
import os

# Load the filtered top securities
file_path_5_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_5_years.csv'
file_path_10_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_10_years.csv'
file_path_15_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_15_years.csv'

filtered_top_securities_5_years = pd.read_csv(file_path_5_years)
filtered_top_securities_10_years = pd.read_csv(file_path_10_years)
filtered_top_securities_15_years = pd.read_csv(file_path_15_years)

# Load the returns data
returns_data_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
returns_data = pd.read_csv(returns_data_path)
returns_data['Date'] = pd.to_datetime(returns_data['Date'])
returns_data.set_index('Date', inplace=True)

# Pivot the returns data to get adjusted close prices for each ticker
returns_data = returns_data.pivot(columns='Ticker', values='Adj Close')

# Calculate daily returns
returns_data = returns_data.pct_change().dropna()

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

# Function to calculate negative Sharpe ratio (for minimization)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# Function to perform portfolio optimization
def optimize_portfolio(filtered_data, returns_data, risk_free_rate=0.03):
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

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(tickers)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for _ in range(num_assets))

    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return pd.DataFrame({
        'Ticker': tickers,
        'Weight': result.x
    })

# Optimize portfolios
optimal_weights_5_years = optimize_portfolio(filtered_top_securities_5_years, returns_data)
optimal_weights_10_years = optimize_portfolio(filtered_top_securities_10_years, returns_data)
optimal_weights_15_years = optimize_portfolio(filtered_top_securities_15_years, returns_data)

# Save the optimal weights to CSV files
optimal_weights_path_5_years = '/Users/frank/Desktop/Project/Data/optimal_weights_5_years.csv'
optimal_weights_path_10_years = '/Users/frank/Desktop/Project/Data/optimal_weights_10_years.csv'
optimal_weights_path_15_years = '/Users/frank/Desktop/Project/Data/optimal_weights_15_years.csv'

if not optimal_weights_5_years.empty:
    optimal_weights_5_years.to_csv(optimal_weights_path_5_years, index=False)
else:
    print(f"Skipping saving for 5 years as no optimal weights were calculated.")

if not optimal_weights_10_years.empty:
    optimal_weights_10_years.to_csv(optimal_weights_path_10_years, index=False)
else:
    print(f"Skipping saving for 10 years as no optimal weights were calculated.")

if not optimal_weights_15_years.empty:
    optimal_weights_15_years.to_csv(optimal_weights_path_15_years, index=False)
else:
    print(f"Skipping saving for 15 years as no optimal weights were calculated.")

# Function to merge and display the combined data
def display_combined_data(filtered_data, optimal_weights, title):
    if optimal_weights.empty:
        print(f"No optimal weights to display for {title}")
        return pd.DataFrame()
    
    combined_data = filtered_data.merge(optimal_weights, on='Ticker')
    combined_data_sorted = combined_data.sort_values(by='Weight', ascending=False)
    combined_data_sorted['Weight'] = combined_data_sorted['Weight'].round(4)
    
    print(title)
    print(combined_data_sorted)
    
    return combined_data_sorted

# 5-Year Horizon
combined_5_years = display_combined_data(filtered_top_securities_5_years, optimal_weights_5_years, '5-Year Horizon')

# 10-Year Horizon
combined_10_years = display_combined_data(filtered_top_securities_10_years, optimal_weights_10_years, '10-Year Horizon')

# 15-Year Horizon
combined_15_years = display_combined_data(filtered_top_securities_15_years, optimal_weights_15_years, '15-Year Horizon')

# Improved plotting function for treemap-like visualization using squarify
def plot_treemap_like(data, title, filename):
    if data.empty:
        print(f"No data to plot for {title}")
        return
    
    plt.figure(figsize=(14, 10))
    plt.title(title, fontsize=24, fontweight='bold')
    
    # Filter out zero weights
    data = data[data['Weight'] > 0]
    
    sizes = data['Weight']
    labels = [f"{ticker}\n{weight:.2%}" for ticker, weight in zip(data['Ticker'], data['Weight'])]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    
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
plot_treemap_like(combined_10_years, 'Optimal Portfolio for 10 Years', os.path.join(figure_directory, 'optimal_portfolio_10_years.png'))
plot_treemap_like(combined_15_years, 'Optimal Portfolio for 15 Years', os.path.join(figure_directory, 'optimal_portfolio_15_years.png'))
