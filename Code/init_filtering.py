import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "pandas", "numpy", "matplotlib", "seaborn"
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration
figure_directory = os.path.join(script_dir, '../Figures')
data_file_path = os.path.join(script_dir, '../Data/yfinance_data.csv')
output_file_path_top_assets_5_years = os.path.join(script_dir, '../Data/top_assets_composite_score_5_years.csv')
output_file_path_top_assets_7_5_years = os.path.join(script_dir, '../Data/top_assets_composite_score_7_5_years.csv')
output_file_path_top_assets_10_years = os.path.join(script_dir, '../Data/top_assets_composite_score_10_years.csv')
top_n = 20  

# Create directory if it does not exist
os.makedirs(figure_directory, exist_ok=True)

# Load and prepare data
try:
    data = pd.read_csv(data_file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    print(f"Data range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Total number of records: {data.shape[0]}")
except FileNotFoundError:
    raise FileNotFoundError(f"The file at path {data_file_path} was not found.")
except Exception as e:
    raise Exception(f"An error occurred while loading the data: {e}")

# Filter data to ensure it falls within the known date range
data = data[(data['Date'] >= '2011-05-04') & (data['Date'] <= '2014-11-10')]

# Pivot the returns data to get adjusted close prices for each ticker
historical_data_pivot = data.pivot(index='Date', columns='Ticker', values='Adj Close')
print(f"Size of historical_data_pivot: {historical_data_pivot.shape}")

# Check for sufficient data range
if historical_data_pivot.shape[0] < 2:
    raise ValueError("Insufficient data range after pivoting. Please check the date filters and data availability.")

# Calculate daily returns and actual returns
historical_returns = historical_data_pivot.pct_change().dropna()
actual_returns = historical_data_pivot.iloc[-1] / historical_data_pivot.iloc[0] - 1
print(f"Size of historical_returns: {historical_returns.shape}")

# Ensure the market index is included in the historical data
if '^GSPC' not in historical_returns.columns:
    raise KeyError("Market index '^GSPC' not found in the historical returns data.")

# Composite Score for Multi-Criteria Filtering
def calculate_composite_score(data, weights):
    score = (data['Beta'] * weights['Beta'] +
             data['Sharpe Ratio'] * weights['Sharpe Ratio'] +
             data['CAPM Predicted Return'] * weights['CAPM Predicted Return'] +
             data['Actual Returns'] * weights['Actual Returns'])
    return score

# Define weights for the metrics for different horizons
weights_5_years = {
    'Beta': 0.20,
    'Sharpe Ratio': 0.30,
    'CAPM Predicted Return': 0.20,
    'Actual Returns': 0.30
}

weights_7_5_years = {
    'Beta': 0.20,
    'Sharpe Ratio': 0.25,
    'CAPM Predicted Return': 0.25,
    'Actual Returns': 0.30
}

weights_10_years = {
    'Beta': 0.15,
    'Sharpe Ratio': 0.25,
    'CAPM Predicted Return': 0.25,
    'Actual Returns': 0.35
}

# Filter and pivot data by type (Stocks, ETFs, Mutual Funds)
def filter_and_pivot_data(data, type_name=None):
    if type_name:
        data = data[data['Type'] == type_name]
    return data.pivot(index='Date', columns='Ticker', values='Adj Close')

# Prepare pivot tables for different asset types
stocks_data_pivot = filter_and_pivot_data(data, 'Stocks')
etfs_data_pivot = filter_and_pivot_data(data, 'ETFs')
mutual_funds_data_pivot = filter_and_pivot_data(data, 'Mutual Funds')
market_index_pivot = filter_and_pivot_data(data[data['Ticker'] == '^GSPC'])

# Calculate daily returns for each asset type and market index
stocks_returns = stocks_data_pivot.pct_change().dropna()
etfs_returns = etfs_data_pivot.pct_change().dropna()
mutual_funds_returns = mutual_funds_data_pivot.pct_change().dropna()
market_returns = market_index_pivot.pct_change().dropna()

# Print sizes of returns DataFrames
print(f"Size of stocks_returns: {stocks_returns.shape}")
print(f"Size of etfs_returns: {etfs_returns.shape}")
print (f"Size of mutual_funds_returns: {mutual_funds_returns.shape}")
print(f"Size of market_returns: {market_returns.shape}")

# Ensure we have valid data
if stocks_returns.empty or etfs_returns.empty or mutual_funds_returns.empty or market_returns.empty:
    raise ValueError("One or more returns DataFrames are empty after filtering and pivoting. Please check the input data.")

# Assume a constant annual risk-free rate of 3%
annual_rf_rate = 0.03
daily_rf_rate = (1 + annual_rf_rate)**(1/252) - 1

# Define a function to calculate beta
def calculate_beta(asset_returns, market_returns):
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    variance = np.var(market_returns)
    beta = covariance / variance
    return beta

# Recalculate metrics for all assets
def recalculate_all_metrics(asset_returns, market_returns, rf_rate):
    betas = asset_returns.apply(lambda x: calculate_beta(x, market_returns['^GSPC']), axis=0)
    sharpe_ratios = (asset_returns.mean() - rf_rate) / asset_returns.std()
    capm_predicted_return = (rf_rate + betas * (market_returns['^GSPC'].mean() - rf_rate)) * 252
    return betas, sharpe_ratios, capm_predicted_return

# Recalculate metrics for each asset type
stocks_betas, stocks_sharpe_ratios, stocks_capm_predicted_return = recalculate_all_metrics(stocks_returns, market_returns, daily_rf_rate)
etfs_betas, etfs_sharpe_ratios, etfs_capm_predicted_return = recalculate_all_metrics(etfs_returns, market_returns, daily_rf_rate)
mutual_funds_betas, mutual_funds_sharpe_ratios, mutual_funds_capm_predicted_return = recalculate_all_metrics(mutual_funds_returns, market_returns, daily_rf_rate)

# Merge recalculated metrics into final DataFrame
def merge_metrics(asset_returns, betas, sharpe_ratios, capm_predicted_return, actual_returns):
    common_tickers = asset_returns.columns
    
    capm_results = pd.DataFrame({
        'Ticker': common_tickers,
        'Beta': betas.values,
        'CAPM Predicted Return': capm_predicted_return.values,
        'Sharpe Ratio': sharpe_ratios.values,
        'Actual Returns': actual_returns[common_tickers].values
    })
    return capm_results

# Merge metrics for each asset type
stocks_capm_results = merge_metrics(stocks_returns, stocks_betas, stocks_sharpe_ratios, stocks_capm_predicted_return, actual_returns)
etfs_capm_results = merge_metrics(etfs_returns, etfs_betas, etfs_sharpe_ratios, etfs_capm_predicted_return, actual_returns)
mutual_funds_capm_results = merge_metrics(mutual_funds_returns, mutual_funds_betas, mutual_funds_sharpe_ratios, mutual_funds_capm_predicted_return, actual_returns)

# Add asset type labels
stocks_capm_results['Type'] = 'Stocks'
etfs_capm_results['Type'] = 'ETFs'
mutual_funds_capm_results['Type'] = 'Mutual Funds'

# Combine all results into a single DataFrame
combined_capm_results = pd.concat([stocks_capm_results, etfs_capm_results, mutual_funds_capm_results])

# Apply additional constraints to diversify recommendations for each time horizon
def apply_additional_constraints(data, horizon):
    if horizon == '5_years':
        return data[(data['Sharpe Ratio'] > data['Sharpe Ratio'].quantile(0.75)) & (data['Beta'] < 1.5)]
    elif horizon == '7_5_years':
        return data[(data['Sharpe Ratio'] > data['Sharpe Ratio'].quantile(0.5)) & (data['Beta'] < 2.0)]
    elif horizon == '10_years':
        return data[(data['Sharpe Ratio'] > data['Sharpe Ratio'].quantile(0.25)) & (data['Beta'] < 2.5)]
   
    return data

# Apply constraints for each horizon
filtered_combined_5_years = apply_additional_constraints(combined_capm_results, '5_years')
filtered_combined_7_5_years = apply_additional_constraints(combined_capm_results, '7_5_years')
filtered_combined_10_years = apply_additional_constraints(combined_capm_results, '10_years')

# Calculate and add the composite score for each horizon
filtered_combined_5_years = filtered_combined_5_years.copy()
filtered_combined_7_5_years = filtered_combined_7_5_years.copy()
filtered_combined_10_years = filtered_combined_10_years.copy()

filtered_combined_5_years['Composite Score 5 Years'] = calculate_composite_score(filtered_combined_5_years, weights_5_years)
filtered_combined_7_5_years['Composite Score 7.5 Years'] = calculate_composite_score(filtered_combined_7_5_years, weights_7_5_years)
filtered_combined_10_years['Composite Score 10 Years'] = calculate_composite_score(filtered_combined_10_years, weights_10_years)

# Verify if 'Composite Score' is added correctly
print("Filtered Combined Results Columns after adding Composite Score:",
      filtered_combined_5_years.columns,
      filtered_combined_7_5_years.columns,
      filtered_combined_10_years.columns)

# Rank assets by composite score within each horizon
filtered_combined_5_years['Rank'] = filtered_combined_5_years['Composite Score 5 Years'].rank(method='first', ascending=False)
filtered_combined_7_5_years['Rank'] = filtered_combined_7_5_years['Composite Score 7.5 Years'].rank(method='first', ascending=False)
filtered_combined_10_years['Rank'] = filtered_combined_10_years['Composite Score 10 Years'].rank(method='first', ascending=False)

# Filter top securities for each horizon
def filter_top_assets_by_composite_score(data, score_column, top_n):
    top_assets = data.nsmallest(top_n, 'Rank')
    return top_assets

filtered_top_securities_5_years = filter_top_assets_by_composite_score(filtered_combined_5_years, 'Composite Score 5 Years', top_n)
filtered_combined_7_5_years = filtered_combined_7_5_years[~filtered_combined_7_5_years['Ticker'].isin(filtered_top_securities_5_years['Ticker'])]
filtered_top_securities_7_5_years = filter_top_assets_by_composite_score(filtered_combined_7_5_years, 'Composite Score 7.5 Years', top_n)
filtered_combined_10_years = filtered_combined_10_years[~filtered_combined_10_years['Ticker'].isin(filtered_top_securities_5_years['Ticker'])]
filtered_combined_10_years = filtered_combined_10_years[~filtered_combined_10_years['Ticker'].isin(filtered_top_securities_7_5_years['Ticker'])]
filtered_top_securities_10_years = filter_top_assets_by_composite_score(filtered_combined_10_years, 'Composite Score 10 Years', top_n)

# Save the top assets based on composite score to CSV files
filtered_top_securities_5_years.to_csv(output_file_path_top_assets_5_years, index=False)
filtered_top_securities_7_5_years.to_csv(output_file_path_top_assets_7_5_years, index=False)
filtered_top_securities_10_years.to_csv(output_file_path_top_assets_10_years, index=False)

# Function to plot top assets by composite score for a given horizon
def plot_top_assets_by_composite_score(filtered_data, score_column, title, file_name, top_n):
    # Sort by the composite score and select the top N
    top_assets = filtered_data.sort_values(by=score_column, ascending=False).head(top_n)
    
    # Define a more appealing color palette
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(16, 12))
    
    bars = ax.bar(top_assets['Ticker'], top_assets[score_column], color=colors, edgecolor='grey')
    
    ax.set_xlabel('Securities', fontsize=24, fontweight='bold')
    ax.set_ylabel('Composite Score', fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=28, fontweight='bold')
    ax.set_xticks(np.arange(len(top_assets)))
    ax.set_xticklabels(top_assets['Ticker'], rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)  

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')

    # Style the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"{title} bar plot saved as: {file_name}")

# Example usage for each time horizon
plot_top_assets_by_composite_score(filtered_top_securities_5_years, 'Composite Score 5 Years', 
                                   'Top Assets by Composite Score - 5 Years', 
                                   os.path.join(figure_directory, 'top_assets_composite_score_5_years.png'), top_n=top_n)

plot_top_assets_by_composite_score(filtered_top_securities_7_5_years, 'Composite Score 7.5 Years', 
                                   'Top Assets by Composite Score - 7.5 Years', 
                                   os.path.join(figure_directory, 'top_assets_composite_score_7_5_years.png'), top_n=top_n)

plot_top_assets_by_composite_score(filtered_top_securities_10_years, 'Composite Score 10 Years', 
                                   'Top Assets by Composite Score - 10 Years', 
                                   os.path.join(figure_directory, 'top_assets_composite_score_10_years.png'), top_n=top_n)

print("Visualizations completed and saved.")
