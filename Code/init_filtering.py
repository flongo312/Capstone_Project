import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
figure_directory = '/Users/frank/Desktop/Project/Figures'
data_file_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
output_file_path_top_assets_5_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_5_years.csv'
output_file_path_top_assets_7_5_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_7_5_years.csv'
output_file_path_top_assets_10_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_10_years.csv'
top_n = 20  # Number of top portfolios

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
print(f"Size of mutual_funds_returns: {mutual_funds_returns.shape}")
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

# Filter top securities for each horizon
def filter_top_assets_by_composite_score(data, score_column, top_n):
    top_assets = data.nlargest(top_n, score_column)
    return top_assets

filtered_top_securities_5_years = filter_top_assets_by_composite_score(filtered_combined_5_years, 'Composite Score 5 Years', top_n)
filtered_top_securities_7_5_years = filter_top_assets_by_composite_score(filtered_combined_7_5_years, 'Composite Score 7.5 Years', top_n)
filtered_top_securities_10_years = filter_top_assets_by_composite_score(filtered_combined_10_years, 'Composite Score 10 Years', top_n)

# Save the top assets based on composite score to CSV files
filtered_top_securities_5_years.to_csv(output_file_path_top_assets_5_years, index=False)
filtered_top_securities_7_5_years.to_csv(output_file_path_top_assets_7_5_years, index=False)
filtered_top_securities_10_years.to_csv(output_file_path_top_assets_10_years, index=False)

# Combined bar plot for Composite Score
def plot_composite_score_bar(filtered_data_5_years, filtered_data_7_5_years, filtered_data_10_years, title, file_name, top_n):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    filtered_data_5_years = filtered_data_5_years.sort_values(by='Composite Score 5 Years', ascending=False)
    filtered_data_7_5_years = filtered_data_7_5_years.sort_values(by='Composite Score 7.5 Years', ascending=False)
    filtered_data_10_years = filtered_data_10_years.sort_values(by='Composite Score 10 Years', ascending=False)

    # Bar width
    bar_width = 0.25

    # Set positions of bar on X axis
    r1 = np.arange(len(filtered_data_5_years))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, filtered_data_5_years['Composite Score 5 Years'], color='b', width=bar_width, edgecolor='grey', label='5 Years')
    ax.bar(r2, filtered_data_7_5_years['Composite Score 7.5 Years'], color='g', width=bar_width, edgecolor='grey', label='7.5 Years')
    ax.bar(r3, filtered_data_10_years['Composite Score 10 Years'], color='r', width=bar_width, edgecolor='grey', label='10 Years')

    ax.set_xlabel('Securities', fontsize=18)
    ax.set_ylabel('Composite Score', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticks([r + bar_width for r in range(len(filtered_data_5_years))])
    ax.set_xticklabels(filtered_data_5_years['Ticker'], rotation=90, ha='center', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Composite Score bar plot saved as: {file_name}")

# Improved scatter plot: Sharpe Ratio vs. CAPM Predicted Return
def plot_scatter_sharpe_vs_capm(filtered_data, title, file_name, colors=None):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    if colors is None:
        colors = {
            'Stocks': 'tab:blue',
            'ETFs': 'tab:green',
            'Mutual Funds': 'tab:orange'
        }
    
    for t in filtered_data['Type'].unique():
        subset = filtered_data[filtered_data['Type'] == t]
        if not subset.empty:
            ax.scatter(subset['CAPM Predicted Return'], subset['Sharpe Ratio'], label=t, alpha=0.7, s=100, color=colors[t])
            sns.regplot(x='CAPM Predicted Return', y='Sharpe Ratio', data=subset, ax=ax, scatter=False, color=colors[t], line_kws={"linestyle": "--"})
    
    for i, row in filtered_data.iterrows():
        ax.annotate(row['Ticker'], (row['CAPM Predicted Return'], row['Sharpe Ratio']), fontsize=9, alpha=0.75)
    
    ax.set_xlabel('CAPM Predicted Return', fontsize=18)
    ax.set_ylabel('Sharpe Ratio', fontsize=18)
    ax.set_title(title, fontsize=22)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Security Types', fontsize=14, title_fontsize=16)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Scatter plot saved as: {file_name}")

# Improved Sharpe Ratio Distribution Plot
def plot_sharpe_ratio_distribution(data, title, file_name, bins=30, color='blue'):
    plt.figure(figsize=(16, 12))
    sns.histplot(data['Sharpe Ratio'], kde=True, bins=bins, color=color)
    plt.title(title, fontsize=22)
    plt.xlabel('Sharpe Ratio', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.axvline(data['Sharpe Ratio'].mean(), color='r', linestyle='--', label=f'Mean: {data["Sharpe Ratio"].mean():.2f}')
    plt.axvline(data['Sharpe Ratio'].median(), color='g', linestyle='--', label=f'Median: {data["Sharpe Ratio"].median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Sharpe Ratio distribution plot saved as: {file_name}")

# Plot Composite Score Bar
plot_composite_score_bar(filtered_top_securities_5_years, filtered_top_securities_7_5_years, filtered_top_securities_10_years, 
                         'Top Assets by Composite Score', 
                         os.path.join(figure_directory, 'top_assets_composite_score_bar.png'), top_n=top_n)

# Plot Scatter Plot
plot_scatter_sharpe_vs_capm(combined_capm_results, 
                    'Sharpe Ratio vs. CAPM Predicted Return', 
                    os.path.join(figure_directory, 'sharpe_vs_capm_scatter.png'))

# Plot Sharpe Ratio Distribution
plot_sharpe_ratio_distribution(combined_capm_results, 
                               'Distribution of Sharpe Ratios', 
                               os.path.join(figure_directory, 'sharpe_ratio_distribution.png'))

print("Visualizations completed and saved.")
