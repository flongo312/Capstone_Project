import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory to save figures
figure_directory = '/Users/frank/Desktop/Project/Figures'
os.makedirs(figure_directory, exist_ok=True)

# Load the dataset
file_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data by type
stocks_data = data[data['Type'] == 'Stocks']
etfs_data = data[data['Type'] == 'ETFs']
mutual_funds_data = data[data['Type'] == 'Mutual Funds']
market_data = data[data['Ticker'] == '^GSPC']

# Pivot the data to get daily closing prices for each type
def pivot_data(data):
    return data.pivot(index='Date', columns='Ticker', values='Adj Close')

stocks_data_pivot = pivot_data(stocks_data)
etfs_data_pivot = pivot_data(etfs_data)
mutual_funds_data_pivot = pivot_data(mutual_funds_data)

# Calculate daily returns for each type
def calculate_returns(data_pivot):
    return data_pivot.pct_change().dropna()

stocks_returns = calculate_returns(stocks_data_pivot)
etfs_returns = calculate_returns(etfs_data_pivot)
mutual_funds_returns = calculate_returns(mutual_funds_data_pivot)
market_data.set_index('Date', inplace=True)
market_returns = market_data['Adj Close'].pct_change().dropna()

# Calculate actual annualized return
def calculate_annualized_return(data_pivot):
    if len(data_pivot) < 2:
        return None  # Or you can return 0 or some other value
    cumulative_return = (data_pivot.iloc[-1] / data_pivot.iloc[0]) - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(data_pivot)) - 1
    return annualized_return

stocks_actual_return = calculate_annualized_return(stocks_data_pivot)
etfs_actual_return = calculate_annualized_return(etfs_data_pivot)
mutual_funds_actual_return = calculate_annualized_return(mutual_funds_data_pivot)

# Merge returns with market returns
def merge_with_market(returns, market_returns):
    return returns.join(market_returns.rename('Market_Return'), how='inner')

combined_stocks_data = merge_with_market(stocks_returns, market_returns)
combined_etfs_data = merge_with_market(etfs_returns, market_returns)
combined_mutual_funds_data = merge_with_market(mutual_funds_returns, market_returns)

# Ensure combined data and market returns have matching indices
def align_indices(combined_data, market_returns):
    common_index = combined_data.index.intersection(market_returns.index)
    return combined_data.loc[common_index], market_returns.loc[common_index]

combined_stocks_data, aligned_market_returns = align_indices(combined_stocks_data, market_returns)
combined_etfs_data, aligned_market_returns = align_indices(combined_etfs_data, aligned_market_returns)
combined_mutual_funds_data, aligned_market_returns = align_indices(combined_mutual_funds_data, aligned_market_returns)

# Further ensure both dataframes have the same length
def ensure_same_length(combined_data, market_returns):
    min_length = min(len(combined_data), len(market_returns))
    return combined_data.iloc[:min_length], market_returns.iloc[:min_length]

combined_stocks_data, aligned_market_returns = ensure_same_length(combined_stocks_data, aligned_market_returns)
combined_etfs_data, aligned_market_returns = ensure_same_length(combined_etfs_data, aligned_market_returns)
combined_mutual_funds_data, aligned_market_returns = ensure_same_length(combined_mutual_funds_data, aligned_market_returns)

# Calculate beta, CAPM predicted return, actual return, and Sharpe ratio using CAPM
risk_free_rate = 0.03 / 252  # Daily risk-free rate

def calculate_metrics(combined_data, actual_returns, market_return, risk_free_rate):
    betas = {}
    capm_returns = {}
    sharpe_ratios = {}
    return_diffs = {}
    market_premium = market_return.mean() - risk_free_rate

    for ticker in combined_data.columns[:-1]:  # Exclude 'Market_Return'
        asset_return = combined_data[ticker]
        covariance = np.cov(asset_return, market_return)[0, 1]
        market_variance = np.var(market_return)
        beta = covariance / market_variance
        betas[ticker] = beta
        capm_return = risk_free_rate + beta * market_premium
        capm_returns[ticker] = capm_return * 252  # Annualize the return
        std_dev = asset_return.std() * np.sqrt(252)  # Annualize the standard deviation
        sharpe_ratio = (actual_returns[ticker] - (risk_free_rate * 252)) / std_dev
        sharpe_ratios[ticker] = sharpe_ratio
        return_diffs[ticker] = actual_returns[ticker] - capm_returns[ticker]

    results = pd.DataFrame({
        'Ticker': list(betas.keys()),
        'Beta': list(betas.values()),
        'CAPM Predicted Return': list(capm_returns.values()),
        'Actual Return': list(actual_returns),
        'Return Difference': list(return_diffs.values()),
        'Sharpe Ratio': list(sharpe_ratios.values())
    })
    
    return results

stocks_capm_results = calculate_metrics(combined_stocks_data, stocks_actual_return, aligned_market_returns, risk_free_rate)
etfs_capm_results = calculate_metrics(combined_etfs_data, etfs_actual_return, aligned_market_returns, risk_free_rate)
mutual_funds_capm_results = calculate_metrics(combined_mutual_funds_data, mutual_funds_actual_return, aligned_market_returns, risk_free_rate)

# Add Type column
stocks_capm_results['Type'] = 'Stocks'
etfs_capm_results['Type'] = 'ETFs'
mutual_funds_capm_results['Type'] = 'Mutual Funds'

# Combine all results into a single DataFrame
combined_capm_results = pd.concat([
    stocks_capm_results,
    etfs_capm_results,
    mutual_funds_capm_results
])

# Define beta ranges for different investment horizons
beta_ranges = {
    '5_years': (0.5, 1.0),
    '10_years': (0.5, 1.5),
    '15_years': (0.5, 2.0)
}

top_n = 10  # Number of top securities to select

# Combine rankings to score each security equally
def combine_rankings(combined_data):
    combined_data.loc[:, 'Rank_CAPM_Return'] = combined_data['CAPM Predicted Return'].rank(ascending=False)
    combined_data.loc[:, 'Rank_Actual_Return'] = combined_data['Actual Return'].rank(ascending=False)
    combined_data.loc[:, 'Rank_Sharpe_Ratio'] = combined_data['Sharpe Ratio'].rank(ascending=False)
    combined_data.loc[:, 'Combined_Score'] = (
        combined_data['Rank_CAPM_Return'] +
        combined_data['Rank_Actual_Return'] +
        combined_data['Rank_Sharpe_Ratio']
    )
    return combined_data

# Apply the combined rankings and filter top securities
def filter_top_securities(combined_data, beta_range, top_n):
    filtered_data = combined_data[
        (combined_data['Beta'] >= beta_range[0]) & 
        (combined_data['Beta'] <= beta_range[1])
    ].copy()
    filtered_data = combine_rankings(filtered_data)
    top_securities = filtered_data.nsmallest(top_n, 'Combined_Score')
    return top_securities.drop(columns=[col for col in ['Sharpe Rank'] if col in top_securities])

filtered_data_5_years = filter_top_securities(combined_capm_results, beta_ranges['5_years'], top_n)
filtered_data_10_years = filter_top_securities(combined_capm_results, beta_ranges['10_years'], top_n)
filtered_data_15_years = filter_top_securities(combined_capm_results, beta_ranges['15_years'], top_n)

# Save the filtered results to CSV files
output_file_path_5_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_5_years.csv'
output_file_path_10_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_10_years.csv'
output_file_path_15_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_15_years.csv'

filtered_data_5_years.to_csv(output_file_path_5_years, index=False)
filtered_data_10_years.to_csv(output_file_path_10_years, index=False)
filtered_data_15_years.to_csv(output_file_path_15_years, index=False)

# Scatter Plot: CAPM Predicted Returns vs. Beta with Sharpe Ratio Returns overlay
def plot_capm_vs_sharpe(filtered_data, title, file_name):
    plt.figure(figsize=(16, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_data)))
    
    jitter_strength = 0.01  # Adjust the jitter strength as needed

    jittered_points = []

    for i, (idx, row) in enumerate(filtered_data.iterrows()):
        capm_color = colors[i]
        sharpe_color = np.clip(capm_color * 0.5, 0, 1)  # Darker shade for Sharpe Ratio returns

        # Add jitter to the beta and return values
        jittered_beta_capm = row['Beta'] + np.random.uniform(-jitter_strength, jitter_strength)
        jittered_return_capm = row['CAPM Predicted Return'] + np.random.uniform(-jitter_strength, jitter_strength)
        jittered_return_actual = row['Actual Return'] + np.random.uniform(-jitter_strength, jitter_strength)

        # Store jittered points for annotations
        jittered_points.append((jittered_beta_capm, jittered_return_capm, jittered_return_actual, row['Ticker']))

        # Plot CAPM Predicted Return with jitter
        plt.scatter(
            jittered_beta_capm, 
            jittered_return_capm, 
            color=capm_color, marker='o', s=300, alpha=0.7, edgecolor='k'
        )
        # Plot Sharpe Ratio Return with jitter
        plt.scatter(
            jittered_beta_capm, 
            jittered_return_actual, 
            color=sharpe_color, marker='o', s=300, alpha=0.9, edgecolor='k'
        )
    
    plt.xlabel('Beta', fontsize=18)
    plt.ylabel('Return', fontsize=18)
    plt.title(title, fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Create a custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='CAPM Predicted Return', markerfacecolor='tab:blue', markersize=15, alpha=0.7, markeredgecolor='k'),
        plt.Line2D([0], [0], marker='o', color='w', label='Sharpe Ratio Return', markerfacecolor='tab:blue', markersize=15, alpha=0.9, markeredgecolor='k')
    ]
    plt.legend(handles=handles, loc='best', fontsize=14)

    plt.tight_layout()
    
    # Add text annotations for each point
    for jittered_beta_capm, jittered_return_capm, jittered_return_actual, ticker in jittered_points:
        plt.text(
            jittered_beta_capm, 
            jittered_return_capm, 
            ticker,
            fontsize=14,
            verticalalignment='bottom', horizontalalignment='right'
        )
        plt.text(
            jittered_beta_capm, 
            jittered_return_actual, 
            ticker,
            fontsize=14,
            verticalalignment='top', horizontalalignment='left'
        )

    # Automatically adjust limits based on data range
    beta_min = filtered_data['Beta'].min()
    beta_max = filtered_data['Beta'].max()
    capm_min = filtered_data['CAPM Predicted Return'].min()
    capm_max = filtered_data['CAPM Predicted Return'].max()
    actual_min = filtered_data['Actual Return'].min()
    actual_max = filtered_data['Actual Return'].max()

    plt.xlim(beta_min - 0.1 * (beta_max - beta_min), beta_max + 0.1 * (beta_max - beta_min))
    plt.ylim(min(capm_min, actual_min) - 0.1 * (max(capm_max, actual_max) - min(capm_min, actual_min)), 
             max(capm_max, actual_max) + 0.1 * (max(capm_max, actual_max) - min(capm_min, actual_min)))

    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"Scatter plot saved as: {os.path.join(figure_directory, file_name)}")

# Plot for each time horizon
plot_capm_vs_sharpe(filtered_data_5_years, 
                    'CAPM Predicted Return vs. Beta (5 Years) with Sharpe Ratio Overlay', 
                    'capm_vs_sharpe_5_years.png')

plot_capm_vs_sharpe(filtered_data_10_years, 
                    'CAPM Predicted Return vs. Beta (10 Years) with Sharpe Ratio Overlay', 
                    'capm_vs_sharpe_10_years.png')

plot_capm_vs_sharpe(filtered_data_15_years, 
                    'CAPM Predicted Return vs. Beta (15 Years) with Sharpe Ratio Overlay', 
                    'capm_vs_sharpe_15_years.png')

print("Visualizations completed and saved.")

