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

# Calculate annualized standard deviation (volatility)
def calculate_annualized_std(data_pivot):
    return data_pivot.pct_change().std() * np.sqrt(252)

stocks_actual_return = calculate_annualized_return(stocks_data_pivot)
etfs_actual_return = calculate_annualized_return(etfs_data_pivot)
mutual_funds_actual_return = calculate_annualized_return(mutual_funds_data_pivot)

stocks_volatility = calculate_annualized_std(stocks_data_pivot)
etfs_volatility = calculate_annualized_std(etfs_data_pivot)
mutual_funds_volatility = calculate_annualized_std(mutual_funds_data_pivot)

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

# Calculate beta, CAPM predicted return, actual return, and standard deviation using CAPM
risk_free_rate = 0.03 / 252  # Daily risk-free rate

def calculate_metrics(combined_data, actual_returns, volatilities, market_return, risk_free_rate):
    betas = {}
    capm_returns = {}
    sharpe_ratios = {}
    market_premium = market_return.mean() - risk_free_rate

    for ticker in combined_data.columns[:-1]:  # Exclude 'Market_Return'
        asset_return = combined_data[ticker]
        covariance = np.cov(asset_return, market_return)[0, 1]
        market_variance = np.var(market_return)
        beta = covariance / market_variance
        betas[ticker] = beta
        capm_return = risk_free_rate + beta * market_premium
        capm_returns[ticker] = capm_return * 252  # Annualize the return
        sharpe_ratio = (actual_returns[ticker] - (risk_free_rate * 252)) / volatilities[ticker]
        sharpe_ratios[ticker] = sharpe_ratio

        # Debug prints
        print(f"{ticker} - Beta: {beta:.2f}, CAPM Return: {capm_returns[ticker]:.2f}, Sharpe Ratio: {sharpe_ratios[ticker]:.2f}")

    results = pd.DataFrame({
        'Ticker': list(betas.keys()),
        'Beta': list(betas.values()),
        'CAPM Predicted Return': list(capm_returns.values()),
        'Actual Return': list(actual_returns),
        'Volatility': list(volatilities),
        'Sharpe Ratio': list(sharpe_ratios.values())
    })
    
    return results

stocks_capm_results = calculate_metrics(combined_stocks_data, stocks_actual_return, stocks_volatility, aligned_market_returns, risk_free_rate)
etfs_capm_results = calculate_metrics(combined_etfs_data, etfs_actual_return, etfs_volatility, aligned_market_returns, risk_free_rate)
mutual_funds_capm_results = calculate_metrics(combined_mutual_funds_data, mutual_funds_actual_return, mutual_funds_volatility, aligned_market_returns, risk_free_rate)

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
    '5_years': (0.5, 0.8),
    '10_years': (0.8, 1.0),
    '15_years': (1.0, 1.2)
}

top_n = 40  # Number of top securities

# Apply the beta filtering
def filter_top_securities(combined_data, beta_range, top_n):
    filtered_data = combined_data[
        (combined_data['Beta'] >= beta_range[0]) & 
        (combined_data['Beta'] <= beta_range[1])
    ].copy()
    top_securities = filtered_data.nsmallest(top_n, 'Beta')
    return top_securities

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

# Double Bar Plot: Actual Returns vs. CAPM Predicted Returns
def plot_actual_vs_capm(filtered_data, title, file_name):
    # Sort the data by CAPM Predicted Return for better readability
    filtered_data = filtered_data.sort_values(by='CAPM Predicted Return', ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, 12))

    bar_width = 0.35
    index = np.arange(len(filtered_data))

    type_colors = {
        'Stocks': 'tab:blue',
        'ETFs': 'tab:green',
        'Mutual Funds': 'tab:orange'
    }

    capm_bar = ax.bar(index - bar_width/2, filtered_data['CAPM Predicted Return'], bar_width, label='CAPM Predicted Return', color='b', alpha=0.7)
    actual_bar = ax.bar(index + bar_width/2, filtered_data['Actual Return'], bar_width, label='Actual Return', color='w', edgecolor='black', hatch='//', alpha=0.7)

    # Set colors based on security type
    for bar, ticker in zip(capm_bar, filtered_data['Ticker']):
        bar.set_color(type_colors[filtered_data.loc[filtered_data['Ticker'] == ticker, 'Type'].values[0]])
    for bar, ticker in zip(actual_bar, filtered_data['Ticker']):
        bar.set_edgecolor('black')

    # Add data labels on each bar
    for i in range(len(filtered_data)):
        ax.text(index[i] - bar_width/2, filtered_data['CAPM Predicted Return'].iloc[i], f'{filtered_data["CAPM Predicted Return"].iloc[i]:.2f}', 
                ha='center', va='bottom', fontsize=10, rotation=45)
        ax.text(index[i] + bar_width/2, filtered_data['Actual Return'].iloc[i], f'{filtered_data["Actual Return"].iloc[i]:.2f}', 
                ha='center', va='bottom', fontsize=10, rotation=45)

    ax.set_xlabel('Securities', fontsize=18)
    ax.set_ylabel('Return', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticks(index)
    ax.set_xticklabels(filtered_data['Ticker'], rotation=90, ha='center', fontsize=10)

    # Adjust y-axis to improve readability
    ax.set_ylim([min(filtered_data['Actual Return'].min(), filtered_data['CAPM Predicted Return'].min()) * 0.95, 
                 max(filtered_data['Actual Return'].max(), filtered_data['CAPM Predicted Return'].max()) * 1.05])

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add legend for security types and actual return pattern
    type_legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in type_colors.values()]
    type_labels = type_colors.keys()
    type_legend = plt.legend(type_legend_handles, type_labels, title='Security Types', fontsize=14, title_fontsize=16, loc='upper right')
    hatch_handle = plt.Rectangle((0, 0), 1, 1, color='w', edgecolor='black', hatch='//', alpha=0.7)
    hatch_legend_handles = [plt.Line2D([0], [0], color='b', lw=4, alpha=0.7), hatch_handle]
    plt.legend(handles=hatch_legend_handles, labels=['CAPM Predicted Return', 'Actual Return'], fontsize=14, loc='upper left')
    
    ax.add_artist(type_legend)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"Bar plot saved as: {os.path.join(figure_directory, file_name)}")

# Plot for each time horizon
plot_actual_vs_capm(filtered_data_5_years, 
                    'Actual Return vs. CAPM Predicted Return (5 Years)', 
                    'actual_vs_capm_5_years.png')

plot_actual_vs_capm(filtered_data_10_years, 
                    'Actual Return vs. CAPM Predicted Return (10 Years)', 
                    'actual_vs_capm_10_years.png')

plot_actual_vs_capm(filtered_data_15_years, 
                    'Actual Return vs. CAPM Predicted Return (15 Years)', 
                    'actual_vs_capm_15_years.png')

print("Visualizations completed and saved.")
