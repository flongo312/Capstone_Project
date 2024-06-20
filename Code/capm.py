import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data by type
stocks_data = data[data['Type'] == 'Stocks']
etfs_data = data[data['Type'] == 'ETFs']
mutual_funds_data = data[data['Type'] == 'Mutual Funds']
crypto_data = data[data['Type'] == 'Cryptocurrencies']
market_data = data[data['Ticker'] == '^GSPC']

# Pivot the data to get daily closing prices for each type
def pivot_data(data):
    return data.pivot(index='Date', columns='Ticker', values='Adj Close')

stocks_data_pivot = pivot_data(stocks_data)
etfs_data_pivot = pivot_data(etfs_data)
mutual_funds_data_pivot = pivot_data(mutual_funds_data)
crypto_data_pivot = pivot_data(crypto_data)

# Calculate daily returns for each type
def calculate_returns(data_pivot):
    return data_pivot.pct_change().dropna()

stocks_returns = calculate_returns(stocks_data_pivot)
etfs_returns = calculate_returns(etfs_data_pivot)
mutual_funds_returns = calculate_returns(mutual_funds_data_pivot)
crypto_returns = calculate_returns(crypto_data_pivot)
market_data.set_index('Date', inplace=True)
market_returns = market_data['Adj Close'].pct_change().dropna()

# Merge returns with market returns
def merge_with_market(returns, market_returns):
    return returns.join(market_returns.rename('Market_Return'), how='inner')

combined_stocks_data = merge_with_market(stocks_returns, market_returns)
combined_etfs_data = merge_with_market(etfs_returns, market_returns)
combined_mutual_funds_data = merge_with_market(mutual_funds_returns, market_returns)
combined_crypto_data = merge_with_market(crypto_returns, market_returns)

# Calculate beta, expected return, and Sharpe ratio using CAPM
risk_free_rate = 0.03 / 252  # Daily risk-free rate

def calculate_beta_expected_return_sharpe(combined_data, market_return, risk_free_rate):
    betas = {}
    expected_returns = {}
    sharpe_ratios = {}
    market_premium = market_return.mean() - risk_free_rate

    for ticker in combined_data.columns[:-1]:  # Exclude 'Market_Return'
        asset_return = combined_data[ticker]
        covariance = np.cov(asset_return, market_return)[0, 1]
        market_variance = np.var(market_return)
        beta = covariance / market_variance
        betas[ticker] = beta
        expected_return = risk_free_rate + beta * market_premium
        expected_returns[ticker] = expected_return * 252  # Annualize the return
        std_dev = asset_return.std() * np.sqrt(252)  # Annualize the standard deviation
        sharpe_ratio = (expected_returns[ticker] - (risk_free_rate * 252)) / std_dev
        sharpe_ratios[ticker] = sharpe_ratio

    results = pd.DataFrame({
        'Ticker': list(betas.keys()),
        'Beta': list(betas.values()),
        'Expected Return': list(expected_returns.values()),
        'Sharpe Ratio': list(sharpe_ratios.values())
    })
    
    results['Sharpe Rank'] = results['Sharpe Ratio'].rank(ascending=False)
    return results

stocks_capm_results = calculate_beta_expected_return_sharpe(combined_stocks_data, market_returns, risk_free_rate)
etfs_capm_results = calculate_beta_expected_return_sharpe(combined_etfs_data, market_returns, risk_free_rate)
mutual_funds_capm_results = calculate_beta_expected_return_sharpe(combined_mutual_funds_data, market_returns, risk_free_rate)
crypto_capm_results = calculate_beta_expected_return_sharpe(combined_crypto_data, market_returns, risk_free_rate)

# Add Type column
stocks_capm_results['Type'] = 'Stocks'
etfs_capm_results['Type'] = 'ETFs'
mutual_funds_capm_results['Type'] = 'Mutual Funds'
crypto_capm_results['Type'] = 'Cryptocurrencies'

# Combine all results into a single DataFrame
combined_capm_results = pd.concat([
    stocks_capm_results,
    etfs_capm_results,
    mutual_funds_capm_results,
    crypto_capm_results
])

# Reorder columns to match the original dataset structure
combined_capm_results = combined_capm_results[['Ticker', 'Type', 'Beta', 'Expected Return', 'Sharpe Ratio', 'Sharpe Rank']]

# Merge CAPM results back with the original data
combined_data_with_original = data.merge(combined_capm_results, on=['Ticker', 'Type'], how='inner')

# Sort by Date and Sharpe Ratio rank
combined_data_with_original = combined_data_with_original.sort_values(by=['Date', 'Sharpe Rank'])

# Remove the 'Sharpe Rank' column but keep the order
combined_data_with_original = combined_data_with_original.drop(columns=['Sharpe Rank'])

# Filter out dates with fewer than a threshold number of securities
min_securities_per_date = 100  # Adjust this threshold based on your dataset
date_counts = combined_data_with_original['Date'].value_counts()
valid_dates = date_counts[date_counts >= min_securities_per_date].index
filtered_data = combined_data_with_original[combined_data_with_original['Date'].isin(valid_dates)]

# Further filter to the top securities by Sharpe Ratio and Beta Range for different time horizons
def filter_top_securities(filtered_data, beta_range, top_n):
    top_securities = combined_capm_results[(combined_capm_results['Beta'] >= beta_range[0]) & 
                                           (combined_capm_results['Beta'] <= beta_range[1])]
    top_securities = top_securities.nlargest(top_n, 'Sharpe Ratio')['Ticker'].unique()
    return filtered_data[filtered_data['Ticker'].isin(top_securities)]

# Define beta ranges for different investment horizons
beta_ranges = {
    '5_years': (0.5, 1.0),
    '10_years': (0.5, 1.5),
    '15_years': (0.5, 2.0)
}

top_n = 50  # Number of top securities to select

filtered_data_5_years = filter_top_securities(filtered_data, beta_ranges['5_years'], top_n)
filtered_data_10_years = filter_top_securities(filtered_data, beta_ranges['10_years'], top_n)
filtered_data_15_years = filter_top_securities(filtered_data, beta_ranges['15_years'], top_n)

# Save the filtered results to CSV files
output_file_path_5_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_5_years.csv'
output_file_path_10_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_10_years.csv'
output_file_path_15_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_15_years.csv'

filtered_data_5_years.to_csv(output_file_path_5_years, index=False)
filtered_data_10_years.to_csv(output_file_path_10_years, index=False)
filtered_data_15_years.to_csv(output_file_path_15_years, index=False)

# Visualization: Risk vs. Return for all securities
def plot_risk_vs_return(capm_results, title, file_name):
    plt.figure(figsize=(12, 8))
    plt.scatter(capm_results['Beta'], capm_results['Expected Return'], c=capm_results['Sharpe Ratio'], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Beta (Risk)')
    plt.ylabel('Expected Return')
    plt.title(title)
    for i in range(len(capm_results)):
        plt.text(capm_results['Beta'][i], capm_results['Expected Return'][i], capm_results['Ticker'][i], fontsize=9)
    plt.grid(True)
    plt.savefig(f'/Users/frank/Desktop/Project/Figures/{file_name}')
    plt.show()

# Visualization: Distribution of Sharpe Ratios
def plot_sharpe_ratio_distribution(capm_results, title, file_name):
    plt.figure(figsize=(12, 8))
    plt.hist(capm_results['Sharpe Ratio'], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'/Users/frank/Desktop/Project/Figures/{file_name}')
    plt.show()

# Plot for stocks
plot_risk_vs_return(stocks_capm_results, 'Stocks Risk (Beta) vs Expected Return', 'stocks_risk_vs_return.png')
plot_sharpe_ratio_distribution(stocks_capm_results, 'Distribution of Sharpe Ratios for Stocks', 'stocks_sharpe_ratio_distribution.png')

# Plot for ETFs
plot_risk_vs_return(etfs_capm_results, 'ETFs Risk (Beta) vs Expected Return', 'etfs_risk_vs_return.png')
plot_sharpe_ratio_distribution(etfs_capm_results, 'Distribution of Sharpe Ratios for ETFs', 'etfs_sharpe_ratio_distribution.png')

# Plot for mutual funds
plot_risk_vs_return(mutual_funds_capm_results, 'Mutual Funds Risk (Beta) vs Expected Return', 'mutual_funds_risk_vs_return.png')
plot_sharpe_ratio_distribution(mutual_funds_capm_results, 'Distribution of Sharpe Ratios for Mutual Funds', 'mutual_funds_sharpe_ratio_distribution.png')

# Plot for cryptocurrencies
plot_risk_vs_return(crypto_capm_results, 'Cryptocurrencies Risk (Beta) vs Expected Return', 'crypto_risk_vs_return.png')
plot_sharpe_ratio_distribution(crypto_capm_results, 'Distribution of Sharpe Ratios for Cryptocurrencies', 'crypto_sharpe_ratio_distribution.png')

# Visualization for filtered top securities for different time horizons
def plot_filtered_data(filtered_data, title, file_name):
    plt.figure(figsize=(12, 8))
    plt.scatter(filtered_data['Beta'], filtered_data['Expected Return'], c=filtered_data['Sharpe Ratio'], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Beta (Risk)')
    plt.ylabel('Expected Return')
    plt.title(title)
    for i in range(len(filtered_data)):
        plt.text(filtered_data['Beta'].iloc[i], filtered_data['Expected Return'].iloc[i], filtered_data['Ticker'].iloc[i], fontsize=9)
    plt.grid(True)
    plt.savefig(f'/Users/frank/Desktop/Project/Figures/{file_name}')
    plt.show()

# Plot for 5 years
plot_filtered_data(filtered_data_5_years, 'Top Securities for 5 Years (Beta 0.5 to 1.0)', 'top_securities_5_years.png')

# Plot for 10 years
plot_filtered_data(filtered_data_10_years, 'Top Securities for 10 Years (Beta 0.5 to 1.5)', 'top_securities_10_years.png')

# Plot for 15 years
plot_filtered_data(filtered_data_15_years, 'Top Securities for 15 Years (Beta 0.5 to 2.0)', 'top_securities_15_years.png')

