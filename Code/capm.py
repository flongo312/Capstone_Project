import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

# Define the directory to save figures
figure_directory = '/Users/frank/Desktop/Project/Figures'
os.makedirs(figure_directory, exist_ok=True)

# Load and prepare data
file_path = '/Users/frank/Desktop/Project/Data/yfinance_data.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])

# Filter and pivot data
def filter_data_by_type(data, type_name):
    return data[data['Type'] == type_name]

def pivot_data(data):
    return data.pivot(index='Date', columns='Ticker', values='Adj Close')

stocks_data = filter_data_by_type(data, 'Stocks')
etfs_data = filter_data_by_type(data, 'ETFs')
mutual_funds_data = filter_data_by_type(data, 'Mutual Funds')

stocks_data_pivot = pivot_data(stocks_data)
etfs_data_pivot = pivot_data(etfs_data)
mutual_funds_data_pivot = pivot_data(mutual_funds_data)

# Calculate daily returns
def calculate_returns(data_pivot):
    return data_pivot.pct_change().dropna()

stocks_returns = calculate_returns(stocks_data_pivot)
etfs_returns = calculate_returns(etfs_data_pivot)
mutual_funds_returns = calculate_returns(mutual_funds_data_pivot)

# Load Fama-French factors
def load_fama_french_data(file_path):
    ff_factors = pd.read_csv(file_path, skiprows=3)
    ff_factors.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    ff_factors = ff_factors[:-1]
    ff_factors['Date'] = pd.to_datetime(ff_factors['Date'], format='%Y%m%d')
    ff_factors.set_index('Date', inplace=True)
    return ff_factors

ff_factors_path = '/Users/frank/Desktop/Project/Data/F-F_Research_Data_Factors_daily.CSV'
ff_factors = load_fama_french_data(ff_factors_path)

# Align indices
def align_indices(*dataframes):
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        common_index = common_index.intersection(df.index)
    return [df.loc[common_index] for df in dataframes]

stocks_returns, etfs_returns, mutual_funds_returns, ff_factors = align_indices(
    stocks_returns, etfs_returns, mutual_funds_returns, ff_factors)

# Calculate annualized return and excess returns
def calculate_annualized_metrics(data_pivot, rf_column):
    returns = data_pivot.pct_change().dropna()
    rf_column_aligned = rf_column.loc[returns.index]  # Align the rf_column with returns
    excess_returns = returns.sub(rf_column_aligned.values, axis=0)
    annualized_return = (1 + returns.mean()) ** 252 - 1
    return annualized_return, excess_returns

stocks_actual_return, stocks_excess_returns = calculate_annualized_metrics(stocks_data_pivot, ff_factors['RF'])
etfs_actual_return, etfs_excess_returns = calculate_annualized_metrics(etfs_data_pivot, ff_factors['RF'])
mutual_funds_actual_return, mutual_funds_excess_returns = calculate_annualized_metrics(mutual_funds_data_pivot, ff_factors['RF'])

# Calculate Fama-French metrics using excess returns
def calculate_fama_french_metrics(asset_excess_returns, fama_french_factors):
    market_excess = fama_french_factors['Mkt-RF']
    smb = fama_french_factors['SMB']
    hml = fama_french_factors['HML']

    X = pd.DataFrame({
        'Market': market_excess,
        'SMB': smb,
        'HML': hml
    })
    X = sm.add_constant(X)

    metrics = {}
    for ticker in asset_excess_returns.columns:
        y = asset_excess_returns[ticker]
        model = sm.OLS(y, X).fit()
        metrics[ticker] = model.params

    return pd.DataFrame(metrics).T

stocks_fama_french_metrics = calculate_fama_french_metrics(stocks_excess_returns, ff_factors)
etfs_fama_french_metrics = calculate_fama_french_metrics(etfs_excess_returns, ff_factors)
mutual_funds_fama_french_metrics = calculate_fama_french_metrics(mutual_funds_excess_returns, ff_factors)

# Calculate additional metrics
def calculate_additional_metrics(asset_returns, benchmark_returns, beta):
    alpha = asset_returns.mean() - (beta * benchmark_returns.mean())
    return alpha

# Assuming 'market_returns' is the benchmark
market_returns = ff_factors['Mkt-RF'] + ff_factors['RF']

stocks_alpha = calculate_additional_metrics(stocks_returns, market_returns, stocks_fama_french_metrics['Market'])
etfs_alpha = calculate_additional_metrics(etfs_returns, market_returns, etfs_fama_french_metrics['Market'])
mutual_funds_alpha = calculate_additional_metrics(mutual_funds_returns, market_returns, mutual_funds_fama_french_metrics['Market'])

# Merge metrics into final DataFrame
def merge_metrics(asset_returns, fama_french_metrics, actual_return, alpha, rf_column):
    capm_results = pd.DataFrame({
        'Ticker': asset_returns.columns,
        'Beta': fama_french_metrics['Market'],
        'SMB': fama_french_metrics['SMB'],
        'HML': fama_french_metrics['HML'],
        'CAPM Predicted Return': (rf_column + fama_french_metrics['Market'] * (asset_returns.mean() - rf_column)).mean() * 252,
        'Actual Return': actual_return,
        'Alpha': alpha,
        'Sharpe Ratio': (actual_return - rf_column.mean() * 252) / (asset_returns.std() * np.sqrt(252))
    })
    return capm_results

stocks_capm_results = merge_metrics(stocks_returns, stocks_fama_french_metrics, stocks_actual_return, stocks_alpha, ff_factors['RF'])
etfs_capm_results = merge_metrics(etfs_returns, etfs_fama_french_metrics, etfs_actual_return, etfs_alpha, ff_factors['RF'])
mutual_funds_capm_results = merge_metrics(mutual_funds_returns, mutual_funds_fama_french_metrics, mutual_funds_actual_return, mutual_funds_alpha, ff_factors['RF'])

stocks_capm_results['Type'] = 'Stocks'
etfs_capm_results['Type'] = 'ETFs'
mutual_funds_capm_results['Type'] = 'Mutual Funds'

combined_capm_results = pd.concat([stocks_capm_results, etfs_capm_results, mutual_funds_capm_results])

# Add VaR and CVaR
def add_risk_metrics(results, var, cvar):
    results['VaR'] = var
    results['CVaR'] = cvar
    return results

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

stocks_var = stocks_returns.apply(calculate_var, confidence_level=0.95)
stocks_cvar = stocks_returns.apply(calculate_cvar, confidence_level=0.95)
etfs_var = etfs_returns.apply(calculate_var, confidence_level=0.95)
etfs_cvar = etfs_returns.apply(calculate_cvar, confidence_level=0.95)
mutual_funds_var = mutual_funds_returns.apply(calculate_var, confidence_level=0.95)
mutual_funds_cvar = mutual_funds_returns.apply(calculate_cvar, confidence_level=0.95)

stocks_capm_results = add_risk_metrics(stocks_capm_results, stocks_var, stocks_cvar)
etfs_capm_results = add_risk_metrics(etfs_capm_results, etfs_var, etfs_cvar)
mutual_funds_capm_results = add_risk_metrics(mutual_funds_capm_results, mutual_funds_var, mutual_funds_cvar)

combined_capm_results = pd.concat([stocks_capm_results, etfs_capm_results, mutual_funds_capm_results])

# Define beta ranges for different investment horizons
beta_ranges = {
    '5_years': (0.5, 0.8),
    '10_years': (0.8, 1.0),
    '15_years': (1.0, 1.2)
}

top_n = 40  # Number of top portfolios

# Calculate portfolio-level betas
def calculate_portfolio_betas(combined_data, beta_range):
    portfolio_betas = combined_data[(combined_data['Beta'] >= beta_range[0]) & 
                                    (combined_data['Beta'] <= beta_range[1])]
    return portfolio_betas

filtered_portfolios_5_years = calculate_portfolio_betas(combined_capm_results, beta_ranges['5_years'])
filtered_portfolios_10_years = calculate_portfolio_betas(combined_capm_results, beta_ranges['10_years'])
filtered_portfolios_15_years = calculate_portfolio_betas(combined_capm_results, beta_ranges['15_years'])

# Composite Score for Multi-Criteria Filtering
def calculate_composite_score(data, weights):
    score = (data['Beta'] * weights['Beta'] +
             data['Sharpe Ratio'] * weights['Sharpe Ratio'] +
             data['Alpha'] * weights['Alpha'])
    return score

# Define weights for the metrics
weights = {
    'Beta': 0.25,
    'Sharpe Ratio': 0.35,
    'Alpha': 0.4
}

combined_capm_results['Composite Score'] = calculate_composite_score(combined_capm_results, weights)

# Filter top N assets based on composite score
def filter_top_assets_by_composite_score(data, top_n):
    top_assets = data.nlargest(top_n, 'Composite Score')
    return top_assets

top_assets_5_years = filter_top_assets_by_composite_score(filtered_portfolios_5_years, top_n)
top_assets_10_years = filter_top_assets_by_composite_score(filtered_portfolios_10_years, top_n)
top_assets_15_years = filter_top_assets_by_composite_score(filtered_portfolios_15_years, top_n)

# Save the top assets based on composite score to CSV files
output_file_path_top_assets_5_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_5_years.csv'
output_file_path_top_assets_10_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_10_years.csv'
output_file_path_top_assets_15_years = '/Users/frank/Desktop/Project/Data/top_assets_composite_score_15_years.csv'

top_assets_5_years.to_csv(output_file_path_top_assets_5_years, index=False)
top_assets_10_years.to_csv(output_file_path_top_assets_10_years, index=False)
top_assets_15_years.to_csv(output_file_path_top_assets_15_years, index=False)

# Scatter Plot: Actual Return vs. CAPM Predicted Return
def plot_scatter_actual_vs_capm(filtered_data, title, file_name):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    type_colors = {
        'Stocks': 'tab:blue',
        'ETFs': 'tab:green',
        'Mutual Funds': 'tab:orange'
    }
    
    for t in filtered_data['Type'].unique():
        subset = filtered_data[filtered_data['Type'] == t]
        ax.scatter(subset['CAPM Predicted Return'], subset['Actual Return'], label=t, alpha=0.7)
    
    ax.plot([filtered_data['CAPM Predicted Return'].min(), filtered_data['CAPM Predicted Return'].max()],
            [filtered_data['CAPM Predicted Return'].min(), filtered_data['CAPM Predicted Return'].max()], 
            ls="--", c=".3")
    
    ax.set_xlabel('CAPM Predicted Return', fontsize=18)
    ax.set_ylabel('Actual Return', fontsize=18)
    ax.set_title(title, fontsize=22)
    
    ax.legend(title='Security Types', fontsize=14, title_fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"Scatter plot saved as: {os.path.join(figure_directory, file_name)}")

# Bar Plot: Alpha of Top Assets
def plot_alpha_bar(filtered_data, title, file_name):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    filtered_data = filtered_data.sort_values(by='Alpha', ascending=False)
    
    ax.bar(filtered_data['Ticker'], filtered_data['Alpha'], color='b', alpha=0.7)
    
    ax.set_xlabel('Securities', fontsize=18)
    ax.set_ylabel('Alpha', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticklabels(filtered_data['Ticker'], rotation=90, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"Alpha bar plot saved as: {os.path.join(figure_directory, file_name)}")

# Bar Plot: Sharpe Ratios of Top Assets
def plot_sharpe_ratio_bar(filtered_data, title, file_name):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    filtered_data = filtered_data.sort_values(by='Sharpe Ratio', ascending=False)
    
    ax.bar(filtered_data['Ticker'], filtered_data['Sharpe Ratio'], color='g', alpha=0.7)
    
    ax.set_xlabel('Securities', fontsize=18)
    ax.set_ylabel('Sharpe Ratio', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticklabels(filtered_data['Ticker'], rotation=90, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"Sharpe ratio bar plot saved as: {os.path.join(figure_directory, file_name)}")

# Box Plot: VaR and CVaR
def plot_var_cvar_box(filtered_data, title, file_name):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    data_to_plot = [filtered_data['VaR'], filtered_data['CVaR']]
    ax.boxplot(data_to_plot, labels=['VaR', 'CVaR'])
    
    ax.set_title(title, fontsize=22)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, file_name), dpi=300)
    plt.close()
    print(f"VaR and CVaR box plot saved as: {os.path.join(figure_directory, file_name)}")

# Plot for each time horizon
plot_scatter_actual_vs_capm(top_assets_5_years, 
                    'Actual Return vs. CAPM Predicted Return (5 Years)', 
                    'actual_vs_capm_5_years.png')

plot_alpha_bar(top_assets_5_years, 
                    'Alpha of Top Assets (5 Years)', 
                    'alpha_top_assets_5_years.png')

plot_sharpe_ratio_bar(top_assets_5_years, 
                    'Sharpe Ratio of Top Assets (5 Years)', 
                    'sharpe_ratio_top_assets_5_years.png')

plot_var_cvar_box(top_assets_5_years, 
                    'VaR and CVaR of Top Assets (5 Years)', 
                    'var_cvar_top_assets_5_years.png')

plot_scatter_actual_vs_capm(top_assets_10_years, 
                    'Actual Return vs. CAPM Predicted Return (10 Years)', 
                    'actual_vs_capm_10_years.png')

plot_alpha_bar(top_assets_10_years, 
                    'Alpha of Top Assets (10 Years)', 
                    'alpha_top_assets_10_years.png')

plot_sharpe_ratio_bar(top_assets_10_years, 
                    'Sharpe Ratio of Top Assets (10 Years)', 
                    'sharpe_ratio_top_assets_10_years.png')

plot_var_cvar_box(top_assets_10_years, 
                    'VaR and CVaR of Top Assets (10 Years)', 
                    'var_cvar_top_assets_10_years.png')

plot_scatter_actual_vs_capm(top_assets_15_years, 
                    'Actual Return vs. CAPM Predicted Return (15 Years)', 
                    'actual_vs_capm_15_years.png')

plot_alpha_bar(top_assets_15_years, 
                    'Alpha of Top Assets (15 Years)', 
                    'alpha_top_assets_15_years.png')

plot_sharpe_ratio_bar(top_assets_15_years, 
                    'Sharpe Ratio of Top Assets (15 Years)', 
                    'sharpe_ratio_top_assets_15_years.png')

plot_var_cvar_box(top_assets_15_years, 
                    'VaR and CVaR of Top Assets (15 Years)', 
                    'var_cvar_top_assets_15_years.png')

print("Visualizations completed and saved.")
