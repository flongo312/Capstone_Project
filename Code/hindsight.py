import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define colorblind-friendly palette
colors = {
    'black': '#000000',
    'blue': '#0072B2',
    'green': '#009E73',
    'red': '#D55E00',
    'orange': '#E69F00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'magenta': '#CC79A7',
    'yellow': '#F0E442',
    'brown': '#A52A2A'
}

# Load the data files
data_path = '/Users/frank/Desktop/Project/Data'
hindsight_data = pd.read_csv(f'{data_path}/hindsight_data.csv')

# Ensure the Date column is properly formatted
hindsight_data['Date'] = pd.to_datetime(hindsight_data['Date'])
hindsight_data.set_index('Date', inplace=True)

# Load the optimal weights files
optimal_weights_5_years = pd.read_csv(f'{data_path}/optimal_weights_5_years.csv')
optimal_weights_7_5_years = pd.read_csv(f'{data_path}/optimal_weights_7_5_years.csv')
optimal_weights_10_years = pd.read_csv(f'{data_path}/optimal_weights_10_years.csv')

# Calculate daily returns for each security
hindsight_data.sort_values(by=['Ticker', 'Date'], inplace=True)
hindsight_data['Daily Return'] = hindsight_data.groupby('Ticker')['Adj Close'].pct_change()

# Pivot the data to get a matrix of returns
returns_data = hindsight_data.pivot_table(index=hindsight_data.index, columns='Ticker', values='Daily Return')
returns_data.dropna(inplace=True)

# Align tickers in optimal weights data frames with those in the returns data
aligned_weights_5_years = optimal_weights_5_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values
aligned_weights_7_5_years = optimal_weights_7_5_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values
aligned_weights_10_years = optimal_weights_10_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values

# Calculate actual cumulative returns for the portfolio using hindsight data
actual_returns_5_years = returns_data.dot(aligned_weights_5_years).dropna()
actual_returns_7_5_years = returns_data.dot(aligned_weights_7_5_years).dropna()
actual_returns_10_years = returns_data.dot(aligned_weights_10_years).dropna()

actual_cum_returns_5_years = (1 + actual_returns_5_years).cumprod()
actual_cum_returns_7_5_years = (1 + actual_returns_7_5_years).cumprod()
actual_cum_returns_10_years = (1 + actual_returns_10_years).cumprod()

# Calculate cumulative returns for S&P 500
sp500_data = hindsight_data[hindsight_data['Ticker'] == '^GSPC'].copy()
sp500_data.loc[:, 'Cumulative Return'] = (1 + sp500_data['Daily Return']).cumprod()

# Directory to save the figures
figures_path = '/Users/frank/Desktop/Project/Figures'

# Plot cumulative returns comparison
plt.figure(figsize=(16, 10))

# S&P 500 cumulative returns using hindsight data
plt.plot(sp500_data.index, sp500_data['Cumulative Return'], label='S&P 500 (Benchmark)', color=colors['black'], linewidth=2, linestyle='-', zorder=10)

# Actual cumulative returns for different horizons
plt.plot(actual_cum_returns_5_years.index, actual_cum_returns_5_years, color=colors['cyan'], label='Actual 5-Year Horizon Portfolio', linestyle='-', linewidth=2)
plt.plot(actual_cum_returns_7_5_years.index, actual_cum_returns_7_5_years, color=colors['purple'], label='Actual 7.5-Year Horizon Portfolio', linestyle='-', linewidth=2)
plt.plot(actual_cum_returns_10_years.index, actual_cum_returns_10_years, color=colors['orange'], label='Actual 10-Year Horizon Portfolio', linestyle='-', linewidth=2)

# Improve readability and professionalism
plt.title('Cumulative Returns Comparison of Actual Portfolios vs. S&P 500', fontsize=24)
plt.xlabel('Date', fontsize=22)
plt.ylabel('Cumulative Return (%)', fontsize=22)  
plt.yticks(np.arange(0, 5.1, step=0.5), [f'{int(x*100)}%' for x in np.arange(0, 5.1, step=0.5)], fontsize=20)
plt.xticks(rotation=45, fontsize=20)
plt.legend(fontsize=20, loc='upper left')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout(pad=3)
plt.savefig(f'{figures_path}/cumulative_returns_comparison.png', bbox_inches='tight')
plt.show()

# Create a simplified comparison table summarizing cumulative returns
final_returns = {
    'S&P 500 (Benchmark)': (sp500_data['Cumulative Return'].iloc[-1] - 1) * 100,
    'Actual 5-Year Horizon Portfolio': (actual_cum_returns_5_years.iloc[-1] - 1) * 100,
    'Actual 7.5-Year Horizon Portfolio': (actual_cum_returns_7_5_years.iloc[-1] - 1) * 100,
    'Actual 10-Year Horizon Portfolio': (actual_cum_returns_10_years.iloc[-1] - 1) * 100
}

# Plot the comparison table for readability
comparison_table = pd.DataFrame(final_returns, index=[pd.to_datetime('2024-07-10')])
comparison_table_transposed = comparison_table.T

plt.figure(figsize=(16, 12))
bars = plt.barh(comparison_table_transposed.index, comparison_table_transposed[pd.to_datetime('2024-07-10')], color=[
    colors['black'], colors['cyan'], colors['purple'], colors['orange']
], alpha=0.7)
plt.xlabel('Cumulative Return (%)', fontsize=22)
plt.title('Cumulative Returns Summary 2011-05-04 to 2024-07-10', fontsize=24)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout(pad=3)

# Add labels to each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center', ha='left', fontsize=20)

plt.savefig(f'{figures_path}/cumulative_returns_summary.png', bbox_inches='tight')
plt.show()