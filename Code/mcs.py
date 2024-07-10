import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf

# Load the data files
data_path = '/Users/frank/Desktop/Project/Data'
optimal_weights_5_years = pd.read_csv(f'{data_path}/optimal_weights_5_years.csv')
optimal_weights_7_5_years = pd.read_csv(f'{data_path}/optimal_weights_7_5_years.csv')
optimal_weights_10_years = pd.read_csv(f'{data_path}/optimal_weights_10_years.csv')

# Load yfinance data and filter for relevant securities
yfinance_data = pd.read_csv(f'{data_path}/yfinance_data.csv')
relevant_tickers = set(optimal_weights_5_years['Ticker']).union(set(optimal_weights_7_5_years['Ticker'])).union(set(optimal_weights_10_years['Ticker']))
yfinance_data = yfinance_data[yfinance_data['Ticker'].isin(relevant_tickers)]

# Calculate daily returns for each security
yfinance_data['Date'] = pd.to_datetime(yfinance_data['Date'])
yfinance_data.sort_values(by=['Ticker', 'Date'], inplace=True)
yfinance_data['Daily Return'] = yfinance_data.groupby('Ticker')['Adj Close'].pct_change()

# Pivot the data to get a matrix of returns
returns_data = yfinance_data.pivot(index='Date', columns='Ticker', values='Daily Return')
returns_data.dropna(inplace=True)

# Define a function to run the Monte Carlo simulation with progress tracking
def monte_carlo_simulation(returns, weights, num_simulations=1000, num_days=252*5):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    portfolio_returns = []
    simulated_paths = []

    for _ in tqdm(range(num_simulations), desc='Simulating portfolios'):
        simulated_daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, int(num_days))
        simulated_portfolio_returns = simulated_daily_returns.dot(weights)
        simulated_paths.append(np.cumprod(simulated_portfolio_returns + 1))
    
    return np.array(simulated_paths)

# Set the start date and end dates for different horizons
start_date = '2012-10-24'
end_date = '2014-11-10'  # Knowledge cutoff date
future_horizons = {
    '5_years': 252 * 5,
    '7.5_years': 252 * 7.5,
    '10_years': 252 * 10
}

# Extract the optimal weights for the different horizons
weights_5_years = optimal_weights_5_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values
weights_7_5_years = optimal_weights_7_5_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values
weights_10_years = optimal_weights_10_years.set_index('Ticker').reindex(returns_data.columns).fillna(0)['Weight'].values

# Run the Monte Carlo simulation for the different horizons from the knowledge cutoff date
returns = returns_data.loc[:end_date]

paths_5_years = monte_carlo_simulation(returns, weights_5_years, num_simulations=1000, num_days=future_horizons['5_years'])
paths_7_5_years = monte_carlo_simulation(returns, weights_7_5_years, num_simulations=1000, num_days=future_horizons['7.5_years'])
paths_10_years = monte_carlo_simulation(returns, weights_10_years, num_simulations=1000, num_days=future_horizons['10_years'])

# Obtain historical S&P 500 data up to the knowledge cutoff date
sp500 = yf.Ticker('^GSPC')
sp500_hist = sp500.history(start=start_date, end=end_date)

# Calculate daily returns for S&P 500
sp500_hist['Daily Return'] = sp500_hist['Close'].pct_change()
sp500_hist.dropna(inplace=True)

# Calculate cumulative returns for S&P 500
sp500_hist['Cumulative Return'] = (1 + sp500_hist['Daily Return']).cumprod()

# Combine historical returns with Monte Carlo simulation for visualization
def combine_hist_and_simulation(hist_returns, sim_paths, num_simulations=10):
    combined_paths = []
    for i in range(num_simulations):
        combined_path = np.concatenate((hist_returns, sim_paths[i]), axis=0)
        combined_paths.append(combined_path)
    return np.array(combined_paths)

# Calculate historical cumulative returns for portfolios up to the knowledge cutoff date
hist_cum_returns_5_years = (1 + returns.dot(weights_5_years)).cumprod().values
hist_cum_returns_7_5_years = (1 + returns.dot(weights_7_5_years)).cumprod().values
hist_cum_returns_10_years = (1 + returns.dot(weights_10_years)).cumprod().values

# Combine historical and simulated returns
combined_paths_5_years = combine_hist_and_simulation(hist_cum_returns_5_years, paths_5_years)
combined_paths_7_5_years = combine_hist_and_simulation(hist_cum_returns_7_5_years, paths_7_5_years)
combined_paths_10_years = combine_hist_and_simulation(hist_cum_returns_10_years, paths_10_years)

# Directory to save the figures
figures_path = '/Users/frank/Desktop/Project/Figures'

# Plot cumulative returns comparison
plt.figure(figsize=(14, 8))

# S&P 500 cumulative returns
plt.plot(sp500_hist.index, sp500_hist['Cumulative Return'], label='S&P 500', color='black', linewidth=2)

# Portfolio cumulative returns for different horizons
for path in combined_paths_5_years[:10]:  # plot first 10 paths
    plt.plot(pd.date_range(start=start_date, periods=len(path), freq='B'), path, color='blue', alpha=0.1)
plt.plot(pd.date_range(start=start_date, periods=len(combined_paths_5_years.mean(axis=0)), freq='B'), combined_paths_5_years.mean(axis=0), color='blue', label='5-Year Horizon Portfolio', linewidth=2)

for path in combined_paths_7_5_years[:10]:  # plot first 10 paths
    plt.plot(pd.date_range(start=start_date, periods=len(path), freq='B'), path, color='green', alpha=0.1)
plt.plot(pd.date_range(start=start_date, periods=len(combined_paths_7_5_years.mean(axis=0)), freq='B'), combined_paths_7_5_years.mean(axis=0), color='green', label='7.5-Year Horizon Portfolio', linewidth=2)

for path in combined_paths_10_years[:10]:  # plot first 10 paths
    plt.plot(pd.date_range(start=start_date, periods=len(path), freq='B'), path, color='red', alpha=0.1)
plt.plot(pd.date_range(start=start_date, periods=len(combined_paths_10_years.mean(axis=0)), freq='B'), combined_paths_10_years.mean(axis=0), color='red', label='10-Year Horizon Portfolio', linewidth=2)

plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{figures_path}/cumulative_returns_comparison.png')
plt.show()
