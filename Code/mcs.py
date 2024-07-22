import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Load historical data
historical_data_path = "/Users/frank/Desktop/Project/Data/yfinance_data.csv"
historical_data = pd.read_csv(historical_data_path, parse_dates=['Date'])

# Load optimal weights
weights_10_years_path = "/Users/frank/Desktop/Project/Data/optimal_weights_10_years.csv"
weights_7_5_years_path = "/Users/frank/Desktop/Project/Data/optimal_weights_7_5_years.csv"
weights_5_years_path = "/Users/frank/Desktop/Project/Data/optimal_weights_5_years.csv"

weights_10_years = pd.read_csv(weights_10_years_path, index_col=0)
weights_7_5_years = pd.read_csv(weights_7_5_years_path, index_col=0)
weights_5_years = pd.read_csv(weights_5_years_path, index_col=0)

# Display the loaded data to verify
print("Historical Data:\n", historical_data.head())
print("10 Year Optimal Weights:\n", weights_10_years.head())
print("7.5 Year Optimal Weights:\n", weights_7_5_years.head())
print("5 Year Optimal Weights:\n", weights_5_years.head())

# Define the number of simulations and time horizon
num_simulations = 1000
end_date = '2024-07-22'

# Extract relevant columns from historical data
historical_data.set_index(['Date', 'Ticker'], inplace=True)
price_data = historical_data['Adj Close'].unstack(level='Ticker')

# Calculate daily returns
daily_returns = price_data.pct_change().dropna()

# Helper function to run a single simulation
def run_simulation(weights, daily_returns, num_days):
    returns = np.dot(daily_returns, weights)
    simulated_returns = np.cumprod(1 + np.random.choice(returns, size=num_days)) - 1
    return simulated_returns

# Function to run Monte Carlo simulation
def monte_carlo_simulation(weights, daily_returns, num_simulations, num_days):
    simulations = np.zeros((num_simulations, num_days))
    
    for i in range(num_simulations):
        simulations[i, :] = run_simulation(weights, daily_returns, num_days)
    
    return simulations

# Function to apply economic shocks
def apply_economic_shocks(simulated_portfolios, shock_intensity=-0.3, shock_probability=0.1):
    num_simulations, num_days = simulated_portfolios.shape
    for i in range(num_simulations):
        if np.random.rand() < shock_probability:
            shock_day = np.random.randint(0, num_days)
            simulated_portfolios[i, shock_day:] *= (1 + shock_intensity)
    return simulated_portfolios

# Define the time horizons and the corresponding optimal weights
time_horizons = {
    '10_years': (weights_10_years, 10),
    '7_5_years': (weights_7_5_years, 7.5),
    '5_years': (weights_5_years, 5)
}

# Define demographic data
median_age = 35
median_income = 104000

summary_stats = []

# Run simulations for each time horizon
for label, (weights_df, horizon) in time_horizons.items():
    optimal_weights = weights_df['Weight'].values
    num_days = int(horizon * 252)  # Approximate trading days in a year
    age_at_start = median_age - horizon
    
    # Align weights with daily_returns columns
    weights_series = pd.Series(optimal_weights, index=weights_df.index)
    daily_returns_aligned = daily_returns[weights_series.index]
    
    # Run the Monte Carlo simulation
    results = monte_carlo_simulation(weights_series.values, daily_returns_aligned, num_simulations, num_days)
    
    # Apply economic shocks
    shocked_results = apply_economic_shocks(results)
    
    # Final portfolio value calculation
    final_values = shocked_results[:, -1] * (median_income * 0.2)  # Assuming 20% annual contribution
    
    # Summary statistics
    mean_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    percentile_10 = np.percentile(final_values, 10)
    percentile_90 = np.percentile(final_values, 90)
    
    summary_stats.append({
        'Time Horizon': f"{horizon} Years",
        'Starting Age': age_at_start,
        'Mean Final Value': mean_final_value,
        'Median Final Value': median_final_value,
        '10th Percentile': percentile_10,
        '90th Percentile': percentile_90
    })
    
    # Visualization: Cumulative Returns Over Time
    plt.figure(figsize=(12, 8))
    for i in range(num_simulations):
        plt.plot(np.arange(num_days), shocked_results[i], color='lightgray', alpha=0.2)
    plt.plot(np.arange(num_days), np.median(shocked_results, axis=0), color='blue', linewidth=2, label='Median Simulation')
    plt.title(f'Cumulative Returns Over Time ({horizon} Years)', fontsize=20)
    plt.xlabel('Days', fontsize=16)
    plt.ylabel('Cumulative Return', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/Users/frank/Desktop/Project/Figures/cumulative_returns_over_time_{label}.png")
    plt.close()

    # Visualization: Distribution of Final Portfolio Values
    plt.figure(figsize=(12, 8))
    sns.histplot(final_values, bins=50, kde=False, color='skyblue', edgecolor='black')
    plt.axvline(mean_final_value, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_final_value, color='green', linestyle='dashed', linewidth=2, label='Median')
    plt.axvline(percentile_10, color='orange', linestyle='dashed', linewidth=2, label='10th Percentile')
    plt.axvline(percentile_90, color='orange', linestyle='dashed', linewidth=2, label='90th Percentile')
    plt.title(f'Distribution of Final Portfolio Values ({horizon} Years)', fontsize=20)
    plt.xlabel('Final Portfolio Value ($)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/Users/frank/Desktop/Project/Figures/final_portfolio_values_distribution_{label}.png")
    plt.close()

# Save summary statistics to a CSV file
summary_stats_df = pd.DataFrame(summary_stats)
summary_stats_df.to_csv('/Users/frank/Desktop/Project/Data/summary_stats.csv', index=False)

print("Summary statistics saved to /Users/frank/Desktop/Project/Data/summary_stats.csv")
