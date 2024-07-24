import subprocess
import sys
import os

# Function to install packages
def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "pandas", 
    "numpy", 
    "matplotlib", 
    "seaborn", 
    "tqdm"
]

# Install required packages
install_packages(required_packages)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Start timer
start_time = time.time()

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define data and figures paths
data_path = os.path.join(script_dir, '../Data')
figures_path = os.path.join(script_dir, '../Figures')

# Ensure the directories exist
os.makedirs(data_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load historical data
historical_data_path = os.path.join(data_path, "yfinance_data.csv")
historical_data = pd.read_csv(historical_data_path, parse_dates=['Date'])

# Load optimal weights
weights_10_years_path = os.path.join(data_path, "optimal_weights_10_years.csv")
weights_7_5_years_path = os.path.join(data_path, "optimal_weights_7_5_years.csv")
weights_5_years_path = os.path.join(data_path, "optimal_weights_5_years.csv")

weights_10_years = pd.read_csv(weights_10_years_path, index_col=0)
weights_7_5_years = pd.read_csv(weights_7_5_years_path, index_col=0)
weights_5_years = pd.read_csv(weights_5_years_path, index_col=0)

# Define the number of simulations and time horizon
num_simulations = 1000
end_date = '2024-07-22'

# Extract relevant columns from historical data
historical_data.set_index(['Date', 'Ticker'], inplace=True)
price_data = historical_data['Adj Close'].unstack(level='Ticker')

# Calculate daily returns
daily_returns = price_data.pct_change().dropna()

# Helper function to run a single simulation
def run_simulation(weights, daily_returns, num_days, initial_contribution, continuous_contribution):
    returns = np.dot(daily_returns, weights)
    portfolio_values = np.zeros(num_days)
    portfolio_values[0] = initial_contribution * (1 + np.random.choice(returns))
    
    for day in range(1, num_days):
        portfolio_values[day] = portfolio_values[day-1] * (1 + np.random.choice(returns)) + continuous_contribution

    return portfolio_values

# Function to run Monte Carlo simulation
def monte_carlo_simulation(weights, daily_returns, num_simulations, num_days, initial_contribution, continuous_contribution):
    simulations = np.zeros((num_simulations, num_days))
    
    for i in tqdm(range(num_simulations)):
        simulations[i, :] = run_simulation(weights, daily_returns, num_days, initial_contribution, continuous_contribution)
    
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
    '10_years': (weights_10_years, 10, 29.82),
    '7_5_years': (weights_7_5_years, 7.5, 30.53),
    '5_years': (weights_5_years, 5, 29.17)
}

# Define demographic data
initial_contribution = 10000  # Initial contribution amount

summary_stats = []

# Run simulations for each time horizon
for label, (weights_df, horizon, continuous_contribution) in time_horizons.items():
    optimal_weights = weights_df['Weight'].values
    num_days = int(horizon * 252)  # Approximate trading days in a year
    
    # Align weights with daily_returns columns
    weights_series = pd.Series(optimal_weights, index=weights_df.index)
    daily_returns_aligned = daily_returns[weights_series.index]
    
    # Run the Monte Carlo simulation
    results = monte_carlo_simulation(weights_series.values, daily_returns_aligned, num_simulations, num_days, initial_contribution, continuous_contribution)
    
    # Apply economic shocks
    shocked_results = apply_economic_shocks(results)
    
    # Final portfolio value calculation
    final_values = shocked_results[:, -1]
    
    # Summary statistics
    mean_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    percentile_10 = np.percentile(final_values, 10)
    percentile_90 = np.percentile(final_values, 90)
    
    # Calculate annual percentage yield (APY) and total percentage yield
    total_yield = (mean_final_value / initial_contribution) - 1
    annual_yield = (1 + total_yield) ** (1 / horizon) - 1
    
    summary_stats.append({
        'Time Horizon (Years)': f"{horizon}",
        'Mean Final Portfolio Value ($)': round(mean_final_value, 2),
        'Median Final Portfolio Value ($)': round(median_final_value, 2),
        '10th Percentile Final Portfolio Value ($)': round(percentile_10, 2),
        '90th Percentile Final Portfolio Value ($)': round(percentile_90, 2),
        'Total Percentage Yield (%)': round(total_yield * 100, 2),
        'Annual Percentage Yield (%)': round(annual_yield * 100, 2)
    })
    
    # Enhanced Visualization: Cumulative Returns Over Time
    plt.figure(figsize=(14, 8))
    for i in range(num_simulations):
        plt.plot(np.arange(num_days), shocked_results[i], color='lightgray', alpha=0.1)
    plt.plot(np.arange(num_days), np.median(shocked_results, axis=0), color='blue', linewidth=2, label='Median Simulation')
    plt.title(f'Cumulative Returns Over Time ({horizon} Years)', fontsize=22, weight='bold')
    plt.xlabel('Days', fontsize=18)
    plt.ylabel('Cumulative Return', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"cumulative_returns_over_time_{label}.png"))
    plt.close()

    # Enhanced Visualization: Distribution of Final Portfolio Values
    plt.figure(figsize=(14, 8))
    sns.histplot(final_values, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(mean_final_value, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_final_value, color='green', linestyle='dashed', linewidth=2, label='Median')
    plt.axvline(percentile_10, color='orange', linestyle='dashed', linewidth=2, label='10th Percentile')
    plt.axvline(percentile_90, color='orange', linestyle='dashed', linewidth=2, label='90th Percentile')
    plt.title(f'Distribution of Final Portfolio Values ({horizon} Years)', fontsize=22, weight='bold')
    plt.xlabel('Final Portfolio Value ($)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"final_portfolio_values_distribution_{label}.png"))
    plt.close()

# Save summary statistics to a CSV file
summary_stats_df = pd.DataFrame(summary_stats)
summary_stats_df.to_csv(os.path.join(data_path, 'summary_stats.csv'), index=False)

print("Summary statistics saved to ../Data/summary_stats.csv")

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
