import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the filtered top securities and optimal weights
file_path_5_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_5_years.csv'
file_path_10_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_10_years.csv'
file_path_15_years = '/Users/frank/Desktop/Project/Data/filtered_top_securities_15_years.csv'

optimal_weights_path_5_years = '/Users/frank/Desktop/Project/Data/optimal_weights_5_years.csv'
optimal_weights_path_10_years = '/Users/frank/Desktop/Project/Data/optimal_weights_10_years.csv'
optimal_weights_path_15_years = '/Users/frank/Desktop/Project/Data/optimal_weights_15_years.csv'

filtered_top_securities_5_years = pd.read_csv(file_path_5_years)
filtered_top_securities_10_years = pd.read_csv(file_path_10_years)
filtered_top_securities_15_years = pd.read_csv(file_path_15_years)

optimal_weights_5_years = pd.read_csv(optimal_weights_path_5_years)
optimal_weights_10_years = pd.read_csv(optimal_weights_path_10_years)
optimal_weights_15_years = pd.read_csv(optimal_weights_path_15_years)

# Function to merge and display the combined data
def display_combined_data(filtered_data, optimal_weights, title):
    combined_data = filtered_data.merge(optimal_weights, on='Ticker')
    combined_data_sorted = combined_data.sort_values(by='Weight', ascending=False)
    combined_data_sorted['Weight'] = combined_data_sorted['Weight'].round(4)
    
    print(title)
    print(combined_data_sorted)
    
    return combined_data_sorted

# 5-Year Horizon
combined_5_years = display_combined_data(filtered_top_securities_5_years, optimal_weights_5_years, '5-Year Horizon')

# 10-Year Horizon
combined_10_years = display_combined_data(filtered_top_securities_10_years, optimal_weights_10_years, '10-Year Horizon')

# 15-Year Horizon
combined_15_years = display_combined_data(filtered_top_securities_15_years, optimal_weights_15_years, '15-Year Horizon')

# Annual income and contribution details
income_contributions = {
    '20-25': {'income': 45000, 'contribution': 0.10, 'years': 5},
    '25-30': {'income': 60000, 'contribution': 0.15, 'years': 5},
    '30-35': {'income': 80000, 'contribution': 0.20, 'years': 5}
}

# Monte Carlo simulation
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_years, weights, annual_contribution):
    num_days = num_years * 252  # 252 trading days per year
    results = np.zeros((num_simulations, num_years + 1))  # Storing end value of each year

    for i in range(num_simulations):
        portfolio_value = 0
        for year in range(num_years):
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, 252)
            portfolio_returns = np.dot(daily_returns, weights)
            annual_return = np.prod(1 + portfolio_returns) - 1
            portfolio_value = (portfolio_value + annual_contribution) * (1 + annual_return)
            results[i, year + 1] = portfolio_value

    return results

# Combined function to simulate for different age brackets
def simulate_portfolio_growth(mean_returns, cov_matrix, num_simulations, weights, income_contributions):
    all_results = []
    
    for age_bracket, details in income_contributions.items():
        annual_income = details['income']
        annual_contribution = annual_income * details['contribution']
        num_years = details['years']
        
        results = monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_years, weights, annual_contribution)
        all_results.append((age_bracket, results))
        
    return all_results

# Example mean returns and covariance matrix for the simulation
selected_tickers = combined_5_years[combined_5_years['Weight'] > 0]['Ticker']

# Calculate mean returns and covariance matrix
filtered_selected_data = filtered_top_securities_5_years.set_index('Ticker').loc[selected_tickers, ['CAPM Predicted Return', 'Actual Return']]
mean_returns = filtered_selected_data.mean(axis=1).values
cov_matrix = np.cov(filtered_selected_data.T.astype(float))

# Optimal weights for 5-year horizon
optimal_weights = combined_5_years[combined_5_years['Weight'] > 0]['Weight'].values

# Number of simulations
num_simulations = 10000

# Simulate the portfolio growth
portfolio_simulations = simulate_portfolio_growth(mean_returns, cov_matrix, num_simulations, optimal_weights, income_contributions)

# Plotting the results
figure_directory = '/Users/frank/Desktop/Project/Figures'
os.makedirs(figure_directory, exist_ok=True)

# Histogram of Portfolio Values
for age_bracket, results in portfolio_simulations:
    plt.figure(figsize=(10, 6))
    plt.hist(results[:, -1], bins=50, alpha=0.7, label=f'{age_bracket} Portfolio End Values')
    plt.title(f'Monte Carlo Simulation of Portfolio Growth ({age_bracket})')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(figure_directory, f'{age_bracket}_portfolio_hist.png'))
    plt.close()
    print(f"{age_bracket} Portfolio End Values histogram saved as: {os.path.join(figure_directory, f'{age_bracket}_portfolio_hist.png')}")

# Line Plot of Median Portfolio Growth
for age_bracket, results in portfolio_simulations:
    median_growth = np.median(results, axis=0)
    plt.plot(range(len(median_growth)), median_growth, label=f'{age_bracket}')

plt.title('Median Portfolio Growth Over Time')
plt.xlabel('Years')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figure_directory, 'median_portfolio_growth.png'))
plt.close()
print(f"Median Portfolio Growth Over Time plot saved as: {os.path.join(figure_directory, 'median_portfolio_growth.png')}")

# Box Plot of Portfolio Values at the End of Each Age Bracket
end_values = []
labels = []

for age_bracket, results in portfolio_simulations:
    end_values.append(results[:, -1])
    labels.append(age_bracket)

plt.figure(figsize=(10, 6))
sns.boxplot(data=end_values)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.title('Portfolio End Values Distribution')
plt.xlabel('Age Bracket')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.savefig(os.path.join(figure_directory, 'portfolio_end_values_boxplot.png'))
plt.close()
print(f"Portfolio End Values Distribution box plot saved as: {os.path.join(figure_directory, 'portfolio_end_values_boxplot.png')}")
