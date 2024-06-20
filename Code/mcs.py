import matplotlib.pyplot as plt

# Monte Carlo simulation
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_days, weights):
    results = np.zeros((num_simulations, 3))
    for i in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        portfolio_returns = np.dot(daily_returns, weights)
        sim_returns = np.sum(portfolio_returns)
        sim_std = np.std(portfolio_returns)
        results[i, 0] = sim_returns
        results[i, 1] = sim_std
        results[i, 2] = (sim_returns - risk_free_rate.mean() * num_days) / sim_std
    return results

num_simulations = 10000
num_days = 252  # 1 year

simulation_results = monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_days, optimal_weights)

plt.hist(simulation_results[:, 0], bins=50, alpha=0.7, label='Simulated Portfolio Returns')
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()
