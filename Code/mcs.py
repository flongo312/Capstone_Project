import numpy as np
import matplotlib.pyplot as plt

# Parameters for Monte Carlo simulation
num_simulations = 1000
num_days = 252  # 1 year of trading days
drift = 0.001
volatility = 0.02

# Simulate stock price paths
np.random.seed(42)
simulated_paths = np.zeros((num_days, num_simulations))
simulated_paths[0] = 100  # Starting price

for t in range(1, num_days):
    random_shocks = np.random.normal(loc=drift, scale=volatility, size=num_simulations)
    simulated_paths[t] = simulated_paths[t-1] * np.exp(random_shocks)

# Plot the Monte Carlo simulation results
plt.figure(figsize=(14, 7))
plt.plot(simulated_paths, lw=0.5)
plt.xlabel('Day')
plt.ylabel('Simulated Stock Price')
plt.title('Monte Carlo Simulation of Stock Prices')
plt.show()
