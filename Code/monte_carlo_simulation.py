import numpy as np
import matplotlib.pyplot as plt

# Parameters for Monte Carlo simulation
S0 = 100  # initial stock price
mu = 0.05  # drift (average return)
sigma = 0.2  # volatility (risk)
T = 1.0  # time horizon
N = 252  # number of time steps
M = 1000  # number of simulations

# Time increment
dt = T / N

# Simulate M paths
paths = np.zeros((N + 1, M))
paths[0] = S0

for t in range(1, N + 1):
    Z = np.random.standard_normal(M)  # standard normal random variables
    paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Plot the first 10 simulated paths
plt.figure(figsize=(10, 6))
plt.plot(paths[:, :10])
plt.title('Monte Carlo Simulation of Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/monte_carlo_simulation.png')
plt.close()

