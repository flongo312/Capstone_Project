import numpy as np
import matplotlib.pyplot as plt

# Parameters for lifecycle investing
T = 30  # total investment period in years
alpha = 0.07  # expected return
beta = 0.15  # risk measure

# Calculate allocation over time
time = np.arange(1, T + 1)
allocation = (1 / (T - time + 1)) * (alpha / beta)

# Plot the allocation strategy
plt.figure(figsize=(10, 6))
plt.plot(time, allocation, label='Dynamic Asset Allocation')
plt.xlabel('Years to Target Date')
plt.ylabel('Asset Allocation Proportion')
plt.title('Lifecycle Investing Strategy')
plt.legend()

# Save the figure
plt.savefig('/Users/frank/Desktop/Project/Figures/lifecycle_investing_strategy.png')
plt.close()

