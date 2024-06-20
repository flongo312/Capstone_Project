import numpy as np
from scipy.optimize import minimize

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Define functions for portfolio statistics
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate.mean()) / p_var

# Constraints: weights must sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(all_securities)))

# Initial guess
init_guess = len(all_securities) * [1. / len(all_securities)]

# Optimize portfolio
opt_results = minimize(neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate.mean()),
                       method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_results.x
print("Optimal weights:", optimal_weights)
