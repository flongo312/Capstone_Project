import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
fred_data = pd.read_csv('fred_data.csv')
yfinance_data = pd.read_csv('yfinance_data.csv')

# Merge data
fred_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
yfinance_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
data = pd.merge(fred_data, yfinance_data, on='Date')

# Streamlit app
st.title('Optimal Investment Portfolio for First-Time Homebuyers')

# User inputs
age = st.number_input('Age', min_value=18, max_value=100, value=30)
risk_tolerance = st.selectbox('Risk Tolerance', ['Conservative', 'Moderate', 'Aggressive'])
investment_horizon = st.slider('Investment Horizon (years)', min_value=1, max_value=30, value=10)
initial_capital = st.number_input('Initial Capital ($)', min_value=1000, value=10000)

# Define risk tolerance weights
risk_weights = {'Conservative': [0.2, 0.3, 0.5], 'Moderate': [0.4, 0.3, 0.3], 'Aggressive': [0.6, 0.3, 0.1]}
weights = risk_weights[risk_tolerance]

# Define optimization function
def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns = portfolio_return(weights, mean_returns)
    volatility = portfolio_volatility(weights, cov_matrix)
    return - (returns - risk_free_rate) / volatility

# Calculate mean returns and covariance matrix
mean_returns = data[['Close']].pct_change().mean()
cov_matrix = data[['Close']].pct_change().cov()

# Initial guess and bounds
init_guess = np.array(weights)
bounds = ((0, 1), (0, 1), (0, 1))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Optimize portfolio
optimized = minimize(negative_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = optimized.x

# Display results
st.subheader('Optimal Portfolio Allocation')
st.write(f'Stocks: {opt_weights[0]:.2f}, Bonds: {opt_weights[1]:.2f}, Cash: {opt_weights[2]:.2f}')

# Plot portfolio allocation
labels = ['Stocks', 'Bonds', 'Cash']
fig, ax = plt.subplots()
ax.pie(opt_weights, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)
