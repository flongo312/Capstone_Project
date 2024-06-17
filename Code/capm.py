import pandas as pd
import matplotlib.pyplot as plt


#Markowitz Model (Modern Portfolio Theory) 


# Load the data
fred_data = pd.read_csv('/mnt/data/fred_data.csv')
yfinance_data = pd.read_csv('/mnt/data/yfinance_data.csv')

# Rename 'Unnamed: 0' to 'Date'
fred_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
yfinance_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Merge the data on 'Date'
data = pd.merge(fred_data, yfinance_data, on='Date')