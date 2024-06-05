import pandas as pd
from fredapi import Fred

# Your FRED API key
api_key = 'b674819ad91362650c58d2d8acc27794'

# Create a Fred object
fred = Fred(api_key=api_key)

# List of series IDs
series_ids = {
    'CPIAUCSL': 'Consumer Price Index (CPI)',
    'PPIACO': 'Producer Price Index (PPI)',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payroll Employment',
    'INDPRO': 'Industrial Production Index',
    'RSAFS': 'Retail Sales',
    'HOUST': 'Housing Starts',
    'PERMIT': 'Building Permits',
    'PI': 'Personal Income',
    'PCE': 'Personal Consumption Expenditures (PCE)',
    'TCU': 'Capacity Utilization',
    'BUSINV': 'Business Inventories',
    'DGORDER': 'Durable Goods Orders',
    'EXHOSLUSM495S': 'Existing Home Sales',
    'TTLCONS': 'Construction Spending',
    'BOPGSTB': 'Trade Balance',  # Corrected Trade Balance series ID
    'MRTSSM44X72USS': 'Manufacturing and Trade Sales',
    'TOTALSA': 'Vehicle Sales',
    'MANEMP': 'ISM Manufacturing Index',  # Changed from 'ISM/MAN_PMI' to 'MANEMP'
    'ECIALLCIV': 'Employment Cost Index (ECI)',
    'M1SL': 'Money Supply (M1)',
    'M2SL': 'Money Supply (M2)'
}

# Fetch the data and combine into a single DataFrame
df_combined = pd.DataFrame()

start_date = '1999-01-01'

for series_id, description in series_ids.items():
    try:
        data = fred.get_series(series_id, start_date)
        df_combined[description] = data
    except ValueError as e:
        print(f"Error fetching {description}: {e}")

# Ensure the DataFrame uses a common index
df_combined.index = pd.to_datetime(df_combined.index)

# Output to a CSV file
df_combined.to_csv('fred_data.csv', index=True)

print("Data has been successfully fetched and saved to fred_data.csv")
