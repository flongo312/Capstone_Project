# Verify the installation of fredapi
try:
    from fredapi import Fred
    print("fredapi module is installed successfully.")
except ImportError:
    print("fredapi module is not installed.")

# Your main script
import pandas as pd
from fredapi import Fred

def fetch_and_clean_fred_data(api_key, series_ids, start_date, output_file):
    """
    Fetches economic data series from FRED, cleans it by removing columns with missing values,
    and saves the cleaned data to a CSV file.

    Parameters:
    api_key (str): FRED API key.
    series_ids (dict): Dictionary mapping FRED series IDs to their descriptions.
    start_date (str): The start date for fetching data (YYYY-MM-DD format).
    output_file (str): The path to the output CSV file.

    Returns:
    None
    """
    # Create a Fred object
    fred = Fred(api_key=api_key)

    # Initialize an empty DataFrame to store combined data
    df_combined = pd.DataFrame()

    # Fetch data for each series ID
    for series_id, description in series_ids.items():
        try:
            # Fetch the series data from FRED starting from start_date
            data = fred.get_series(series_id, start_date)
            # Add the series data to the combined DataFrame
            df_combined[description] = data
        except ValueError as e:
            # Print an error message if there's an issue fetching the data
            print(f"Error fetching {description}: {e}")

    # Ensure the DataFrame uses a common index with datetime format
    df_combined.index = pd.to_datetime(df_combined.index)

    # Remove columns with missing values
    df_cleaned = df_combined.dropna(axis=1)

    # Output the cleaned DataFrame to the specified CSV file
    df_cleaned.to_csv(output_file, index=True)

    print(f"Data has been successfully fetched, cleaned, and saved to {output_file}")

# Your FRED API key
api_key = 'b674819ad91362650c58d2d8acc27794'

# List of series IDs and their descriptions
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

# Define the start date for data fetching
start_date = '1999-01-01'

# Define the output CSV file path
output_file = 'fred_data.csv'

# Fetch, clean, and save the data
fetch_and_clean_fred_data(api_key, series_ids, start_date, output_file)
