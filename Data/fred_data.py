import requests
import pandas as pd
from fredapi import Fred

# Your FRED API key
api_key = 'b674819ad91362650c58d2d8acc27794'

# Base URL for FRED API
base_url = 'https://api.stlouisfed.org/fred/series'

def get_all_series(api_key, base_url):
    params = {
        'api_key': api_key,
        'file_type': 'json',
        'limit': 1000,  # Max limit per request
        'order_by': 'observation_start',  # Order by the start date of observation
        'sort_order': 'asc'  # Sort in ascending order to get earliest dates first
    }

    all_series = []
    offset = 0

    while True:
        params['offset'] = offset
        response = requests.get(base_url, params=params)
        data = response.json()

        series_list = data.get('seriess', [])
        if not series_list:
            break

        all_series.extend(series_list)
        offset += len(series_list)

    return all_series

def find_earliest_common_start_date(series_list):
    earliest_start_date = '9999-12-31'  # Initialize with a far future date
    for series in series_list:
        start_date = series['observation_start']
        if start_date < earliest_start_date:
            earliest_start_date = start_date

    earliest_series = [s for s in series_list if s['observation_start'] == earliest_start_date]
    return earliest_start_date, earliest_series

def fetch_and_clean_fred_data(api_key, series_ids, start_date, output_file):
    """
    Fetches economic data series from FRED, cleans it by removing columns with missing values,
    and saves the cleaned data to a CSV file.

    Parameters:
    api_key (str): FRED API key.
    series_ids (list): List of FRED series IDs.
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
    for series_id in series_ids:
        try:
            # Fetch the series data from FRED starting from start_date
            data = fred.get_series(series_id, start_date)
            # Add the series data to the combined DataFrame
            df_combined[series_id] = data
        except ValueError as e:
            # Print an error message if there's an issue fetching the data
            print(f"Error fetching {series_id}: {e}")

    # Ensure the DataFrame uses a common index with datetime format
    df_combined.index = pd.to_datetime(df_combined.index)

    # Remove columns with missing values
    df_cleaned = df_combined.dropna(axis=1)

    # Output the cleaned DataFrame to the specified CSV file
    df_cleaned.to_csv(output_file, index=True)

    print(f"Data has been successfully fetched, cleaned, and saved to {output_file}")

# Retrieve all series
all_series = get_all_series(api_key, base_url)

# Find the earliest common start date and the series that have it
earliest_start_date, earliest_series = find_earliest_common_start_date(all_series)

# Extract series IDs from the earliest series
earliest_series_ids = [s['id'] for s in earliest_series]

if earliest_series_ids:
    # Fetch, clean, and save the data from the earliest series
    fetch_and_clean_fred_data(api_key, earliest_series_ids, earliest_start_date, 'fred_data.csv')

print(f"Earliest Start Date: {earliest_start_date}")
print(f"Series IDs with the earliest start date: {earliest_series_ids}")
