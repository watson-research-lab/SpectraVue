"""
    Helper methods used for SpectraVue platform. 

    Contact: hamidtarek3@gmail.com
"""

import re
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def parse_spec_data_and_return_df(path_to_spec_file, start_time):
    '''
        Given path to spectrometer data, parse through JSON string and return a dataframe with values. Also calculates timestamp.

        Parameters:
            path_to_spec_file (string): relative path to spectrometer data file.
            start_time (string): start time of testing in 24H format (format: HH:MM:SS)

        Returns:
            df (dataframe): dataframe with spectrometer values.
    '''

    # Load data file
    with open(path_to_spec_file) as f:
        data = f.read().strip()

    # Use regex pattern matching to find JSON objects
    pattern = r'\{[^}]*\}'
    matches = re.findall(pattern, data)

    # Remove newline characters within JSON objects
    matches = [match.replace('\n', '') for match in matches]

    # Store in dataframe
    spec_df = pd.DataFrame([{k: v for k, v in eval(x).items()} for x in matches])

    # Check if sampling_time_ms exists, otherwise use default value
    default_sampling_time_ms = 384
    if 'sampling_time_ms' not in spec_df.columns:
        spec_df['sampling_time_ms'] = default_sampling_time_ms

    # Calculate time col
    start_time = datetime.strptime(start_time, "%H:%M:%S")
    time_col = [start_time + timedelta(milliseconds=sum(spec_df["sampling_time_ms"].iloc[:idx])) for idx in
                range(len(spec_df))]

    # Add time and timestamp columns
    spec_df.insert(0, 'Time', time_col)
    spec_df['Time'] = spec_df['Time'].apply(lambda x: x.time())
    spec_df.insert(1, 'Timestamp (s)', [(t - start_time).total_seconds() for t in time_col])

    # Drop unneeded cols
    spec_df = spec_df.drop('sampling_time_ms', axis=1)

    return spec_df

def flatten_spec_data(df):
    '''
        Given df containing spectrometer data, flatten data into single Lumos measurements.

        Parameters:
            df (dataframe): dataframe containing spectrometer photodiode values.

        Returns:
            flattened_df (dataframe): flattened dataframe with single Lumos measurements per row.
    '''

    # Find led order
    led_order = df['LED'].unique()
    led_order = led_order[~np.isnan(led_order)]

    # Define the start LED (assuming it's the first LED in led_order)
    start_led = led_order[0]
    
    # Add 'Cycle' column to dataframe
    cycle = 0
    df['Cycle'] = 0
    for i in range(1, len(df)):
        if df['LED'].iloc[i] == start_led:
            cycle += 1
        df.loc[i, 'Cycle'] = cycle

    # Identify incomplete cycles and remove them
    cycle_counts = df['Cycle'].value_counts()
    complete_cycles = cycle_counts[cycle_counts == len(led_order)].index
    df = df[df['Cycle'].isin(complete_cycles)]

    # Reset 'Cycle' values to be consecutive
    df['Cycle'] = np.repeat(range(len(df) // len(led_order)), len(led_order))

    # Group dataframe by LED cycles
    grouped_df = df.groupby('Cycle')

    # Regex to find cols - TODO: change to include Clear and NIR
    regex = '(F[1-9]_[0-9]+nm)|(Clear)|(NIR)'

    # Flatten each group structure
    extract_LED_values = lambda group: (
        group.filter(regex=regex)
        .values.reshape(1, -1)
        .flatten()
        .tolist()
    )

    # Apply lambda function
    LED_values_list = [extract_LED_values(group) for _, group in grouped_df]

    # Create flattened df
    flattened_df = pd.DataFrame(LED_values_list, columns=[f'LED{x}_PD{y}' for x in led_order for y in
                                                          ['415', '445', '480', '515', '555', '590', '630', '680',
                                                           'Clear', 'NIR']])

    # Use first values for Time and Timestamp (s)
    flattened_df['Time'] = grouped_df.first()['Time'].values
    flattened_df['Timestamp (s)'] = grouped_df.first()['Timestamp (s)'].values

    # Reorder columns to put Time and Timestamp (s) first
    flattened_df = flattened_df[
        ['Time', 'Timestamp (s)'] + [col for col in flattened_df.columns if col not in ['Time', 'Timestamp (s)']]]

    flattened_df = flattened_df.dropna()

    return flattened_df

def generate_synthetic_lumos_data(cycles):
    '''
        Given the number of cycles needed, generates synthetic Lumos in JSON format (for demo/testing purposes) as a text file
        in the current directory. 
        
        A random number is generated for each PD. Sampling time is static at 300ms per sample. 
        
        Parameters: 
            cycles (integer): the number of cycles to generate. A cycle is defined as data for each LED/PD. 
        
        Returns:
            N/A
    
    '''
    
    LED_list = [460, 470, 530, 940, 580, 600, 633, 670, 415]

    try:
        with open('synthetic_data.txt', 'w') as file:
            for _ in range(cycles):
                for LED in LED_list:
                    data = {
                        "F1_415nm": random.randint(100, 10000),
                        "F2_445nm": random.randint(100, 10000),
                        "F3_480nm": random.randint(100, 10000),
                        "F4_515nm": random.randint(100, 10000),
                        "F5_555nm": random.randint(100, 10000),
                        "F6_590nm": random.randint(100, 10000),
                        "F7_630nm": random.randint(100, 10000),
                        "F8_680nm": random.randint(100, 10000),
                        "Clear": random.randint(100, 10000),
                        "NIR": random.randint(100, 10000),
                        "LED": LED,
                        "sampling_time_ms": 300
                    }
                    file.write(json.dumps(data, indent=4))
                    file.write("\n")

    except Exception as e:
        print(f"An error occurred: {e}")