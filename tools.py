"""
    Helper methods used for SpectraVue platform. 

    Tarek Hamid, hamidtarek3@gmail.com
"""

import os
import re
import json
import random

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_synthetic_lumos_data(cycles):
    '''
        Given the number of cycles needed, generates synthetic Lumos in txt/JSON format (for demo/testing purposes) as a text file
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

def process_spec_csv(data):
   '''
        Given a dataframe with user uploaded txt data, process and return processed dataframe.

        The difference between txt and csv uploaded data is the data format. Text data is usually straight from the device, which is in JSON format. CSV is usually downloaded
        from the server, which is in standard csv format. 

        Parameters:
            data (list of dicts): input that is read from user uploaded csv file as a dataframe.
        
        Returns: 
            df (dataframe): processed dataframe with data. 
    '''
   data = json.loads(data)
   df = pd.DataFrame(data)
   df['Timestamp (s)'] = (df['Timestamp'] - df['Timestamp'].iloc[0])

   df = df.drop(['Unnamed: 0', 'Timestamp'], axis=1)

   return df

def process_spec_txt(data):
    '''
        Given a str with user uploaded txt data, process and return processed dataframe.

        The difference between txt and csv uploaded data is the data format. Text data is usually straight from the device, which is in JSON format. CSV is usually downloaded
        from the server, which is in standard csv format. 

        Parameters:
            data (str): input that is read from user uploaded txt file. 
        
        Returns: 
            processed_data (dataframe): processed dataframe with data. 
    '''

    # Use regex pattern matching to find JSON objects
    pattern = r'\{[^}]*\}'
    matches = re.findall(pattern, data)

    # Remove newline characters within JSON objects
    matches = [match.replace('\n', '') for match in matches]

    # Store in dataframe
    processed_df = pd.DataFrame([{k:v for k,v in eval(x).items()} for x in matches])

    # Check if sampling_time_ms exists, otherwise use default value
    default_sampling_time_ms = 384
    if 'sampling_time_ms' not in processed_df.columns:
        processed_df['sampling_time_ms'] = default_sampling_time_ms

    # Calculate timestamp col
    processed_df['Timestamp (s)'] = (processed_df['sampling_time_ms'].cumsum() - processed_df['sampling_time_ms']) / 1000

    # Drop unneeded cols
    processed_df = processed_df.drop('sampling_time_ms', axis=1)

    return processed_df

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
    led_order = np.char.mod('%d', led_order)

    # Group dataframe by 9 rows (cycled LED counts, including 9th LED)
    grouped_df = df.groupby(np.arange(len(df)) // len(led_order))

    # Regex to find cols - TODO: change to include Clear and NIR
    regex = '(F[1-9]_[0-9]+nm)|(Clear)|(NIR)'

    # Flatten each group structure
    extract_LED_values = lambda group: (
        group.filter(regex=regex)
        .values.reshape(1, -1)
        .flatten()
        .tolist()
    )

    # Apply lambda fnc
    LED_values_list = grouped_df.apply(extract_LED_values).tolist()

    # Create flattened df
    flattened_df = pd.DataFrame(LED_values_list, columns=[f'LED{x}_PD{y}' for x in led_order for y in ['415', '445', '480', '515', '555', '590', '630', '680', 'Clear', 'NIR']])

    # Use first values for Time and Timestamp (s)
    flattened_df['Timestamp (s)'] = grouped_df.first()['Timestamp (s)'].values

    # Reorder columns to put Time and Timestamp (s) first
    flattened_df = flattened_df[['Timestamp (s)'] + [col for col in flattened_df.columns if col not in ['Timestamp (s)']]]
    
    # Drop NaNs (TODO: replace with interpolation or padding?)
    flattened_df = flattened_df.dropna()
    
    return flattened_df

def get_mean_counts_per_LED(df):
    '''
    Compiles mean counts for each wavelength from medium data. 

        Parameters:
            df (df): dataframe containing timestamp and counts per LED. 
        
        Returns: 
            temp_df (df): processed dataframe with mean counts for each wavelength. 
    '''

    # Drop nulls
    temp_df = df[df['LED'].notna()].copy()

    # Convert LED col to int
    temp_df['LED'] = temp_df['LED'].astype(int)

    # Drop timestamp col
    temp_df = temp_df.drop(['Timestamp (s)'], axis=1)

    # Group by LED and calculate the mean of each group
    temp_df = temp_df.groupby('LED').mean()

    return temp_df

def static_graph_output(processed_df):
    '''
        Given a processed df, return a static graph with mean counts per LED. 

        Parameters: 
            processed_df (df): dataframe with processed uploaded user data.
        
        Returns:
            fig (Plotly figure): plotly static data figure. 
    '''
    # Get mean counts per LED
    processed_df = get_mean_counts_per_LED(processed_df)

    # Drop 'Clear' and 'NIR' columns
    processed_df = processed_df.drop(['Clear', 'NIR'], axis=1)
    
    # Choose method of interpolation
    method = 'cubic'
    processed_df = processed_df.reset_index().melt(id_vars='LED', var_name='F', value_name='Value')
    X = processed_df['LED'].values
    Y = processed_df['F'].apply(lambda x: int(x.split('_')[1].replace('nm', ''))).values
    Z = processed_df['Value'].values

    # Find mins and maxes
    x_min = X.min()
    x_max = X.max()
    y_min = Y.min()
    y_max = Y.max()

    # Create linearly spaced arrays between min and maxes
    x_new = np.linspace(x_min, x_max, 600)
    y_new = np.linspace(y_min, y_max, 600)

    # Construct new axises
    z_new = griddata((X, Y), Z, (x_new[None,:], y_new[:,None]), method=method)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    fig = go.Figure(data=[go.Surface(x=x_new_grid, y=y_new_grid, z=z_new, colorscale='Viridis')])
    
    fig.update_layout(
        title='Spectral Response - Static Plot',
        scene=dict(
            xaxis_title='LED',
            yaxis_title='PD',
            zaxis_title='Response (counts)'
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    return fig


def animation_graph_output(df):
    '''
    Given a processed df, generate an animated graph with counts per LED-PD combination per timestamp.

    Parameters: 
        df (DataFrame): Dataframe with processed uploaded user data.
    
    Returns:
        video_filename (str): Filename of the generated video file.
    '''

    # Drop NIR and Clear
    cols_to_drop = [col for col in df.columns if 'NIR' in col or 'Clear' in col]
    df = df.drop(cols_to_drop, axis=1)

    # Cycle starts with the first LED value in the first row
    first_led = df['LED'][0]
    led_cycle = df['LED'].unique()
    cycle_start_indices = df[df['LED'] == first_led].index
    df['Timestamp (s)'] = df.loc[cycle_start_indices, 'Timestamp (s)'].repeat(len(led_cycle)).values
    
    # Create a new DataFrame for unpacked data
    value_vars = [f'F{i}_{j}nm' for i, j in zip(range(1, 9), [415, 445, 480, 515, 555, 590, 630, 680])]
    unpacked_df = pd.melt(df, id_vars=['LED', 'Timestamp (s)'], value_vars=value_vars, var_name='PD', value_name='response')

    # Extract PD and LED values from strings to integer format
    unpacked_df['PD'] = unpacked_df['PD'].apply(lambda x: int(x.split('_')[1].replace('nm', '')))
    unpacked_df['LED'] = unpacked_df['LED'].astype(int)

    # Compute maxs of each axis
    Response_max = unpacked_df['response'].max()

    # Create a 3D surface plot using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_graph(timestamp):
        timestamp_df = unpacked_df[unpacked_df['Timestamp (s)'] == timestamp]
        timestamp_df = timestamp_df.pivot_table(index='PD', columns='LED', values='response')
        
        # Perform cubic interpolation using griddata
        x_grid, y_grid = np.meshgrid(timestamp_df.columns, timestamp_df.index)
        z_grid = timestamp_df.values
        x_interp = np.linspace(timestamp_df.columns.min(), timestamp_df.columns.max(), 100)
        y_interp = np.linspace(timestamp_df.index.min(), timestamp_df.index.max(), 100)
        x_interp_grid, y_interp_grid = np.meshgrid(x_interp, y_interp)
        z_interp = griddata((x_grid.flatten(), y_grid.flatten()), z_grid.flatten(), (x_interp_grid, y_interp_grid), method='cubic')

        # Compute the baseline for this frame and subtract from z_interp
        baseline = np.mean(z_interp)
        z_interp_normalized = z_interp - baseline

        # Plot the interpolated surface
        ax.clear()
        ax.set_title(f'Timestamp: {timestamp: .2f}')
        ax.plot_surface(y_interp_grid, x_interp_grid, z_interp_normalized, cmap='viridis')
        
        # Set labels
        ax.set_xlabel('PD')
        ax.set_ylabel('LED')
        ax.set_zlabel('Response (counts)')

        # Set same limits throughout the frames
        ax.set_zlim([-Response_max, Response_max])

    # Create animation frames
    timestamps = unpacked_df['Timestamp (s)'].unique()
    ani = FuncAnimation(fig, update_graph, frames=timestamps, repeat=True)

    # Save the animation as a video file
    video_filename = os.path.join('assets', 'animation.mp4')
    ani.save(video_filename, writer='ffmpeg')
    
    return video_filename

def animation_graph_biomarker(df):
    '''
    Given a processed df, generate an animated graph with counts per LED-PD combination per timestamp and a line plot of biomarker data over time.

    Parameters: 
        df (DataFrame): Dataframe with processed uploaded user data.
    
    Returns:
        video_filename (str): Filename of the generated video file.
    '''

    # Get the second to last column name from the DataFrame
    biomarker = df.columns[-2]

    # Drop NIR/Clear cols
    cols_to_drop = [col for col in df.columns if 'NIR' in col or 'Clear' in col]
    df = df.drop(cols_to_drop, axis=1)

    first_led = df['LED'][0]
    led_cycle = df['LED'].unique()
    cycle_start_indices = df[df['LED'] == first_led].index
    df['Timestamp (s)'] = df.loc[cycle_start_indices, 'Timestamp (s)'].repeat(len(led_cycle)).values

    # Process biomarker data: Grab the first value in each cycle
    df[biomarker] = df.loc[cycle_start_indices, biomarker].repeat(len(led_cycle)).values

    # Create a new DataFrame for unpacked data
    value_vars = [f'F{i}_{j}nm' for i, j in zip(range(1, 9), [415, 445, 480, 515, 555, 590, 630, 680])]
    unpacked_df = pd.melt(df, id_vars=['LED', 'Timestamp (s)', biomarker], value_vars=value_vars, var_name='PD', value_name='response')
    
    unpacked_df['PD'] = unpacked_df['PD'].apply(lambda x: int(x.split('_')[1].replace('nm', '')))
    unpacked_df['LED'] = unpacked_df['LED'].astype(int)

    Response_max = unpacked_df['response'].max()
    Biomarker_max = unpacked_df[biomarker].max()
    
    # Create figure
    fig = plt.figure(figsize=(12,4))

    # Create 3D surface plot on the first subplot
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Spectrometer Data") 

    # Create line plot on the second subplot
    ax2 = fig.add_subplot(122)
    ax2.set_title("Biomarker Data")

    def update_graph(timestamp):
        # Update 3D surface plot
        timestamp_df = unpacked_df[unpacked_df['Timestamp (s)'] == timestamp]
        timestamp_df = timestamp_df.pivot_table(index='PD', columns='LED', values='response')
        
        x_grid, y_grid = np.meshgrid(timestamp_df.columns, timestamp_df.index)
        z_grid = timestamp_df.values
        x_interp = np.linspace(timestamp_df.columns.min(), timestamp_df.columns.max(), 100)
        y_interp = np.linspace(timestamp_df.index.min(), timestamp_df.index.max(), 100)
        x_interp_grid, y_interp_grid = np.meshgrid(x_interp, y_interp)
        z_interp = griddata((x_grid.flatten(), y_grid.flatten()), z_grid.flatten(), (x_interp_grid, y_interp_grid), method='cubic')

        baseline = np.mean(z_interp)
        z_interp_normalized = z_interp - baseline

        ax.clear()
        ax.plot_surface(x_interp_grid, y_interp_grid, z_interp_normalized, cmap='viridis')
        ax.set_title(f'Timestamp: {timestamp: .2f}')
        ax.set_xlabel('LED')
        ax.set_ylabel('PD')
        ax.set_zlabel('Response (counts)')
        ax.set_zlim([-Response_max, Response_max])

        # Update line plot
        ax2.clear()
        ax2.plot(unpacked_df['Timestamp (s)'].unique(), unpacked_df.groupby('Timestamp (s)')[biomarker].first().values, marker='o')
        ax2.set_xlabel('Timestamp (s)')
        ax2.set_ylabel(biomarker)
        ax2.set_ylim([0, Biomarker_max])

        # Mark current position on the line plot
        current_biomarker = unpacked_df[unpacked_df['Timestamp (s)'] == timestamp][biomarker].values[0]
        ax2.plot([timestamp], [current_biomarker], 'ro')

    timestamps = unpacked_df['Timestamp (s)'].unique()
    ani = FuncAnimation(fig, update_graph, frames=timestamps, repeat=True)

    # Adjust the layout
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    video_filename = os.path.join('assets', 'animation_biomarker.mp4')
    ani.save(video_filename, writer='ffmpeg')

    plt.tight_layout() 
    
    return video_filename

"""

This fnc was used to generate interactive Plotly charts. However, for large time series data, it took way too long to render.

def animation_graph_output(df):
    '''
    Given a processed df, generate an animated graph with counts per LED-PD combination per timestamp.

    Parameters: 
        df (DataFrame): Dataframe with processed uploaded user data.
    
    Returns:
        fig (Plotly figure): Plotly figure object for displaying the animated graph.
        video_filename (str): Filename of the generated video file.
    '''

    # Drop NIR and Clear
    cols_to_drop = [col for col in df.columns if 'NIR' in col or 'Clear' in col]
    df = df.drop(cols_to_drop, axis=1)

    # Cycle starts with the first LED value in the first row
    first_led = df['LED'][0]
    led_cycle = df['LED'].unique()
    cycle_start_indices = df[df['LED'] == first_led].index
    df['Timestamp (s)'] = df.loc[cycle_start_indices, 'Timestamp (s)'].repeat(len(led_cycle)).values
    
    # Create a new DataFrame for unpacked data
    value_vars = [f'F{i}_{j}nm' for i, j in zip(range(1, 9), [415, 445, 480, 515, 555, 590, 630, 680])]
    unpacked_df = pd.melt(df, id_vars=['LED', 'Timestamp (s)'], value_vars=value_vars, var_name='PD', value_name='response')

    # Extract PD and LED values from strings to integer format
    unpacked_df['PD'] = unpacked_df['PD'].apply(lambda x: int(x.split('_')[1].replace('nm', '')))
    unpacked_df['LED'] = unpacked_df['LED'].astype(int)

    # Create 3D surface plot
    fig = go.Figure()

    for timestamp in unpacked_df['Timestamp (s)'].unique():
        timestamp_df = unpacked_df[unpacked_df['Timestamp (s)'] == timestamp]
        timestamp_df = timestamp_df.pivot_table(index='PD', columns='LED', values='response')
        
        # Create a grid of points for cubic interpolation
        x_grid, y_grid = np.meshgrid(timestamp_df.columns, timestamp_df.index)
        z_grid = timestamp_df.values

        # Define the desired grid for interpolated surface
        x_interp = np.linspace(timestamp_df.columns.min(), timestamp_df.columns.max(), 100)
        y_interp = np.linspace(timestamp_df.index.min(), timestamp_df.index.max(), 100)
        x_interp_grid, y_interp_grid = np.meshgrid(x_interp, y_interp)

        # Perform cubic interpolation
        z_interp = griddata((x_grid.flatten(), y_grid.flatten()), z_grid.flatten(), (x_interp_grid, y_interp_grid), method='cubic')

        fig.add_trace(go.Surface(x=x_interp_grid, y=y_interp_grid, z=z_interp, visible=False))

    # Make the first trace visible
    fig.data[0].visible = True

    # Create animation frames
    steps = []
    for i in range(len(fig.data)):
        step = dict(method='restyle', args=['visible', [False] * len(fig.data)], label=unpacked_df['Timestamp (s)'].unique()[i])
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 20}, steps=steps)]

    # Update plot layouts
    fig.update_layout(scene=dict(
                        xaxis_title='LED',
                        yaxis_title='PD',
                        zaxis_title='Spectral Response',
                        xaxis=dict(range=[unpacked_df['LED'].min(), unpacked_df['LED'].max()]),
                        yaxis=dict(range=[unpacked_df['PD'].min(), unpacked_df['PD'].max()]),
                        zaxis=dict(range=[unpacked_df['response'].min(), unpacked_df['response'].max()])),
                    sliders=sliders,
                    width=700,
                    margin=dict(r=10, l=50, b=10, t=10))

    return fig

"""