"""
    Dash app used to generate animations/figures for Lumos. 
"""
import dash
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import base64
import os
import io
import tools
import json
import re
import datetime

app = dash.Dash(
    external_stylesheets=[
       dbc.themes.BOOTSTRAP, 
       dbc.icons.BOOTSTRAP,
       'https://fonts.googleapis.com/css?family=Open+Sans&display=swap'],
    suppress_callback_exceptions=True,
)

app.title = "SpectraVue"


# Header
header = html.Header(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    html.Img(src=app.get_asset_url("logo.png"), height="100px", className="me-3"),
                    width=9,
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                              "GitHub",
                              href="https://github.com/tarek-hamid/SpectraVue",
                              target="_blank",
                              color="dark",
                              className="me-3 my-button", 
                              external_link=True,
                              outline=True
                           ),
                           dbc.Button(
                              "Contact",
                              href="mailto:hamidtarek3@gmail.com",
                              color="dark",
                              className="my-button",
                              external_link=True,
                              outline=True
                           )
                        ],
                        className="d-flex justify-content-end",
                    ),
                    width=3,
                    align="center",
                ),
            ],
            align="center",
            className="py-3",
        ),
        fluid=True,
        className="bg-light",
    )
)

# User Inputs and Buttons
inputs = dbc.Row(
    [
        dbc.Col(
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    dbc.Button(
                        "Upload file",
                        id="upload-button",
                        color="dark",
                        outline=True,
                    ),
                ),
                multiple=False,
            ),
            width=True,
            className="mb-3",
        ),
        dbc.Col(
            dcc.RadioItems(
                id="checklist",
                options=[
                    {"label": "Static", "value": "static"},
                    {"label": "Animation", "value": "animation"},
                ],
                value="static",  # default value
                labelStyle={"display": "inline-block", "margin-top": "10px"},
            ),
            width=True,
            className="text-center",
        ),
    ],
    justify="center",
    align="center",
    className="mb-3",
)

# Graph
graph = dbc.Col(
    [
        dcc.Graph(id="graph", style={"height": "600px"}),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Download as PNG",
                        id="download-png-button",
                        color="dark",
                        className="me-1",
                    ),
                    width=True,
                ),
                dbc.Col(
                    dbc.Button(
                        "Download as PDF",
                        id="download-pdf-button",
                        color="dark",
                        className="me-1",
                    ),
                    width=True,
                ),
            ],
            justify="center",
            align="center",
            className="my-3",
        ),
    ],
    width=True,
)

# App Layout
app.layout = dbc.Container(
    [
        header,
        html.Br(),
        inputs,
        graph,
        # Store user uploaded data
        dcc.Store(id="data-store", storage_type="session"),
        dcc.Store(id="filename", storage_type="session"),
    ],
    fluid=True,
)


'--------------------------------------------------------------------------------------------------------------------------------------'

# Callbacks

# Displays the file name that is uploaded
@callback(
   Output('output-data-upload', 'children'),
   Input('upload-data', 'filename'),
)
def display_file_name(filename):
   if filename is not None:
      return html.Div([
         html.B('Uploaded file: '), html.P(filename)
      ])

# Stores the contents of files in local data store on update
@callback(
   Output('data-store', 'data'),
   Input('upload-data', 'contents'),
)
def store_file_contents(contents):
   if contents is not None:
      content_type, content_string = contents.split(',')
      decoded = base64.b64decode(content_string)

      if content_type == 'data:text/plain;base64':
         data = decoded.decode('utf-8')
         return json.dumps(data)
      elif content_type == 'data:text/csv;base64':
         df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
         return df.to_json()
   else:
      raise PreventUpdate

# Downloads generated graph as a PNG
@callback(
    Output("download-png", "data"),
    Input("download-png-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True
)
def download_graph_png(n_clicks, figure):
    if figure is None:
        raise PreventUpdate
    img_bytes = pio.to_image(figure, format='png')
    return dcc.send_bytes(img_bytes, filename="output.png")

# Downloads generated graph as a PDF
@callback(
    Output("download-pdf", "data"),
    [Input("download-pdf-button", "n_clicks")],
    State('graph', 'figure'),
    prevent_initial_call=True
)
def download_graph_pdf(n_clicks, figure):
    if figure is None:
        raise PreventUpdate
    pdf_bytes = io.BytesIO(pio.to_image(figure, format='pdf'))
    return dcc.send_bytes(pdf_bytes.getvalue(), filename="output.pdf")

@callback(
    Output('file-name-input', 'style'),
    Output('file-name-input-label', 'style'),
    Output('file-name-output', 'children'),
    Input('checklist', 'value')
)
def show_hide_file_name_input(value):
    if value == 'animation':
        return {'display': 'block'}, {'display': 'inline-block', 'margin-right': '10px'}, html.Label('File name:', htmlFor='file-name-input', style={'margin-right': '10px'})
    else:
        return {'display': 'none'}, {'display': 'none'}, None
    
# create an instance of the FFMpegWriter to save animation
writer = FFMpegWriter(fps=30)

@callback(
   Output('graph', 'figure'),
   Output('animation-success', 'children'),
   Output('animation-success', 'style'),
   Input('generate-button', 'n_clicks'),
   Input('output-data-upload', 'children'),
   Input('data-store', 'data'),
   Input('start-time', 'value'),
   Input('end-time', 'value'),
   Input('led-order', 'value'),
   Input('file-name-input', 'value'),
   State('generate-button', 'n_clicks_timestamp')
)
def process_data_and_output_graph(generate_button, filename, data, start_time, end_time, led_order, filename_input, generate_button_ts):
   if not generate_button or not generate_button_ts:
      raise PreventUpdate
   if filename and data is not None and generate_button_ts > generate_button:
      raw_filename = filename['props']['children'][1]['props']['children']
      
      # If a csv was uploaded
      if raw_filename.endswith('csv'):
         fig = process_csv_and_output_graph(data, start_time, end_time, led_order)
         return fig, None, None

      # If a text was uploaded
      elif raw_filename.endswith('txt'):
         if not filename_input or filename_input is None:
            filename_input = 'Output'

         animation = process_txt_and_output_graph(data, start_time, end_time)
         animation.save(f'{filename_input}.mp4', writer=writer)
         return None, f'{filename_input}.mp4 has been successfully saved.', {'display': 'block'}

      # Everything else is unsupported for now
      else:
         raise PreventUpdate
      
   else:
      raise PreventUpdate


def process_csv_and_output_graph(data, start_time, end_time, led_order):
   '''
      Processes CSV as input and generates static graph. 
   '''
   data_path = 'Data/'

   led_file = data_path + 'LEDs_watch.csv'
   pd_file = data_path + 'PDs.csv'
   led_df = pd.read_csv(led_file)
   pd_df = pd.read_csv(pd_file)

   df = pd.read_json(data)
   start_time = int(start_time)
   end_time = int(end_time)
   led_order = [eval(i) for i in led_order.split(',')]

   df = df[['1', '2', '5']]

   df = tools.get_counts_per_LED(df)

   # Hard-coded value for start time
   xx = df.iloc[0]['timestamp']
   xx = xx + 42_000

   # Loop through LED wavelengths
   for led in led_order:
    
      # Create LED column and assign value between start and end times
      df.loc[df.timestamp.between(xx + 3000, xx + 25000), 'LED'] = led
    
      # Loop through every 30,000 samples
      xx = xx + 30_000

   # Get mean counts per LED
   mean_df = tools.get_mean_counts_per_LED(df)

   # Drop 940 Wavelength
   mean_df = mean_df.drop([940], axis=0)
   
   # Convert to numpy array for faster computation
   medium_arr = mean_df.to_numpy() 

   theory_path = 'Data/Medium_Data/theory_arr.csv'

   # Load Data
   theory_areas = np.loadtxt(theory_path, delimiter = ",")

   # Subtract experimental data with theoretical approx
   med_arr = np.subtract(medium_arr, theory_areas)
   med_arr = med_arr * -1

   # Choose method of interpolation
   # Options: nearest, linear, cubic, etc.
   method = 'cubic'

   # Construct axises for 3D representation
   X = np.array(led_df.Wavelength.tolist()*len(pd_df.Wavelength.tolist()))
   Y = np.array(np.repeat(pd_df.Wavelength.tolist(), len(led_df.Wavelength.tolist())))
   Z = np.array([item for sublist in med_arr for item in sublist])

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
         scene = dict(
            xaxis_title='LED',
            yaxis_title='PD',
            zaxis_title='Spectral Response',
            xaxis=dict( 
                  tickmode='array',
                  tickvals=led_df.Wavelength.tolist(),
                  ticktext=[str(i) for i in led_df.Wavelength.tolist()]
            ),
            yaxis=dict(
                  tickmode='array',
                  tickvals=pd_df.Wavelength.tolist(),
                  ticktext=[str(i) for i in pd_df.Wavelength.tolist()]
            )
         )
      )
   
   return fig

def process_txt_and_output_graph(data, start_time, end_time):
   '''
      Given a text file with JSON formatted data, parse through the data and output an animation graph.    
   '''
   parsed_data = tools.parse_json(data)
   
   pd = [415,445,480,515,555,590,630,680]
   led2=[415,460,470,530,580,600,633,670]
   def update_plot(frame_number, zarray, plot):
      plot[0].remove()
      plot[0] = ax.plot_surface(x, y, -1*(zarray[:,:,frame_number]-baseline), cmap="magma")


   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   N = 8
   nmax=len(parsed_data)

   X = np.array(led2*len(pd))
   Y = np.array(np.repeat(pd, len(led2)))
   #Z = np.array([item for sublist in med_arr for item in sublist])

   x_min = X.min()
   x_max = X.max()
   y_min = Y.min()
   y_max = Y.max()

   x_new = np.linspace(x_min, x_max, 8)
   y_new = np.linspace(y_min, y_max, 8)
   #z_new = griddata((X, Y), Z, (x_new[None,:], y_new[:,None]), method='cubic')
   x, y = np.meshgrid(x_new, y_new)


   #x = np.linspace(-4,4,N+1)
   #x, y = np.meshgrid(x, x)
   zarray = np.zeros((N, N, nmax))

   #f = lambda x,y,sig : 1/np.sqrt(sig)*np.exp(-(x/100**2+y/100**2)/sig**2)

   #for i in range(nmax):
   #    zarray[:,:,i] = f(x,y,1.5+np.sin(i*2*np.pi/nmax))

   for idx,nd in enumerate(parsed_data):
      #print(nd)
      keys = list(nd.keys())
      values = list(nd.values())

      array = np.zeros((len(keys), len(values[0].keys())))

      for i in range(len(keys)):
         for j, key in enumerate(values[0].keys()):
               # if key not in ['timestamp', 'Clear', 'NIR', 'LED_ON']:
               array[i][j] = values[i][key]

      array = array[:, :-3]
      #print(array)

      zarray[:,:,idx] = array

      #print(zarray)

   baseline = np.average(zarray[:,:,:10], axis=2)

   plot = [ax.plot_surface(x, y, -1*(zarray[:,:,0]-baseline), color='0.75', rstride=1, cstride=1)]
   ax.set_zlim(-2000,10000)
   ax.set_xticks(led2)
   ax.set_yticks(pd)
   ax.set_xlabel('LED')
   ax.set_ylabel('PD')
   ax.set_zlabel('Counts')
   animate = animation.FuncAnimation(fig, update_plot, nmax, fargs=(zarray, plot))

   return animate

if __name__ == '__main__':
   app.run_server(debug=True)