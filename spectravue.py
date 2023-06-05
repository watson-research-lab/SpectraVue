"""
    SpectraVue: a Python Dash application used to visualize and analyze wearable spectroscopic data. 

    Tarek Hamid, hamidtarek3@gmail.com
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_player as dp
from dash import callback_context

import pandas as pd

import matplotlib
matplotlib.use('agg')
import plotly.graph_objects as go
import plotly.io as pio

import base64

import io
import tools

# Imports and Title
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
                              "Tutorial",
                              href="",
                              color="dark",
                              className="me-2 header-button",
                              external_link=True,
                              outline=True
                           ),
                            dbc.Button(
                              "GitHub",
                              href="https://github.com/tarek-hamid/SpectraVue",
                              target="_blank",
                              color="dark",
                              className="me-2 header-button", 
                              external_link=True,
                              outline=True
                           ),
                           dbc.Button(
                              "Contact",
                              href="mailto:hamidtarek3@gmail.com",
                              color="dark",
                              className="header-button",
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
inputs = dbc.Tabs(
    [
        dbc.Tab(
            [
               html.P('The spectrometer mode is used to visualize raw spectrometer data in a static or animated graph.', className='tab-text'),
                dbc.Row([
                    dbc.Col(
                        dcc.Upload(
                            id="upload-data-spectrometer-spec-mode",
                            children=html.Div(
                                dbc.Button(
                                    "Upload Spectrometer Data",
                                    id="upload-button-spectrometer",
                                    color="light",
                                    className="tab-button",
                                ),
                            ),
                            multiple=False,
                            style={"width": "100%"},
                        ),
                        className="mb-3",
                    ),
                    html.Div(id='output-data-upload-spectrometer-spec-mode'), 
            ]),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div("Output:", className="label-text"), 
                            className="d-flex justify-content-center align-items-center",
                            width=3
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='dropdown',
                                options=[
                                    {'label': 'Static', 'value': 'static'},
                                    {'label': 'Animation', 'value': 'animation'}
                                ],
                                value='static',
                                clearable=False,
                                style={'width': '100%'}
                            ),
                            width=7, 
                            className="d-flex justify-content-center align-items-center"
                        ),
                    ],
                    className="d-flex justify-content-center align-items-center"
                ),
                html.Br(),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Generate",
                            color="light",
                            id="generate-button-spectrometer",
                            className="tab-button generate-button",
                        ),
                    ),
                )            
            ],
            label="Spectrometer",
        ),
        dbc.Tab(
            [
                html.P('The biomarker mode is used to visualize spectrometer data in comparison with a clinical biomarker.', className='tab-text'),
                dbc.Row([
                    dbc.Col(
                        dcc.Upload(
                            id="upload-data-spectrometer-bio-mode",
                            children=html.Div(
                                dbc.Button(
                                    "Upload Spectrometer + Biomarker Data",
                                    id="upload-button-spectrometer-bio",
                                    color="light",
                                    className="tab-button",
                                ),
                            ),
                            multiple=False,
                            style={"width": "100%"},
                        )
                    ),
                    html.Div(id='output-data-upload-spectrometer-bio-mode'),
                ]),
                html.Br(),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Generate",
                            id="generate-button-biomarker",
                            color="light",
                            className="tab-button generate-button",
                        ),
                    ),
                )
            ],
            label="Spectro + Biomarker",
        ),
    ],
    className="mb-3",
)

# Graph
graph = dbc.Row([
    dbc.Col(
        dcc.Graph(
            id="graph",
            figure={},
            style={
                "height": "600px",
                "width": "100%",
            }
        ),
        align="center",
    )
], justify="center")

# Layout
app.layout = dbc.Container(
    [
        header,
        html.Br(),
        dbc.Row(
            [
                dbc.Col(inputs, width=3),
                dbc.Col(
                    html.Div(id="graph-container", style={"height": "600px", "width": "100%"}),
                    width=9,
                ),
            ],
        ),
        # Store user uploaded data
        dcc.Store(id="data-store-spectrometer", storage_type="session"),
        dcc.Store(id="data-store-combined", storage_type="session"),
        dcc.Store(id="filename-spectrometer", storage_type="session"),
        dcc.Store(id="filename-combined", storage_type="session"),
    ],
    fluid=True,
)

# Callbacks

# Display the uploaded file name (Spectrometer Mode)
@app.callback(
    Output('output-data-upload-spectrometer-spec-mode', 'children'),
    Input('upload-data-spectrometer-spec-mode', 'filename'),
)
def update_output_spec_mode(filename):
    if filename is not None:
        return html.Div([
            html.B('Uploaded file: '), html.P(filename)
        ], className='file-name')
    else:
        return dash.no_update
    
# Display the uploaded file name (Biomarker Mode)
@app.callback(
    Output('output-data-upload-spectrometer-bio-mode', 'children'),
    Input('upload-data-spectrometer-bio-mode', 'filename'),
)
def update_output_bio_mode(filename_bio):
    if filename_bio is not None:
        return [html.Div(), 
            html.Div([ html.B('Uploaded biomarker file: '), html.P(filename_bio)], className='file-name')]
    else:
        raise PreventUpdate

# Callback to store the uploaded file for spectrometer and biomarker data in both modes
@app.callback(
    [
        Output('filename-spectrometer', 'data'),
        Output('data-store-spectrometer', 'data'),
        Output('filename-combined', 'data'),
        Output('data-store-combined', 'data'),
    ],
    [
        Input('upload-data-spectrometer-spec-mode', 'filename'),
        Input('upload-data-spectrometer-spec-mode', 'contents'),
        Input('upload-data-spectrometer-bio-mode', 'filename'),
        Input('upload-data-spectrometer-bio-mode', 'contents'),
    ]
)
def update_output_spectrometer_and_biomarker(filename_spec, contents_spec, filename_combined, contents_combined):
    outputs = [dash.no_update]*4
    if filename_spec is not None and contents_spec is not None:
        outputs[0], outputs[1] = decode_content_and_store(filename_spec, contents_spec)
    if filename_combined is not None and contents_combined is not None:
        outputs[2], outputs[3] = decode_content_and_store(filename_combined, contents_combined)
    
    return outputs


def decode_content_and_store(filename, contents):
    '''
        Helper method used to decode uploaded file content and store contents in data store. 
    '''
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if content_type == 'data:text/plain;base64':
        data = decoded.decode('utf-8')
        return filename, data
    elif content_type == 'data:text/csv;base64':
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return filename, df.to_json()
    else:
        raise PreventUpdate

# Callback to update graph container based on user input
@app.callback(
    Output('graph-container', 'children'),
    Input('generate-button-spectrometer', 'n_clicks'),
    Input('generate-button-biomarker', 'n_clicks'),
    State('filename-spectrometer', 'data'),
    State('data-store-spectrometer', 'data'),
    State('data-store-combined', 'data'),
    State('dropdown', 'value'),
)
def generate_graph_spec(spec_clicks, biomarker_clicks, filename, spec_data, combined_data, output):

    ctx = callback_context
    prop_id = ctx.triggered[0]['prop_id']

    if 'generate-button-spectrometer' not in prop_id and 'generate-button-biomarker' not in prop_id:
        # Return a blank 3D plot with axis planes
        fig = go.Figure(
            data=[go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=0))],
            layout=go.Layout(scene=dict(xaxis=dict(showgrid=False, zeroline=False),
                                        yaxis=dict(showgrid=False, zeroline=False),
                                        zaxis=dict(showgrid=False, zeroline=False)))
        )
        return dcc.Graph(figure=fig, style={"height": "600px", "width": "100%"})
    elif 'generate-button-spectrometer' in prop_id:
        if filename.endswith('.csv'):
            processed_df = tools.process_spec_csv(spec_data)
        elif filename.endswith('.txt'):
            processed_df = tools.process_spec_txt(spec_data)

        if output == 'static':
            fig = tools.static_graph_output(processed_df)
            return dcc.Graph(figure=fig, style={"height": "600px", "width": "100%"})

        elif output == 'animation':
            video_filename = tools.animation_graph_output(processed_df)
            return dp.DashPlayer(
                id="player",
                url=f"/{video_filename}",
                controls=True,
                width="100%",
                height="600px",
            )
    else: 
        processed_df = tools.process_spec_csv(combined_data)
        video_filename = tools.animation_graph_biomarker(processed_df)
        return dp.DashPlayer(
                id="player",
                url=f"/{video_filename}",
                controls=True,
                width="100%",
                height="600px",
            )
        
if __name__ == '__main__':
   app.run_server(debug=False)