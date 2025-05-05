
from dash import dcc, html
import dash_bootstrap_components as dbc

parameters = html.Div([
    html.H3("Adjust Parameters:"),
    html.Div([
        html.Label("days:"),
        dcc.Dropdown(
            id="day-selector",
            options=[{'label': f'Day {i}', 'value': i} for i in range(1, 10)],
            multi=True,
            placeholder="Select day",
        ),
        html.Label("Stocks:"),
        dcc.Dropdown(
            id="stock-selector",
            options=[{'label': f'Stock {i}', 'value': i} for i in range(1, 5)],
            multi=True,
            placeholder="Select stock",
        ),
        html.Label("Level:"),
        dcc.Dropdown(
            id='level-selector',
            options=[{'label': f'Level {i}', 'value': i} for i in range(1, 11)],
        ),
        html.Hr(),
        html.Label("Horizon (T):"),
        dcc.Input(
            id="T",
            type="number",
            value=10,
            step=1,
            style={'marginBottom': '10px', 'width': '100%'},
        ),
        html.Label("Prediction (k):"),
        dcc.Dropdown(
            id="prediction-selector",
            options=[
                {'label': 1, 'value': 0},
                {'label': 2, 'value': 1},
                {'label': 3, 'value': 2},
                {'label': 4, 'value': 3},
                {'label': 10, 'value': 4},
            ]
        ),
        dbc.Button(id='load-button')
    ])
], style={
    "position": "fixed",
    "top": "0px",
    "left": "0px",
    "width": "8vw",
    "height": "100vh",
    "overflow-y": "auto",
    "backgroundColor": "#f8f9fa",
    "padding": "10px",
    "borderRight": "2px solid #ccc"
})

results = html.Div([
    html.H3("Price Evolution:"),
    dcc.Graph(id='price-graph'),
    html.Hr(),
    html.H3("Volume Evolution:"),
    dcc.Graph(id='volume-graph'),
    html.H3("Spread Evolution:"),
    dcc.Graph(id='spread-graph'),
    html.H3("MidPrice Evolution:"),
    dcc.Graph(id='midprice-graph'),
    dcc.Store(id='x', storage_type='session'),
], style={
    "marginLeft": "8vw",
    "width": "90vw"
})

layout = html.Div([
    html.Div(
    [parameters, results],
    style={"display": "flex", "flexDirection": "row", "width": "100%"},
)])