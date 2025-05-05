
from dash import html, dcc


parameters = html.Div([
    html.H3('Adjust Parameters: '),
    html.Div([
        html.Label("Stock selection"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': name, 'value': idx} for idx, name in enumerate(['KESBV', 'OUT1V', 'SAMPO', 'RTRKS', 'WRT1V'])],
            value=0,
            multi=True,
            placeholder="Select stock",
        )
    ], style={
        'marginTop': '10px',
        'width': '100%'
    }),

    html.Div([
        html.Label('day selection'),
        dcc.RangeSlider(
            id='day-slider',
            min=0,
            max=10,
            step=1,
            value=[0],
            allowCross=False,
            tooltip={'placement': 'bottom', 'always_visible': True}
        ),
    ], style={
        'marginTop': '10px',
        'width': '100%'
    }),

    html.Div([
        html.Label('Normalization:'),
        dcc.RadioItems(
            id='norm-method',
            options=[
                {'label': 'Zscore', 'value': 'zscore'},
                {'label': 'Min-Max', 'value': 'MinMax'},
                {'label': 'DecPre', 'value': 'DecPre'},
            ],
            value='Zscore',
            labelStyle={'display': 'inline-block'}
        )
    ], style={
        'marginTop': '10px',
        'width': '100%'
    }),
], style={
    'position': 'fixed',
    'top': '0px',
    'left': '0px',
    'width': '10vw',
    'height': '100vh',
    'overflow-y': 'auto',
    'backgroundColor': '#f8f9fa',
    'padding': '10px',
    'borderRight': '2px solid #ccc'
})

layout = parameters