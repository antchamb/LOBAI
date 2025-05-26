import dash
from dash import html, dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from data_loader.FI2010 import *
import pandas as pd



dash.register_page(__name__, name='fi2010', path='/fi2010')

###################################################################################################################
# PARAMETERS AND LAYOUT
###################################################################################################################

parameters = html.Div([
    html.H4('Adjust Parameters: '),
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
        'width': '100%'
    }),

    html.Div([
        html.Label('day selection'),
        dcc.RangeSlider(
            id='day-slider',
            min=1,
            max=10,
            step=1,
            value=[1, 2],
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
                {'label': 'Zscore', 'value': 'Zscore'},
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

    html.Div([
        html.Label('Horizon'),
        dcc.Dropdown(
            id='k',
            options=[
                {'label': '1', 'value': 1},
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3},
                {'label': '5', 'value': 5},
                {'label': '10', 'value': 10},
            ]
        )
    ])
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


results = html.Div([
    html.Label('Label Analysis:'),
    dcc.Graph(id='label-hist'),

    html.Hr(),
    html.Label('bid ask spread by tick:'),
    html.Div(id='spread-line'),

    html.Hr(),
    html.Label('Order Flow Imbalance by tick:'),
    html.Div(id='ofi-line'),

    html.Hr(),
    html.Label('feature correlation heatmap:'),
    dcc.Graph('corr-heat'),

    html.Hr(),
    html.Label('Volatility vs Horizon:'),
    dcc.Graph(id='vol-hist')
], style={
    "marginLeft": "10vw",
    "width": "90vw"
})

layout = html.Div(
    [parameters, results],
    style={"display": "flex", "flexDirection": "row", "width": "100%"},
)

###################################################################################################################
# CALLBACK FUNCTIONS
###################################################################################################################

@callback(
    Output('label-hist', 'figure'),
    [
        Input('stock-dropdown', 'value'),
        Input('day-slider', 'value'),
        Input('norm-method', 'value'),
    ]
)
def get_label_histogram(stocks, days, norm):
    if None in locals().values():
        raise PreventUpdate

    stocks = [stocks] if isinstance(stocks, int) else list(stocks)
    days = list(range(days[0], days[1]+1))

    label_hist = {
        k: Dataset_fi2010(
            auction=False,
            normalization=norm,
            stock_idx=stocks,
            days=days,
            T=10,
            k=k_index,  # adjusted below
            lighten=True,
        ).labels_count
        for k, k_index in zip([1, 2, 3, 5, 10], [0, 1, 2, 3, 4])  # FI-2010 label indices
    }

    first_k = next(iter(label_hist))
    label_types = label_hist[first_k].keys()

    fig = go.Figure()

    for label in label_types:
        fig.add_trace(
            go.Bar(
                x=[str(k) for k in label_hist],
                y=[label_hist[k].get(label, 0) for k in label_hist],
                name=label
            )
        )

    fig.update_layout(
        barmode='group',
        title="Label Distribution per Prediction Horizon (k)",
        xaxis_title="Prediction Horizon (k)",
        yaxis_title="Sample Count",
        legend_title="Label"
    )

    return fig

@callback(
    Output('spread-line', 'children'),
    [
        Input('stock-dropdown', 'value'),
        Input('day-slider', 'value'),
        Input('norm-method', 'value'),
    ]
)
def intraday_spread(stocks, days, norm):
    if None in locals().values():
        raise PreventUpdate

    stocks = [stocks] if isinstance(stocks, int) else list(stocks)
    days = list(range(days[0], days[1]+1))

    graphs = []
    for stock in stocks:
        spread_data = Dataset_fi2010(
            auction=False,
            normalization=norm,
            stock_idx=[stock],
            days=days,
            T=1,
            k=0,  # k is not used in spread calculation
            lighten=True,
        ).get_spread()

        tick_time = np.arange(len(spread_data))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tick_time,
            y=spread_data,
            mode='lines',
            name=f'Stock {stock}'
        ))
        fig.update_layout(
            title=f'Bid-Ask Spread for Stock {stock} by Tick',
            xaxis_title='Tick',
            yaxis_title='Spread',
            legend_title='Stock'
        )

        graphs.append(dcc.Graph(figure=fig))

    return graphs


@callback(
    Output('ofi-line', 'children'),
    [
        Input('stock-dropdown', 'value'),
        Input('day-slider', 'value'),
        Input('norm-method', 'value'),
    ]
)
def get_ofi_graphs(stocks, days, norm):
    if None in locals().values():
        raise PreventUpdate

    stocks = [stocks] if isinstance(stocks, int) else list(stocks)
    days = list(range(days[0], days[1]+1))

    graphs = []
    for stock in stocks:
        ofi_data = Dataset_fi2010(
            auction=False,
            normalization=norm,
            stock_idx=[stock],
            days=days,
            T=1,
            k=0,  # k is not used in OFI calculation
            lighten=True,
        ).get_ofi()

        tick_time = np.arange(len(ofi_data))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tick_time,
            y=ofi_data,
            mode='lines',
            name=f'Stock {stock}'
        ))
        fig.update_layout(
            title=f'Order Flow Imbalance for Stock {stock} by Tick',
            xaxis_title='Tick',
            yaxis_title='Order Flow Imbalance',
            legend_title='Stock'
        )

        graphs.append(dcc.Graph(figure=fig))

    return graphs

@callback(
    Output('corr-heat', 'figure'),
    [
        Input('stock-dropdown', 'value'),
        Input('day-slider',    'value'),
        Input('norm-method',   'value')
    ]
)
def draw_feature_heat(stocks, days, norm):
    if None in locals().values():
        raise PreventUpdate

    stocks = [stocks] if isinstance(stocks, int) else list(stocks)
    days = list(range(days[0], days[1]+1))

    ds = Dataset_fi2010(
        auction=False,
        normalization=norm,
        stock_idx=stocks,
        days=days,
        T=1,
        k=0,
        lighten=True,
    )

    X = ds.as_2d()
    corr = np.corrcoef(X, rowvar=False)
    fig = go.Figure(
        go.Heatmap(
            z=corr,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='ρ')
        )
    )
    fig.update_layout(
        title=f'Feature Correlation (stocks = {len(stocks)}, days={days})',
        xaxis_title='Features',
        yaxis_title='Features'
    )

    return fig

@callback(
    Output('vol-hist', 'figure'),
    [
        Input('stock-dropdown', 'value'),
        Input('day-slider', 'value'),
        Input('norm-method', 'value'),
        Input('k', 'value')
    ]
)
def volatility_horizon(stocks, days, norm, k):
    if None in locals().values():
        raise PreventUpdate

    stocks = [stocks] if isinstance(stocks, int) else list(stocks)
    days = list(range(days[0], days[1]+1))

    vols = []
    for s in stocks:
        vol_data = Dataset_fi2010(
            auction=False,
            normalization=norm,
            stock_idx=[s],
            days=days,
            T=1,
            k=k,  # k is used for volatility calculation
            lighten=True,
        ).get_volatility(horizon=k)

        vols.append(vol_data)

    fig = go.Figure(go.Bar(
        x=[f'Stock {stock}' for stock in stocks],
        y=vols, name=f"σ(k={k})"
    ))
    fig.update_layout(
        title=f"Realised volatility vs stock (k={k} snapshots)",
        xaxis_title="Stock", yaxis_title="σ",
        template="plotly_white"
    )
    return fig