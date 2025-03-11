import dash
from dash import Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

from app.layout import *
from loader.fi2010 import *
from loader.features import *

app = dash.Dash(__name__)

app.layout = layout


@app.callback(
    Output('price-graph', 'figure'),
    Output('volume-graph', 'figure'),
    # Output('spread-graph', 'figure'),
    # Output('midprice-graph', 'figure'),
    [
        Input('day-selector', 'value'),
        Input('stock-selector', 'value'),
        Input('level-selector', 'value'),
        Input('T', 'value'),
        Input('prediction-selector', 'value'),
    ]
)
def generate_graph(day, stock, level, T, k):
    if None in locals().values():
        raise PreventUpdate
    x, y = Dataset_fi2010(False, "DecPre", list(stock), list(day), T, k, level).__init_dataset__()
    features = get_features_structure(level)
    print(features)
    x *= 10 ** 6
    level += 1

    price_fig = go.Figure()
    volume_fig = go.Figure()
    for i in range(1, level):
        price_fig.add_trace(go.Scatter(
            y=x[0, :, i], mode='lines+markers', name=f'{features[str(i)]}',
        ))
        volume_fig.add_trace(go.Scatter(
            y=x[0, :, i+1], mode='lines+markers', name=f'{features[str(i+1)]}'
        ))

    return price_fig, volume_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=7777)