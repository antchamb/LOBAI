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
    Output('x', 'data'),
    Output('y', 'data'),
    Input('load-button', 'n_clicks'),
    [
        State('day-selector', 'value'),
        State('stock-selector', 'value'),
        State('level-selector', 'value'),
        State('T', 'value'),
        State('prediction-selector', 'value'),
    ]
)
def load_data(n, day, stock, level, T, k):
    if None in locals().values():
        raise PreventUpdate

    level += 1
    x, y = Dataset_fi2010(False, "DecPre", list(stock), list(day), T, k, level).__init_dataset__()

    x *= 10 ** 3


    # price_fig = go.Figure()
    # volume_fig = go.Figure()
    # for i in range(1, 4*(level-1), level):
    #     print(i)
    #     price_fig.add_trace(go.Scatter(
    #         y=x[0, :, i], mode='lines+markers', name=f'{features[str(i)]}',
    #     ))
    #     volume_fig.add_trace(go.Scatter(
    #         y=x[0, :, i+1], mode='lines+markers', name=f'{features[str(i+1)]}'
    #     ))
    #     price_fig.add_trace(go.Scatter(
    #         y=x[0, :, i+2], mode='lines+markers', name=f'{features[str(i+2)]}'
    #     ))
    #     volume_fig.add_trace(go.Scatter(
    #         y=x[0, :, i+3], mode='lines+markers', name=f'{features[str(i+3)]}'
    #     ))

    return x, y

# @app.callback(
#     Output('price-graph', 'figure'),
#     Output('volume-graph', 'figure'),
#     Output('spread-graph', 'figure'),
# )

if __name__ == '__main__':
    app.run_server(debug=True)