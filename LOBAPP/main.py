import dash
import dash_bootstrap_components as dbc

from utils.layout import layout
from utils.callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = layout

from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from data_loader.FI2010 import *


@app.callback(
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
        ).labels_count()
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







if __name__ == "__main__":
    app.run_server(debug=True)
