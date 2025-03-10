import dash
from dash import Output, Input, State

from app.layout import *

app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = parameters

@app.callback(
    Output("modal", "is_open"),
    [
        Input("open-modal", "n_clicks"),
        Input("close-modal", "n_clicks")
    ],
    [
        State("modal", "is_open")
    ]
)
def toggle_modal(n_open, n_close, is_open):
    if n_open or n_close:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run_server(debug=True)