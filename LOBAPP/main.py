import base64

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from torch.onnx.symbolic_opset11 import unsqueeze



app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


header = html.Div([
    html.Div([
        html.Div([
            dcc.Link(
                page['name'] + ' | ',
                href=page['path'],
                style={
                    'color': '#fff',
                    'fontSize': '18px',
                    'padding': '0 10px',
                    'textDecoration': 'none',
                }
            )
            for page in dash.page_registry.values()
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'height': '100%',
        }),
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'padding': '0',
        'height': '100%',
    })
], style={
    'backgroundColor': '#1a1a1a',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'width': '100%',
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'height': '7vh',
    'zIndex': '999',
    'margin': '0',
    'padding': '0'
})

# force html structure
app.index_string = '''
<!DOCTYPE html>

<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Page layout
app.layout = html.Div([
    header,
    html.Div(
        dash.page_container,
        style={
            'marginTop': '7vh',  # Creates space exactly matching the header height
            'padding': '0',  # No padding around the content
            'flex': '1',  # Allow the content to take the remaining space
        }
    )
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'margin': 0,
    'padding': 0,
    'minHeight': '100vh',  # Full height layout
})

if __name__ == '__main__':
    app.run(debug=True)



from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from data_loader.FI2010 import *
from models.deeplob import Deeplob


# model = Deeplob(lighten=True)
# model.load_state_dict(torch.load("deeplob_weights.pth"))
# model.eval()
#
# def predict_lob(input_data):
#     input_tensor = torch.tensor(input_data.unsqueeze(0).unsqueeze(0).float())
#     with torch.no_grafd():
#         output = model(input_tensor)
#     return output.numpy()


# @app.callback(
#     Output('label-hist', 'figure'),
#     [
#         Input('stock-dropdown', 'value'),
#         Input('day-slider', 'value'),
#         Input('norm-method', 'value'),
#     ]
# )
# def get_label_histogram(stocks, days, norm):
#     if None in locals().values():
#         raise PreventUpdate
#
#     stocks = [stocks] if isinstance(stocks, int) else list(stocks)
#     days = list(range(days[0], days[1]+1))
#
#     label_hist = {
#         k: Dataset_fi2010(
#             auction=False,
#             normalization=norm,
#             stock_idx=stocks,
#             days=days,
#             T=10,
#             k=k_index,  # adjusted below
#             lighten=True,
#         ).labels_count
#         for k, k_index in zip([1, 2, 3, 5, 10], [0, 1, 2, 3, 4])  # FI-2010 label indices
#     }
#
#     first_k = next(iter(label_hist))
#     label_types = label_hist[first_k].keys()
#
#     fig = go.Figure()
#
#     for label in label_types:
#         fig.add_trace(
#             go.Bar(
#                 x=[str(k) for k in label_hist],
#                 y=[label_hist[k].get(label, 0) for k in label_hist],
#                 name=label
#             )
#         )
#
#     fig.update_layout(
#         barmode='group',
#         title="Label Distribution per Prediction Horizon (k)",
#         xaxis_title="Prediction Horizon (k)",
#         yaxis_title="Sample Count",
#         legend_title="Label"
#     )
#
#     return fig


