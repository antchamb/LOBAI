import base64

import dash
import dash_bootstrap_components as dbc
from torch.onnx.symbolic_opset11 import unsqueeze

from utils.layout import layout
from utils.callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = layout

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

#
# import plotly.graph_objects as go
#
# @app.callback(
#     [Output('prediction-output', 'children'),
#      Output('decision-graph', 'figure')],
#     [Input('upload-data', 'contents')],
#     [State('upload-data', 'filename')]
# )
# def update_decision(contents, filename):
#     if not contents:
#         return "No data uploaded.", go.Figure()
#
#     # Parse uploaded data
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     lob_data = np.loadtxt(decoded.splitlines())  # Adjust parsing logic as needed
#
#     # Predict using the model
#     prediction = predict_lob(lob_data)
#     predicted_class = np.argmax(prediction, axis=1)
#
#     # Create a bar chart for the decision
#     fig = go.Figure(data=[
#         go.Bar(x=['Down', 'Flat', 'Up'], y=prediction[0])
#     ])
#     fig.update_layout(title="Model Decision", xaxis_title="Class", yaxis_title="Probability")
#
#     return f"Predicted Class: {['Down', 'Flat', 'Up'][predicted_class[0]]}", fig
#



if __name__ == "__main__":
    app.run_server(debug=True)
