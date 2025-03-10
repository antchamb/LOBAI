import dash_bootstrap_components as dbc
from dash import dcc, html


basic_features = [
    f"{feature}_{i}"
    for i in range(1, 11)
    for feature in ["P_Ask", "V_Ask", "P_Bid", "V_Bid"]
]

time_insensitive_features = [
        f"{feature}_{i}"
        for i in range(1,11)
        for feature in ["Spread", "MidPrice"]
    ] + [
        "P_Diff_Ask", "P_Diff_Bid"
    ] + [
        f"{feature}_{i}"
        for i in range(1,10)
        for feature in ["P_AbsDiffRel_Ask", "P_AbsDiffRel_Bid"]
    ] + [
        "P_Mean_Ask",
        "P_Mean_Bid",
        "V_Mean_Ask",
        "V_Mean_Bid"
    ] + [
        "P_AccDiff",
        "V_AccDiff"
    ]

time_sensitive_features = [
        f"{feature}_{i}"
        for i in range(1,11)
        for feature in ["P_Deriv_Ask", "P_Deriv_Bid", "V_Deriv_Ask", "V_Deriv_Bid"]
    ] + [
        f"IntensityAverage_{i}"
        for i in range(1,7)
    ] + [
        f"IntensityRelComparison_{i}"
        for i in range(1,7)
    ] + [
        f"LimitActivityAcceleration_{i}"
        for i in range(1,7)
    ]



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
            options=[{'label': f'Stock {i}', 'value': i} for i in range(1, 10)],
            multi=True,
            placeholder="Select stock",
        ),
        html.Label("Level:"),
        dcc.Dropdown(
            id='level-selector',
        )
        # dbc.Button("Open Modal", id="open-modal", n_clicks=0, className="mb-3"),
    #     dbc.Modal([
    #         dbc.ModalHeader("Select Features"),
    #         dbc.ModalBody([
    #             dcc.Dropdown(
    #                 id='feature-selector',
    #                 options=[{'label': features[i], 'value': f'feature_{i}'} for i in range(1, 144)],
    #                 multi=True,
    #                 placeholder='Select features...'
    #             ),
    #         ]),
    #         dbc.ModalFooter(
    #             dbc.Button("Close", id="close-modal", className="ml-auto", n_clicks=0)
    #         ),
    # ], id="modal", is_open=False),
    ])
])