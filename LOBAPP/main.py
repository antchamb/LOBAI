
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc




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



