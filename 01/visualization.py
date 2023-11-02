from dash import Dash, dcc, html, callback, Output, Input
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd

app = Dash(__name__)
hover_data = ["'Full Name'", "'Wage(in Euro)'", "'Overall'"]
app.layout = html.Div(
    [
        html.H4("Interactive scatter plot with the dataset"),
        html.Div(
            [
                "Attribut",
                dcc.Dropdown(
                    ["'Wage(in Euro)'", "'Overall'"],
                    value="Overall",
                    id="my-input",
                    clearable=False,
                ),
                dcc.Graph(id="dist-plot"),
                dcc.Graph(id="scatter-plot"),
            ]
        ),
    ]
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input(component_id="my-input", component_property="value"),
)
def update_bar_chart(input_value):
    df = pd.read_csv("ivda/01/data/Aufgabe-1.csv", on_bad_lines="skip")
    try:
        fig = px.scatter(df, x="'Age'", y=input_value, hover_data=hover_data)
    except:
        pass
    return fig


@app.callback(
    Output("dist-plot", "figure"),
    Input(component_id="my-input", component_property="value"),
)
def update_bar_chart(input_value):
    df = pd.read_csv("ivda/01/data/Aufgabe-1.csv", on_bad_lines="skip")
    try:
        fig2 = px.histogram(df, x="'Age'", y=input_value, hover_data=hover_data)
    except:
        pass
    return fig2


app.run_server(debug=True)


"""
app.layout = html.Div([
    html.H4('Interactive scatter plot with the dataset'),
    dcc.RadioItems(options=["\'Wage(in Euro)\'", '\'Overall\''], value="\'Overall\'",id='controls-and-radio-item'),
    dcc.Graph(id="scatter-plot"),
    html.P("Filter by Age:"),
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=60, step=1,
        marks={10:'10',15: '15',20:'20',25: '25',30:'30',35: '35',40:'40',45: '45'},
        value=[df['\'Age\''].min(), df['\'Age\''].max()]
    ),
])


@app.callback(
    Output("scatter-plot", "figure"), 
    Input(component_id='controls-and-radio-item', component_property='value'),
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df['\'Age\''] > low) & (df['\'Age\''] < high)
    fig = px.scatter(
        df[mask], x='\'Age\'', y=col_chosen,
        hover_data=['\'Full Name\''])
    return fig


app.run_server(debug=True)
"""
