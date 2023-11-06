import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

DATA_PATH = os.getcwd() + "/data/Aufgabe-1.csv"

# TODO: Get preprocessed data instead of skip
df = pd.read_csv(DATA_PATH, on_bad_lines="skip", low_memory=False)

app = Dash(__name__)

hover_data = ["'Full Name'", "'Wage(in Euro)'", "'Overall'", "'Nationality'"]
nationalities = df["\'Nationality\'"]
clubs = df["\'Club Name\'"]

app.layout = html.Div(
    [
        html.H4("Interactive scatter plot with the dataset"),
        html.Div(
            [
                "Attribute",
                dcc.Dropdown(
                    ["'Wage(in Euro)'", "'Overall'"],
                    value="'Overall'",
                    id="attribute",
                    clearable=False,
                ), "Filter Nationality",
                dcc.Dropdown(
                    nationalities.to_list(),
                    id="filter_nationality",
                    multi=True
                ), "Filter Club Name",
                dcc.Dropdown(
                    clubs.to_list(),
                    id="filter_club",
                    multi=True
                ),
                "Verteilung",
                dcc.Graph(id="dist-plot"),
                "Gegenueberstellung",
                dcc.Graph(id="scatter-plot"),
            ]
        )
    ]
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
)
def update_scatter_chart(input_value, filter_nationality, filter_club):
    df = pd.read_csv(DATA_PATH, on_bad_lines="skip", low_memory=False)

    color = None
    if filter_nationality:
        df = df[df["\'Nationality\'"].isin(filter_nationality)]
        color = "'Nationality'"
    if filter_club:
        df = df[df["\'Club Name\'"].isin(filter_club)]
        color = "'Club Name'"
    if not input_value:
        raise PreventUpdate

    return px.scatter(
        df,
        x="'Age'",
        y=input_value,
        hover_data=hover_data,
        color=color
    )


@app.callback(
    Output("dist-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
)
def update_bar_chart(input_value, filter_nationality, filter_club):
    df = pd.read_csv(DATA_PATH, on_bad_lines="skip", low_memory=False)

    color = None
    if filter_nationality:
        df = df[df["\'Nationality\'"].isin(filter_nationality)]
        color = "'Nationality'"
    if filter_club:
        df = df[df["\'Club Name\'"].isin(filter_club)]
        color = "'Club Name'"
    if not input_value:
        raise PreventUpdate

    return px.histogram(
        df,
        x="'Age'",
        y=input_value,
        hover_data=hover_data,
        color=color
    )


if __name__ == "__main__":
    print("starting app")
    app.run_server(debug=True)


"""
@app.callback(
    Output("filt_key", "filter_option"),
    Input("filt_key", "search_value")
)
def update_filter_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [o for o in filter_option if search_value in o["label"]]

@app.callback(
    Output("filter_value", "options"),
    Input("filt_key", "attr_value"),
    Input("filter_value", "search_value")
)
def update_options(search_value,attr_value):
   # if not search_value:
   #     raise PreventUpdate
   # options = 
    #return [o for o in options if search_value in o]
    return df[attr_value].tolist()
"""
"""
app.layout = html.Div([
    html.H4('Interactive scatter plot with the dataset'),
    dcc.RadioItems(filter_options=["\'Wage(in Euro)\'", '\'Overall\''], value="\'Overall\'",id='controls-and-radio-item'),
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
