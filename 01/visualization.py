import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, dash_table
from dash.exceptions import PreventUpdate

DATA_PATH = os.getcwd() + "/ivda/01/data/processed.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

app = Dash(__name__)

hover_data = ["Full Name", "Wage(in Euro)", "Overall", "Nationality"]
nationalities = df["Nationality"]
clubs = df["Club Name"]


app.layout = html.Div(
    [
        html.H1("Abgabe 1 von Yannick Brenning und Yannik Lange"),
        html.Div(
            [
                html.H4("Attribute"),
                dcc.Dropdown(
                    ["Wage(in Euro)", "Overall"],
                    value="Overall",
                    id="attribute",
                    clearable=False,
                ),
            ]
        ),
        html.Div(
            [
                html.H4("Filter by Nationality"),
                dcc.Dropdown(
                    nationalities.to_list(),
                    id="filter_nationality",
                    multi=True
                ),
                html.H4("Filter by Club Name"),
                dcc.Dropdown(clubs.to_list(), id="filter_club", multi=True),
            ]
        ),
        html.Div(
            [
                html.H4("Distribution by Age"),
                dcc.Graph(id="dist-plot"),
                dcc.RangeSlider(
                    id="range-slider",
                    min=df["Age"].min(),
                    max=df["Age"].max(),
                    step=1,
                    value=[df["Age"].min(), df["Age"].max()],
                ),
            ]
        ),
        html.Div(
            [
                html.H4("Comparison by Age"),
                dcc.Graph(id="scatter-plot"),
                dcc.RangeSlider(
                    id="range-slider2",
                    min=df["Age"].min(),
                    max=df["Age"].max(),
                    step=1,
                    value=[df["Age"].min(), df["Age"].max()],
                ),
            ]
        ),
        html.Div(
            [
                dcc.Input(
                    id="input1",
                    type="number",
                    value=1,
                    min=1,
                    max=len(df.index)
                ),
                dcc.Input(
                    id="input2",
                    type="number",
                    value=2,
                    min=1,
                    max=len(df.index)
                ),
                dash_table.DataTable(id="table1"),
            ]
        ),
    ],
    style={"marginBottom": 50, "marginTop": 25},
)


@app.callback(
    Output("table1", "data"),
    Output("table1", "columns"),
    Input(component_id="input1", component_property="value"),
    Input(component_id="input2", component_property="value"),
)
def update_table(input1, input2):
    if not input1 or not input2:
        raise PreventUpdate
    df_filtered = df.iloc[[int(input1), int(input2)]]
    data = df_filtered.to_dict("records")
    columns = [{"name": i, "id": i} for i in df_filtered.columns]

    return data, columns


@app.callback(
    Output("scatter-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
    Input("range-slider2", "value"),
)
def update_scatter_chart(
        input_value,
        filter_nationality,
        filter_club,
        slider_range
):
    df_filtered = df
    low, high = slider_range
    mask = (df["Age"] > low) & (df["Age"] < high)
    color = None
    if filter_nationality:
        df_filtered = df_filtered[
            df_filtered["Nationality"].isin(filter_nationality)
        ]
        color = "Nationality"
    if filter_club:
        df_filtered = df_filtered[df_filtered["Club Name"].isin(filter_club)]
        color = "Club Name"
    if not input_value:
        raise PreventUpdate

    return px.scatter(
        df_filtered[mask],
        x="Age",
        y=input_value,
        hover_data=hover_data,
        color=color
    )


@app.callback(
    Output("dist-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
    Input("range-slider", "value"),
)
def update_bar_chart(
        input_value,
        filter_nationality,
        filter_club,
        slider_range
):
    df_filtered = df
    low, high = slider_range
    color = None
    if filter_nationality:
        df_filtered = df_filtered[
            df_filtered["Nationality"].isin(filter_nationality)
        ]
        color = "Nationality"
    if filter_club:
        df_filtered = df_filtered[df_filtered["Club Name"].isin(filter_club)]
        color = "Club Name"
    if not input_value:
        raise PreventUpdate
    mask = (df_filtered["Age"] > low) & (df_filtered["Age"] < high)

    return px.histogram(
        df_filtered[mask],
        x="Age",
        y=input_value,
        hover_data=hover_data,
        color=color
    )


if __name__ == "__main__":
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
