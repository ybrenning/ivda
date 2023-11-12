import os
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.exceptions import PreventUpdate

DATA_PATH = os.getcwd() + "/data/processed.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

app = Dash(__name__)

hover_data = ["Full Name", "Wage(in Euro)", "Overall", "Nationality"]
clubs = df["Club Name"]

STAT_NAMES = df.columns[df.columns.tolist().index("Pace Total") :].tolist()

excluded_attributes = [
    "Known As",
    "Full Name",
    "National Team Image Link",
    "Image Link",
]

app.layout = html.Div(
    [
        html.H1("Submission 1 (Yannick Brenning, Yannik Lange)"),
        html.H2("View Attribute Distributions"),
        html.Div(
            [
                html.H4("Select Attribute"),
                dcc.Dropdown(
                    [
                        "Overall",
                        "Potential",
                        "Value(in Euro)",
                        "Age",
                        "Height(in cm)",
                        "Weight(in kg)",
                        "TotalStats",
                        "BaseStats",
                        "Wage(in Euro)",
                    ]
                    + STAT_NAMES,
                    value="Overall",
                    id="attribute-dist",
                    clearable=False,
                ),
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        html.Div(
            [
                html.H4("Filter by Nationality"),
                dcc.Dropdown(
                    df["Nationality"].to_list(), id="hist-1-filter-nationality", multi=True
                ),
                html.H4("Filter by Club Name"),
                dcc.Dropdown(clubs.to_list(), id="hist-1-filter-club", multi=True),
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        dcc.Graph(id="items-hist"),
        html.H2("Plot ages"),
        html.Div(
            [
                html.H4("Attribute"),
                dcc.Dropdown(
                    ["Wage(in Euro)", "Overall"],
                    value="Overall",
                    id="attribute",
                    clearable=False,
                ),
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        html.Br(style={"height": "30px"}),
        html.Div(
            [
                html.H4("Filter by Nationality"),
                dcc.Dropdown(
                    df["Nationality"].to_list(), id="filter_nationality", multi=True
                ),
                html.H4("Filter by Club Name"),
                dcc.Dropdown(clubs.to_list(), id="filter_club", multi=True),
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        html.Br(style={"height": "30px"}),
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
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        html.Br(style={"height": "30px"}),
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
            ],
            style={
                "textAlign": "left",
                "display": "inline-block",
                "margin-left": 20,
                "width": "90%",
            },
        ),
        html.Br(style={"margin": "100"}),
        html.H2("Compare players"),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Row Number:"),
                        dcc.Input(
                            id="number_input1",
                            type="number",
                            value=1,
                            min=0,
                            max=len(df.index),
                        ),
                        html.H3("Player Name:"),
                        dcc.Dropdown(
                            df["Full Name"].tolist(),
                            id="name_input1",
                        ),
                    ],
                    style={
                        "textAlign": "right",
                        "margin-right": 20,
                        "display": "inline-block",
                        "width": "40%",
                    },
                ),
                html.Div(
                    [
                        html.H3("Row Number:"),
                        dcc.Input(
                            id="number_input2",
                            type="number",
                            value=2,
                            min=0,
                            max=len(df.index),
                        ),
                        html.H3("Player Name:"),
                        dcc.Dropdown(
                            df["Full Name"].tolist(),
                            id="name_input2",
                        ),
                    ],
                    style={
                        "textAlign": "left",
                        "display": "inline-block",
                        "margin-left": 20,
                        "width": "40%",
                    },
                ),
                html.Br(),
                html.Div(
                    [
                        html.H3("Choose Attributes"),
                        dcc.Dropdown(
                            list(set(df.columns.to_list()) ^ set(excluded_attributes)),
                            id="attribute_input",
                            multi=True,
                        ),
                        dash_table.DataTable(
                            id="table1",
                        ),
                    ],
                    style={"text-align": "left", "margin": "auto", "width": "50%"},
                ),
                html.Div([
                    html.Img(id="img-1", src=""),
                ], style={"text-align": "left", "display": "inline-block", "margin-top": 20, "margin-right": 20}),
                html.Div([
                    html.Img(id="img-2", src=""),
                ], style={"text-align": "right", "display": "inline-block", "margin-left": 20}),
            ],
            style={
                "margin": "auto",
                "width": "50%",
                "fontFamily": "DejaVu Serif, sans-serif",
            },
        ),
    ],
    style={
        "text-align": "center",
        "margin-top": 100,
        "margin-bottom": 100,
        "margin-right": 100,
        "margin-left": 100,
        "width": "90%",
    },
)


@app.callback(
    Output("items-hist", "figure"),
    Input("attribute-dist", "value"),
    Input("hist-1-filter-nationality", "value"),
    Input("hist-1-filter-club", "value"),
)
def update_histogram(attribute, filter_nationality, filter_club):
    df_filtered = df
    color = None

    if filter_nationality:
        df_filtered = df_filtered[df_filtered["Nationality"].isin(filter_nationality)]
        color = "Nationality"
    if filter_club:
        df_filtered = df_filtered[df_filtered["Club Name"].isin(filter_club)]
        color = "Club Name"

    return px.histogram(df_filtered, x=attribute, hover_data=hover_data, color=color)


@app.callback(
    Output("table1", "data"),
    Output("table1", "columns"),
    Output("img-1", "src"),
    Output("img-2", "src"),
    Input(component_id="number_input1", component_property="value"),
    Input(component_id="number_input2", component_property="value"),
    Input(component_id="name_input1", component_property="value"),
    Input(component_id="name_input2", component_property="value"),
    Input(component_id="attribute_input", component_property="value"),
)
def update_table(input1, input2, name_input1, name_input2, attribute_input):
    if input1 is None or input2 is None or attribute_input is None:
        raise PreventUpdate

    attribute_input.insert(0, "Full Name")

    df_filtered = df.iloc[[int(input1), int(input2)]]
    df_filtered = df_filtered[attribute_input]

    df_filtered_transposed = df_filtered.transpose()
    df_filtered_transposed["Row Number"] = df_filtered_transposed.index
    data = df_filtered_transposed.to_dict("records")

    columns = [{"name": str(i), "id": str(i)} for i in df_filtered_transposed.columns]

    img1, img2 = df.loc[df_filtered.index.tolist()[0], "Image Link"], df.loc[df_filtered.index.tolist()[1], "Image Link"]

    return data, columns, img1, img2


@app.callback(
    Output("scatter-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
    Input("range-slider2", "value"),
)
def update_scatter_chart(input_value, filter_nationality, filter_club, slider_range):
    df_filtered = df
    low, high = slider_range
    mask = (df["Age"] > low) & (df["Age"] < high)
    color = None

    if filter_nationality:
        df_filtered = df_filtered[df_filtered["Nationality"].isin(filter_nationality)]
        color = "Nationality"
    if filter_club:
        df_filtered = df_filtered[df_filtered["Club Name"].isin(filter_club)]
        color = "Club Name"
    if not input_value:
        raise PreventUpdate

    return px.scatter(
        df_filtered[mask], x="Age", y=input_value, hover_data=hover_data, color=color
    )


@app.callback(
    Output("dist-plot", "figure"),
    Input("attribute", "value"),
    Input("filter_nationality", "value"),
    Input("filter_club", "value"),
    Input("range-slider", "value"),
)
def update_bar_chart(input_value, filter_nationality, filter_club, slider_range):
    df_filtered = df
    low, high = slider_range
    color = None

    if filter_nationality:
        df_filtered = df_filtered[df_filtered["Nationality"].isin(filter_nationality)]
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
        color=color,
    )


@app.callback(
    Output("number_input2", "value"),
    Output("number_input1", "value"),
    Input("name_input2", "value"),
    Input("name_input1", "value"),
)
def update_numbers_from_names(name2, name1):
    value2 = df[df["Full Name"] == name2].index[0] if name2 is not None else None

    value1 = df[df["Full Name"] == name1].index[0] if name1 is not None else None

    return value2, value1


if __name__ == "__main__":
    app.run_server(debug=True)
