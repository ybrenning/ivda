import os
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, dash_table
from dash.exceptions import PreventUpdate

DATA_PATH = os.getcwd() + "/01/data/processed.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

app = Dash(__name__)

hover_data = ["Full Name", "Wage(in Euro)", "Overall", "Nationality"]
nationalities = df["Nationality"]
clubs = df["Club Name"]
excluded_attributes = [
    "Unnamed: 0",
    "Known As",
    "Full Name",
    "National Team Image Link",
    "Image Link",
    "Full Name",
]

# Spieler vergleich schoener machen,
# Anhand der Namen oder Zeilennummer auwähelen
# Spielerbild erscheint
# Dropdown mit Attribut auswählbar die angezeigt werden sollen

app.layout = html.Div(
    [
        html.H1("Abgabe 1 von Yannick Brenning und Yannik Lange"),
        html.Br(style={"height": "30px"}),
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
                    nationalities.to_list(), id="filter_nationality", multi=True
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
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Row Number:"),
                        dcc.Input(
                            id="number_input1",
                            type="number",
                            value=1,
                            min=1,
                            max=len(df.index),
                        ),
                        html.H3("Player Name: (overwrites Row Number)"),
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
                            min=1,
                            max=len(df.index),
                        ),
                        html.H3("Player Name: (overwrites Row Number)"),
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
            ],
            style={
                "margin": "auto",
                "width": "50%",
                'fontFamily': 'DejaVu Serif, sans-serif'
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
    Output("table1", "data"),
    Output("table1", "columns"),
    Output("table1", "style_data_conditional"),
    Input(component_id="number_input1", component_property="value"),
    Input(component_id="number_input2", component_property="value"),
    Input(component_id="name_input1", component_property="value"),
    Input(component_id="name_input2", component_property="value"),
    Input(component_id="attribute_input", component_property="value"),
)
def update_table(input1, input2, name_input1, name_input2, attribute_input):
    if not input1 or not input2 or not attribute_input:
        raise PreventUpdate
    attribute_input.insert(0, "Full Name")
    if name_input1:
        input1 = np.where(df["Full Name"] == name_input1)[0][0]

    if name_input2:
        input2 = np.where(df["Full Name"] == name_input2)[0][0]

    df_filtered = df.iloc[[int(input1), int(input2)]]
    df_filtered = df_filtered[attribute_input]

    df_filtered_transposed = df_filtered.transpose()
    df_filtered_transposed["Row Number"] = df_filtered_transposed.index
    style_data_conditional = []
    """
    style_data_conditional = [
        {
            "if": {
                "filter_query": "{{{col}}} = {}".format(i,col=col),
                "column_id": col,
            },
            "backgroundColor": "rgb(144, 238, 144)",  # Light green
            "color": "black",
        }
        # idxmax(axis=1) finds the max indices of each row
        for (i, col) in enumerate(df_numeric_columns.map(lambda x: pd.to_numeric(x, errors='coerce')).select_dtypes("number").dropna().idxmax(axis=1)
)
    ]
    """

    data = df_filtered_transposed.to_dict("records")
    columns = [{"name": str(i), "id": str(i)} for i in df_filtered_transposed.columns]
    return data, columns, style_data_conditional


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
        # TODO: wie oft kommt jedes attribut vor
        df_filtered[mask],
        x="Age",
        y=input_value,
        hover_data=hover_data,
        color=color,
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
