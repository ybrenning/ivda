from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
app = Dash(__name__)


app.layout = html.Div([
    html.H4('Interactive scatter plot with Our dataset'),
    dcc.Graph(id="scatter-plot"),
    html.P("Filter by petal width:"),
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=60, step=1,
        marks={10:'10',15: '15',20:'20',25: '25',30:'30',35: '35',40:'40',45: '45'},
        value=[10, 45]
    ),
])


@app.callback(
    Output("scatter-plot", "figure"), 
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    df = pd.read_csv('ivda/01/data/Aufgabe-1.csv', on_bad_lines='skip')
    low, high = slider_range
    mask = (df['\'Age\''] > low) & (df['\'Age\''] < high)
    fig = px.scatter(
        df[mask], x='\'Age\'', y="\'Overall\'", size='\'Overall\'',color='\'Wage(in Euro)\'',
        hover_data=['\'Age\''])
    return fig


app.run_server(debug=True)