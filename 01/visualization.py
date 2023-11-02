from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
df = pd.read_csv('ivda/01/data/Aufgabe-1.csv', on_bad_lines='skip')
app = Dash(__name__)


app.layout = html.Div([
    html.H4('Interactive scatter plot with the dataset'),
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
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df['\'Age\''] > low) & (df['\'Age\''] < high)
    fig = px.scatter(
        df[mask], x='\'Age\'', y="\'Wage(in Euro)\'", size='\'Overall\'',color='\'Overall\'',
        hover_data=['\'Full Name\''])
    return fig


app.run_server(debug=True)