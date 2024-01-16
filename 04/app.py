import pandas as pd
import plotly.express as px

from dash import Dash, Input, Output, dcc, html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC


df = pd.read_csv('data/preprocessed.csv')

X = df[df.columns[:-1]].to_numpy()
y = df['target_class'].to_numpy()
X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    random_state=42
)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Submission 2 (Yannick Brenning, Yannik Lange)"),

    html.Div([
        html.H4("Select Parameter to visualize"),
        dcc.Dropdown(
            ["C", "degree", "gamma"],
            value="C",
            id="parameter"
        )
    ]),

    html.Div([
        html.H4("Parameter value comparison"),
        dcc.Graph(id="params-plot")
    ])
])


@app.callback(
    Output("params-plot", "figure"),
    Input("parameter", "value")
)
def plot_params(parameter):
    if parameter == "C":
        cs = [10**x for x in range(-2, 4)]
        scores = []
        for c in cs:
            svc = SVC(C=c, random_state=42)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))

        df_plot = pd.DataFrame({"C": [str(c) for c in cs], "score": scores})
        return px.bar(
            df_plot,
            x="C",
            y="score",
            text=[round(score, 2) for score in scores]
        )

    elif parameter == "degree":
        degrees = list(range(1, 7))
        scores = []
        for degree in degrees:
            svc = SVC(kernel="poly", degree=degree, random_state=42)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))

        df_plot = pd.DataFrame(
            {
                "degree": [str(d) for d in degrees],
                "score": scores
            }
        )
        return px.bar(
            df_plot,
            x="degree",
            y="score",
            text=[round(score, 2) for score in scores]
        )

    elif parameter == "gamma":
        gammas = [10**x for x in range(-2, 3)]
        scores = []
        for gamma in gammas:
            svc = SVC(gamma=gamma, random_state=42)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))

        df_plot = pd.DataFrame(
            {
                "gamma": [str(g) for g in gammas],
                "score": scores
            }
        )
        return px.bar(
            df_plot,
            x="gamma",
            y="score",
            text=[round(score, 2) for score in scores]
        )


if __name__ == '__main__':
    app.run_server(debug=True)
