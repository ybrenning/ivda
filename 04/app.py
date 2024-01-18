import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, Input, Output, dcc, html
from sklearn.neural_network import MLPClassifier
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

mlp1 = MLPClassifier(
    hidden_layer_sizes=(4,),
    max_iter=1000,
    random_state=42
)

mlp2 = MLPClassifier(
    hidden_layer_sizes=(16, 16,),
    max_iter=1000,
    random_state=42
)

mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)


def plot_loss_curves():
    fig = go.Figure()
    y1 = mlp1.loss_curve_
    y2 = mlp2.loss_curve_

    fig.add_trace(
        go.Scatter(x=list(range(0, len(y1))), y=y1, name="Neural Network 1"),
        # row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(range(0, len(y2))), y=y2, name="Neural Network 2"),
        # row=1, col=2
    )

    fig.update_layout(
        height=600,
        width=1000,
        xaxis_title="Iteration",
        yaxis_title="Loss"
    )
    return fig


def show_weights(mlp):
    fig = px.imshow(
        mlp.coefs_[0],
        color_continuous_scale='gray',
        labels={"color": "weight"}
    )

    fig.update_layout(xaxis_title="Hidden Layer", yaxis_title="Input Layer")
    return fig


def plot_learning_curve(mlp_name):
    fig = go.Figure()

    train_scores = np.load(f"assets/train_{mlp_name}.npy")
    test_scores = np.load(f"assets/test_{mlp_name}.npy")
    train_size_abs = np.load(f"assets/train_abs_{mlp_name}.npy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig.add_trace(go.Scatter(
        x=train_size_abs, y=train_scores_mean,
        error_y=dict(array=train_scores_std),
        line_color='rgb(0,100,80)',
        name='Train',
    ))

    fig.add_trace(go.Scatter(
        x=train_size_abs, y=test_scores_mean,
        error_y=dict(array=test_scores_std),
        line_color='rgb(0,176,246)',
        name='Cross Val',
    ))

    if mlp_name == "1":
        yrange = [0.95, 1.]
    else:
        yrange = [0.8, 1.]

    fig.update_yaxes(range=yrange)
    fig.update_traces(mode='lines')

    fig.update_layout(
        title=f"Neural Network {mlp_name}",
        height=400,
        width=600,
        xaxis_title="No. of instances",
        yaxis_title="Accuracy"
    )
    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Submission 2 (Yannick Brenning, Yannik Lange)"),
    ], className="center"),

    html.Div([
        html.H2("Classification of Pulsar Stars using SVM"),
    ], className="center"),

    html.Div([
        html.H4("Select Parameter to visualize"),
        html.Div([
            dcc.Dropdown(
                ["C", "degree", "gamma"],
                value="C",
                id="parameter"
            )
        ], style={"width": "25%", "margin": "auto"})
    ], className="center"),

    html.Div([
        html.H4("Parameter value comparison"),
        dcc.Graph(id="params-plot")
    ], className="center"),

    html.Div([
        html.H2("Comparison of Neural Networks"),
    ], className="center"),

    html.Div([
        html.H3("Neural Network Topologies"),
    ], className="center"),

    html.Div([
        html.Img(src=app.get_asset_url("nn-1.png"), className="image"),
        html.Img(src=app.get_asset_url("nn-2.png"), className="image"),
    ], className="container"),

    html.Div([
        html.Div([
            html.H4("Loss Curves"),
        ], className="center"),
        dcc.Graph(
            figure=plot_loss_curves(),
            id="loss-curves",
        )
    ], style={'display': 'flex', 'justify-content': 'center'}),

    html.Div([
        html.Div([
            html.H4("Learning Curves"),
        ], className="center"),
        dcc.Graph(figure=plot_learning_curve("1"), id="learning-curve-1"),
        dcc.Graph(figure=plot_learning_curve("2"), id="learning-curve-2")
    ], style={'display': 'flex', 'justify-content': 'center'}),

    html.Div([
        html.H4("First layer weights: Neural Network 1"),
        dcc.Graph(figure=show_weights(mlp1), id="weights-1")
    ], className="center"),

    html.Div([
        html.H4("First layer weights: Neural Network 2"),
        dcc.Graph(figure=show_weights(mlp2), id="weights-2")
    ], className="center")

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
            text=[round(score, 3) for score in scores]
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
            text=[round(score, 3) for score in scores]
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
            text=[round(score, 3) for score in scores]
        )


if __name__ == '__main__':
    app.run_server(debug=True)