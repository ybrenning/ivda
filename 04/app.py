import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

df = pd.read_csv("data/preprocessed.csv")
seed = 42

# X = df[df.columns[:-1]].to_numpy()
# y = df['target_class'].to_numpy()
# X_scaled = scale(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled,
#     y,
#     random_state=seed
# )


# returns svm, X_train, X_test, y_train, y_test
def train_svm(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target_class", axis=1),
        df["target_class"],
        test_size=0.2,
        random_state=seed,
    )

    # training
    svm = SVC(random_state=seed)
    svm.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = svm.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    return svm, X_train, X_test, y_train, y_test


def do_pca(df):
    pca = PCA(n_components=2)

    X = df.drop("target_class", axis=1).to_numpy()
    y = df["target_class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # dimension reduction
    X_reduced = pca.fit_transform(X_scaled)

    # Create a DataFrame with reduced data and target variable
    df_reduced = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    df_reduced = df_reduced.assign(target_class=y)
    return df_reduced


def plot_loss_curves():
    fig = go.Figure()
    y1 = mlp1.loss_curve_
    y2 = mlp2.loss_curve_

    fig.add_trace(go.Scatter(x=list(range(0, len(y1))), y=y1, name="Neural Network 1"))

    fig.add_trace(
        go.Scatter(x=list(range(0, len(y2))), y=y2, name="Neural Network 2"),
    )

    fig.update_layout(
        height=600, width=1000, xaxis_title="Iteration", yaxis_title="Loss"
    )
    return fig


def show_weights(mlp):
    fig = px.imshow(
        mlp.coefs_[0], color_continuous_scale="gray", labels={"color": "weight"}
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

    fig.add_trace(
        go.Scatter(
            x=train_size_abs,
            y=train_scores_mean,
            error_y=dict(array=train_scores_std),
            line_color="rgb(0,100,80)",
            name="Train",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_size_abs,
            y=test_scores_mean,
            error_y=dict(array=test_scores_std),
            line_color="rgb(0,176,246)",
            name="Cross Val",
        )
    )

    yrange = [0.965, 0.985]

    fig.update_yaxes(range=yrange)
    fig.update_traces(mode="lines")

    fig.update_layout(
        title=f"Neural Network {mlp_name}",
        height=500,
        width=600,
        xaxis_title="No. of instances",
        yaxis_title="Accuracy",
    )
    return fig


def plot_scores(mlps):
    mlp1, mlp2 = mlps
    mlp1.fit(X_train, y_train)
    y_preds_1 = mlp1.predict(X_test)

    mlp2.fit(X_train, y_train)
    y_preds_2 = mlp1.predict(X_test)

    y_preds = [y_preds_1, y_preds_2]
    names = ["Neural Network 1", "Neural Network 2"]
    classes = ["Not Pulsar Star", "Pulsar Star"]

    df = pd.DataFrame()
    for i in range(len(names)):
        report = classification_report(
            y_test, y_preds[i], target_names=classes, output_dict=True
        )

        # Extract metrics for each class
        precision = [report[class_name]["precision"] for class_name in classes]
        recall = [report[class_name]["recall"] for class_name in classes]
        f1_score = [report[class_name]["f1-score"] for class_name in classes]

        model_df = pd.DataFrame(
            {
                "Class": classes * 3,
                "Metric": ["Precision"] * len(classes)
                + ["Recall"] * len(classes)
                + ["F1-Score"] * len(classes),
                "Score": precision + recall + f1_score,
                "Model": [names[i]] * (3 * len(classes)),
            }
        )

        df = pd.concat([df, model_df], ignore_index=True)

    fig = px.bar(
        df,
        x="Class",
        y="Score",
        color="Metric",
        facet_col="Model",
        color_discrete_map={
            "Precision": "blue",
            "Recall": "green",
            "F1-Score": "orange",
        },
        labels={"Score": "Score", "variable": "Metric", "x": "Class"},
        title="Interactive Model Performance by Class",
        barmode="group",
    )

    fig.update_layout(yaxis_title="Score", xaxis_title="Class")
    return fig


## returns mlp1, mlp2
def do_mlp(X_train, y_train):
    mlp1 = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=seed)
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(
            16,
            16,
        ),
        max_iter=1000,
        random_state=seed,
    )

    mlp1.fit(X_train, y_train)
    mlp2.fit(X_train, y_train)
    return mlp1, mlp2


svm, X_train, X_test, y_train, y_test = train_svm(df)

# df_reduced = do_pca(df)

mlp1, mlp2 = do_mlp(X_train, y_train)

app = Dash(__name__)

# TODO: Show scatter w/ SVM hyperplane
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Submission 4 (Yannick Brenning, Yannik Lange)"),
            ],
            className="center",
        ),
        html.Div(
            [
                html.H2("Classification of Pulsar Stars using SVM"),
            ],
            className="center",
        ),
        html.Div(
            [
                html.H4("Select Parameter to visualize"),
                html.Div(
                    [dcc.Dropdown(["C", "degree", "gamma"], value="C", id="parameter")],
                    style={"width": "25%", "margin": "auto"},
                ),
            ],
            className="center",
        ),
        html.Div(
            [html.H4("Parameter value comparison"), dcc.Graph(id="params-plot")],
            className="center",
        ),
        html.Div(
            [html.H4("Scatter Plot SVM"), dcc.Graph(id="scatter-plot")],
            className="center",
        ),
        html.Div(
            [
                html.H2("Comparison of Neural Networks"),
            ],
            className="center",
        ),
        html.Div(
            [
                html.H3("Neural Network Topologies"),
            ],
            className="center",
        ),
        html.Div(
            [
                html.Img(src=app.get_asset_url("nn-1.png"), className="image"),
                html.Img(src=app.get_asset_url("nn-2.png"), className="image"),
            ],
            className="container",
        ),
        dcc.Graph(figure=plot_scores((mlp1, mlp2)), id="plot-scores"),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Loss Curves"),
                    ],
                    className="center",
                ),
                dcc.Graph(
                    figure=plot_loss_curves(),
                    id="loss-curves",
                ),
            ],
            style={"display": "flex", "justify-content": "center"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Learning Curves"),
                    ],
                    className="center",
                ),
                dcc.Graph(figure=plot_learning_curve("1"), id="learning-curve-1"),
                dcc.Graph(figure=plot_learning_curve("2"), id="learning-curve-2"),
            ],
            style={"display": "flex", "justify-content": "center"},
        ),
        html.Div(
            [
                html.H4("First layer weights: Neural Network 1"),
                dcc.Graph(figure=show_weights(mlp1), id="weights-1"),
            ],
            className="center",
        ),
        html.Div(
            [
                html.H4("First layer weights: Neural Network 2"),
                dcc.Graph(figure=show_weights(mlp2), id="weights-2"),
            ],
            className="center",
        ),
    ]
)


@app.callback(Output("scatter-plot", "figure"), Input("parameter", "value"))
def plot_scatter(parameter):
    # Extract features and target
    features = df.drop("target_class", axis=1)
    target = df["target_class"]

    # Create a mesh grid to plot the decision boundary
    h = 0.02  # Step size in the mesh

    # Create a list to store mesh grids for each pair of features
    mesh_grids = []

    for i in range(features.shape[1]):
        for j in range(i + 1, features.shape[1]):
            x_min, x_max = features.iloc[:, i].min() - 1, features.iloc[:, i].max() + 1
            y_min, y_max = features.iloc[:, j].min() - 1, features.iloc[:, j].max() + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            mesh_grids.append((xx, yy))

    # Create a scatter plot matrix
    fig = go.Figure()

    # Add scatter plots for each pair of features
    for i in range(features.shape[1]):
        for j in range(i + 1, features.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=features.iloc[:, i],
                    y=features.iloc[:, j],
                    mode="markers",
                    marker=dict(
                        color=target,
                        colorscale="Viridis",
                        size=8,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False,
                )
            )

            # Add decision boundary to the plot
            mesh_predictions = svm.predict(
                np.c_[mesh_grids.pop(0)[0].ravel(), mesh_grids.pop(0)[1].ravel()]
            )
            mesh_predictions = mesh_predictions.reshape(mesh_grids.pop(0)[0].shape)

            fig.add_trace(
                go.Contour(
                    x=mesh_grids.pop(0)[0].ravel(),
                    y=mesh_grids.pop(0)[1].ravel(),
                    z=mesh_predictions,
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.5,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="SVM Decision Boundary",
        dragmode="select",
        width=1800,
        height=1800,
        hovermode="closest",
    )

    return fig


# def plot_scatter(parameter):
#     # Create a meshgrid to represent the decision boundary
#     x_min, x_max = df_reduced["PC1"].min() - 1, df_reduced["PC1"].max() + 1
#     y_min, y_max = df_reduced["PC2"].min() - 1, df_reduced["PC2"].max() + 1
#     h = 0.02  # Step size in the mesh
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     # Obtain decision values for each point in the meshgrid
#     Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     # Create a filled contour plot
#     fig = px.scatter(
#         df_reduced,
#         x="PC1",
#         y="PC2",
#         color="target_class",
#         title="SVM Decision Boundary in Reduced 2D Space",
#     )
#     fig = fig.add_contour(
#         x=np.arange(x_min, x_max, h),
#         y=np.arange(y_min, y_max, h),
#         z=Z,
#         contours=dict(coloring="lines", showlabels=False),
#         line=dict(width=0),
#         colorscale="Viridis",
#         showscale=False,
#     )
#     return fig


# TODO: Show precision, recall, F1??
@app.callback(Output("params-plot", "figure"), Input("parameter", "value"))
def plot_params(parameter):
    if parameter == "C":
        cs = [10**x for x in range(-2, 4)]
        scores = []
        for c in cs:
            svc = SVC(C=c, random_state=seed)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))
        #
        df_plot = pd.DataFrame({"C": [str(c) for c in cs], "score": scores})
        return px.bar(
            df_plot, x="C", y="score", text=[round(score, 3) for score in scores]
        )

    elif parameter == "degree":
        degrees = list(range(1, 7))
        scores = []
        for degree in degrees:
            svc = SVC(kernel="poly", degree=degree, random_state=seed)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))

        df_plot = pd.DataFrame({"degree": [str(d) for d in degrees], "score": scores})
        return px.bar(
            df_plot, x="degree", y="score", text=[round(score, 3) for score in scores]
        )

    elif parameter == "gamma":
        gammas = [10**x for x in range(-2, 3)]
        scores = []
        for gamma in gammas:
            svc = SVC(gamma=gamma, random_state=seed)
            svc.fit(X_train, y_train)
            scores.append(svc.score(X_test, y_test))

        df_plot = pd.DataFrame({"gamma": [str(g) for g in gammas], "score": scores})
        return px.bar(
            df_plot, x="gamma", y="score", text=[round(score, 3) for score in scores]
        )


if __name__ == "__main__":
    app.run_server(debug=True)
