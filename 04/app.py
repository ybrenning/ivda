import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dash.exceptions import PreventUpdate

df = pd.read_csv("data/preprocessed.csv")
seed = 42

X = df[df.columns[:-1]].to_numpy()
y = df["target_class"].to_numpy()
X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed)


# returns svm, X_train, X_test, y_train, y_test
def train_svm(df, k, c):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target_class", axis=1),
        df["target_class"],
        test_size=0.2,
        random_state=seed,
    )

    # training
    svm = SVC(random_state=seed, kernel=k, C=float(c))
    svm.fit(X_train, y_train)

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


df_reduced = do_pca(df)

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
                html.H3("C Value"),
                dcc.Input(
                    id="c_input",
                    type="number",
                    value=0.1,
                    min=0,
                    # max=1,
                ),
                html.H3("Kernel Function:"),
                dcc.Dropdown(
                    ["linear", "poly", "rbf", "sigmoid"],
                    value="linear",
                    id="kernel_input",
                ),
            ],
        ),
        html.Div([dcc.Graph(id="svm_bar")]),
        html.Div(
            [html.H3("Scatter Plot SVM with PCA"), html.Img(id="plot_scatter")],
            className="center",
            style={
                "margin": "auto",
                "width": "80%",
                "fontFamily": "DejaVu Serif, sans-serif",
            },
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
    ],
)


@app.callback(
    Output("svm_bar", "figure"),
    [Input("kernel_input", "value"), Input("c_input", "value")],
)
def svm_bar_chart(kernel_input, c_input):
    if (
        kernel_input is None
        or c_input is None
        or not (isinstance(c_input, float) or isinstance(c_input, int))
    ):
        raise PreventUpdate
    svm, X_train, X_test, y_train, y_test = train_svm(df, kernel_input, c_input)

    df_reduced = do_pca(df)
    svm_reduced, X_r_train, X_r_test, y_train, y_test = train_svm(
        df_reduced, kernel_input, c_input
    )

    y_pred_svm = svm.predict(X_test)
    y_pred_svm_reduced = svm_reduced.predict(X_r_test)

    metrics_svm = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred_svm),
        "Precision": metrics.precision_score(y_test, y_pred_svm),
        "Recall": metrics.recall_score(y_test, y_pred_svm),
    }

    metrics_svm_reduced = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred_svm_reduced),
        "Precision": metrics.precision_score(y_test, y_pred_svm_reduced),
        "Recall": metrics.recall_score(y_test, y_pred_svm_reduced),
    }

    # Create a new DataFrame for the bar chart with both SVM and PCA metrics
    new_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall"] * 2,
            "Value": list(metrics_svm.values()) + list(metrics_svm_reduced.values()),
            "Model": ["SVM"] * 3 + ["SVM with PCA"] * 3,
        }
    )

    # Use Plotly to create the grouped bar chart
    fig = px.bar(
        new_df,
        x="Metric",
        y="Value",
        hover_data="Value",
        color="Metric",
        color_discrete_map={
            "Accuracy": "blue",
            "Precision": "green",
            "Recall": "orange",
        },
        facet_col="Model",
        title="SVM Metrics",
        facet_col_spacing=0.08,
    )

    return fig


@app.callback(
    Output("plot_scatter", "src"),
    [Input("kernel_input", "value"), Input("c_input", "value")],
)
def plot_scatter(kernel_input, c_input):
    if kernel_input is None or c_input is None:
        raise PreventUpdate
    svm_reduced, X_r_train, X_r_test, y_r_train, y_r_test = train_svm(
        df_reduced, kernel_input, c_input
    )

    X_reduced = df_reduced.drop("target_class", axis=1)
    y = df_reduced["target_class"]

    def make_meshgrid(x, y, h=0.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    # title for the plots
    title = (
        "Decision surface of "
        + kernel_input
        + " SVM with PCA(n=2) and C="
        + str(c_input)
    )
    # Set-up grid for plotting.
    X0, X1 = X_reduced["PC1"], X_reduced["PC2"]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, svm_reduced, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend("Kernel Function: " + kernel_input)

    # Convert the Matplotlib plot to a base64-encoded image
    with BytesIO() as img_buf:
        fig.savefig(img_buf, format="png")
        # img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getbuffer()).decode("ascii")
        fig_scat_matplotlib = f"data:image/png;base64,{img_data}"

        return fig_scat_matplotlib


if __name__ == "__main__":
    app.run_server(debug=True)
