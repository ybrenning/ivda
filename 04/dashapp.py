import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html


def prepare_data(df):
    # df.head()

    # delete leading spaces in column names and
    # replace the rest of the spaces with '_'
    df.columns = df.columns.str.strip()

    df.columns = [c.replace(" ", "_") for c in df.columns]

    # since they all seem pretty much normaly distributed,
    # it seams reasonable to fill the missing values with
    # median since there seem to be some outliers

    for key in [
        "Excess_kurtosis_of_the_integrated_profile",
        "Standard_deviation_of_the_DM-SNR_curve",
        "Skewness_of_the_DM-SNR_curve",
    ]:
        df[key].fillna(df[key].median(), inplace=True)
    return df


def do_pca(df):
    # TODO Kernel Method Interaktiv waehlbar

    pca = PCA(n_components=2)

    # Standardize the data for pcs
    feature_labels = [
        "Mean_of_the_integrated_profile",
        "Standard_deviation_of_the_integrated_profile",
        "Excess_kurtosis_of_the_integrated_profile",
        "Skewness_of_the_integrated_profile",
        "Mean_of_the_DM-SNR_curve",
        "Standard_deviation_of_the_DM-SNR_curve",
        "Excess_kurtosis_of_the_DM-SNR_curve",
        "Skewness_of_the_DM-SNR_curve",
    ]

    X = df[feature_labels].to_numpy()
    y = df["target_class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # dimension reduction
    X_reduced = pca.fit_transform(X_scaled)

    # Create a DataFrame with reduced data and target variable
    df_reduced = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    df_reduced = df_reduced.assign(target_class=y)
    return df_reduced


def do_svm(df_reduced):
    # prepare svm

    X_train, X_test, y_train, y_test = train_test_split(
        df_reduced.drop("target_class", axis=1),
        df_reduced["target_class"],
        test_size=0.2,
        random_state=seed,
    )
    # training
    svm = SVC(random_state=seed)
    svm.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = svm.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    df_test = pd.DataFrame(X_test, columns=["PC1", "PC2"])
    df_test = df_test.assign(target_class=y_test.values)

    df_train = pd.DataFrame(X_train, columns=["PC1", "PC2"])
    df_train = df_test.assign(target_class=y_test.values)

    return (svm, df_train, df_reduced)


def create_svm_scatter(svm, df_train):
    # Create a meshgrid to represent the decision boundary
    x_min, x_max = df_train["PC1"].min() - 1, df_train["PC1"].max() + 1
    y_min, y_max = df_train["PC2"].min() - 1, df_train["PC2"].max() + 1
    h = 0.02  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain decision values for each point in the meshgrid
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a filled contour plot
    fig = px.scatter(
        df_train,
        x="PC1",
        y="PC2",
        color="target_class",
        title="SVM Decision Boundary in Reduced 2D Space",
    )
    fig = fig.add_contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        contours=dict(coloring="lines", showlabels=False),
        line=dict(width=0),
        colorscale="Viridis",
        showscale=False,
    )
    return fig


seed = 42
np.random.seed(seed)
df = pd.read_csv("data/pulsar_data.csv")
print(df.head())
df = prepare_data(df)
print(df.head())
df_reduced = do_pca(df)
print(df_reduced.head())
svm, df_train, df_test = do_svm(df_reduced)
fig = create_svm_scatter(svm, df_train)

# Build App
app = Dash(__name__)
app.layout = html.Div(
    [
        # Create a scatter plot with decision boundary
        dcc.Graph(figure=fig)
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
