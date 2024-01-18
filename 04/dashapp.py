import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html

np.random.seed(42)
df = pd.read_csv("data/pulsar_data.csv")
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

# create training and test Split
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
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_labels], df["target_class"], test_size=0.2, random_state=42
)

# TODO Kernel Method Interaktiv waehlbar
svm = SVC(random_state=42)

# Train the model using the training sets
svm.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = svm.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))

# for Visual
pca = PCA(n_components=2)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# dimension reduction on train data
X_train_reduced = pca.fit_transform(X_train_scaled)

# Create a DataFrame with reduced data and target variable
df_train = pd.DataFrame(X_train_reduced, columns=["PC1", "PC2"])
df_train = df_train.assign(target_class=y_train.values)

# Build App
app = Dash(__name__)
app.layout = html.Div(
    [
        # Create a scatter plot with decision boundary
        dcc.Graph(
            figure=px.scatter(
                df_train,
                x="PC1",
                y="PC2",
                color="target_class",
                title="SVM Decision Boundary in Reduced 2D Space",
            )
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
