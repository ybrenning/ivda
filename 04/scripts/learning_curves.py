import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale

df = pd.read_csv('../data/preprocessed.csv')

X = df[df.columns[:-1]].to_numpy()
y = df['target_class'].to_numpy()
X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    random_state=42
)

if __name__ == "__main__":
    mlp1 = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
    mlp2 = MLPClassifier(hidden_layer_sizes=(16, 16,), max_iter=1000, random_state=42)
    train_size_abs, train_scores, test_scores = learning_curve(mlp1, X_scaled, y, train_sizes=np.linspace(0.1, 1.0, 10))
    np.save('../assets/train_abs_1.npy', train_size_abs)
    np.save('../assets/train_1.npy', train_scores)
    np.save('../assets/test_1.npy', test_scores)
    train_size_abs, train_scores, test_scores = learning_curve(mlp2, X_scaled, y, train_sizes=np.linspace(0.1, 1.0, 10))
    np.save('../assets/train_abs_2.npy', train_size_abs)
    np.save('../assets/train_2.npy', train_scores)
    np.save('../assets/test_2.npy', test_scores)
