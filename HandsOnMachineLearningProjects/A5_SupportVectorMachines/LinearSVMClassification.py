import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    # Linear
    iris = load_iris(as_frame=True)
    x = iris.data[['petal length (cm)', 'petal width (cm)']]
    y = iris.target == 2

    svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
    svm_clf.fit(x, y)

    x_new = [[5.5, 1.7], [5.0, 1.5]]
    predictions = svm_clf.predict(x_new)
    print(predictions)

    decisions = svm_clf.decision_function(x_new)
    print(decisions)

    # Polynomial
    x, y = make_moons(n_samples=100, noise=0.15, random_state=42)

    axes = [-1.5, 2.5, -1, 1.5]
    plt.scatter(x[:, 0][y == 0], x[:, 1][y == 0])
    plt.scatter(x[:, 0][y == 1], x[:, 1][y == 1])
    plt.grid(True)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

    polynomial_svm_clf = make_pipeline(
        PolynomialFeatures(degree=3),
        StandardScaler(),
        LinearSVC(random_state=42, C=10, max_iter=10000)
    )

    polynomial_svm_clf.fit(x, y)

    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = polynomial_svm_clf.predict(X).reshape(x0.shape)
    y_decision = polynomial_svm_clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    plt.show()

