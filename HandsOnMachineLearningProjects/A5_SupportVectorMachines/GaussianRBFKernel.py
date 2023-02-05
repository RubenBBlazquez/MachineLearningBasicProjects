import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __name__ == '__main__':
    x, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    print(x, y)

    poly_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=12, coef0=1, C=5))
    poly_kernel_svm_clf.fit(x, y)

    axes = [-1.5, 2.5, -1, 1.5]
    plt.scatter(x[:, 0][y == 0], x[:, 1][y == 0])
    plt.scatter(x[:, 0][y == 1], x[:, 1][y == 1])
    plt.grid(True)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

    rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=5, C=0.1))

    rbf_kernel_svm_clf.fit(x, y)

    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = rbf_kernel_svm_clf.predict(X).reshape(x0.shape)
    y_decision = rbf_kernel_svm_clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    plt.show()
