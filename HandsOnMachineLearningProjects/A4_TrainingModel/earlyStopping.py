from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

if __name__ == '__main__':
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    x_train, y_train = X[: m // 2], y[: m // 2, 0]
    x_test, y_test = X[m // 2:], y[m // 2:, 0]

    plt.scatter(X, y)
    plt.show()

    model = make_pipeline(
        PolynomialFeatures(degree=90, include_bias=False),
        StandardScaler()
    )

    x_train_prep = model.fit_transform(x_train, y_train)
    x_test_prep = model.transform(x_test)

    sgd_regressor = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
    n_epochs = 500
    best_rmse = float('inf')
    best_model = None

    for epoch in range(n_epochs):
        sgd_regressor.partial_fit(x_train_prep, y_train.ravel())
        prediction = sgd_regressor.predict(x_test_prep)
        new_mse = mean_squared_error(y_test, prediction)

        if new_mse < best_rmse:
            print(f'epoch: {epoch}, best error: {new_mse}')
            best_rmse = new_mse
            best_model = deepcopy(sgd_regressor)

    #print(best_model.intercept_, best_model.coef_)

