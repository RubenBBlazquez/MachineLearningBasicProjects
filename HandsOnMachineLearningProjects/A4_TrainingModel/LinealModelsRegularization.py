import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def plot_model(x, y, alphas, is_polynomial, test_x, scale_values):
    """
        Using Ridge Regularization, with this method we have to scale values if you want that work well
    """
    for alpha, style in zip(alphas, ['b:', 'g--', 'r-']):

        if alpha <= 0:
            model = LinearRegression()
        else:
            model = Ridge(alpha, random_state=42)

        if is_polynomial:

            if scale_values:
                model = make_pipeline(
                    PolynomialFeatures(degree=10, include_bias=False),
                    StandardScaler(),
                    model
                )
            else:
                model = make_pipeline(
                    PolynomialFeatures(degree=10, include_bias=False),
                    model
                )

        model.fit(x, y)

        y_predictions = model.predict(test_x)
        plt.plot(test_x, y_predictions, style, linewidth=2, label=fr'$\alpha = {alpha}$')


if __name__ == '__main__':
    np.random.seed(42)
    m = 20
    x = 6 * np.random.rand(m, 1) - 3
    x_test = np.linspace(-3, 3, 100).reshape(100, 1)
    y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

    plt.scatter(x, y)

    alphas = [0, 10, 100]

    plot_model(x, y, alphas, False, x_test, True)
    plt.legend()
    plt.title('Ridge Model Lineal Scaling Information')
    plt.show()

    # as we can see in this image, with a ridge of alpha 10, we get a smooth green line
    # and with alpha 0, using polynomial and linear regression we get a line with a little bit of over fitting
    plt.scatter(x, y)
    plot_model(x, y, alphas, True, x_test, True)
    plt.legend()
    plt.title('Ridge Model Polynomial Scaling Information')
    plt.show()

    # if we don't scale values, we can get something like this, and not working too well such as the previous example
    plt.scatter(x, y)
    plot_model(x, y, alphas, True, x_test, False)
    plt.legend()
    plt.title('Ridge Model Polynomial Without Scale Information')
    plt.show()

    # With Ridge Model we can use a closed-form solution from André-Louis Cholesky
    ridge_reg = Ridge(alpha=0.1, solver='cholesky')

    polynomial = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), StandardScaler(), ridge_reg)
    polynomial.fit(x, y)

    predictions = polynomial.predict(x_test)
    plt.scatter(x, y)
    plt.plot(x_test, predictions, 'r--')
    plt.title('Ridge Cholesky Model Polynomial')
    plt.show()

    # Using stochastic gradient descent with SGD Regressor
    sgd_reg = SGDRegressor(penalty='l2', alpha=0.1/m, tol=None, max_iter=1000, eta0=0.01, random_state=42)

    polynomial = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), sgd_reg)
    polynomial.fit(x, y.ravel())

    predictions = polynomial.predict(x_test)

    plt.scatter(x, y)
    plt.plot(x_test, predictions, 'r--')
    plt.title('Ridge Model With Stochastic Gradient Descent')
    plt.show()

