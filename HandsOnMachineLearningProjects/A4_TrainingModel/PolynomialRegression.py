import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

if __name__ == '__main__':
    np.random.seed(42)
    m = 100
    x = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

    plt.scatter(x, y)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)

    x_poly = poly_features.fit_transform(x)

    # x_poly return the value of the x and the value of x**2, since its 2 degree
    print(x[0], x_poly[0])

    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    # so we can see the model predict y = 0.56x^2 + 0.93x + 1.78 ,
    # which is near to our y defined y = 0.5 * x ** 2 + x + 2

    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)

    plt.plot(X_new, y_new, color='r')
    plt.show()

    # now we look how a 300 degree line could be, we can see that with that high degree
    # we get over fitting

    plt.figure(figsize=(6, 4))

    for style, width, degree in (("r-+", 2, 1), ("b--", 2, 2), ("g-", 1, 300)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)

        std_scaler = StandardScaler()
        lin_reg = LinearRegression()

        polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg)
        polynomial_regression.fit(x, y)

        y_newbig = polynomial_regression.predict(X_new)
        label = f"{degree} degree{'s' if degree > 1 else ''}"
        plt.plot(X_new, y_newbig, style, label=label, linewidth=width)

    plt.plot(x, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([-3, 3, 0, 10])
    plt.grid()
    plt.show()
