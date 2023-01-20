import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature

if __name__ == '__main__':
    np.random.seed(42)
    m = 100
    x = 2 * np.random.rand(m, 1)
    y = 4 + 3 * x + np.random.randn(m, 1)

    plt.scatter(x, y)

    x_b = add_dummy_feature(x)

    # now we calculate the normal equation < (XT * X)-1 * XT * Y >, we use @ to multiply matrix's
    theta_best = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
    print('theta best from formula of normal equation: ', theta_best)

    # now we can make predictions using our best theta
    x_new = np.array([[0], [2]])
    x_new_b = add_dummy_feature(x_new)

    y_predict = x_new_b @ theta_best

    plt.plot(x_new, y_predict, color='r')
    plt.yticks(range(0, 16, 2))
    plt.show()

    # now we can use the linear regression to get the same theta best values
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    intercept_lin_reg = lin_reg.intercept_
    coefficients = lin_reg.coef_

    print('theta best from linear regression: ', intercept_lin_reg, coefficients)

    # this is the method that the linear regression model uses internally
    # using the formula theta = X+ Y, where X+ is the pseudo-inverse of X
    theta_best_svd, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=1e-6)

    # we can use this to compute pseudo-inverse directly, sometimes
    # you cant use normal equation to get the best theta because have a singular matrix
    # or m < n, so in that cases we can use this pseudo-inverse to get the equation
    theta_best_pinv = np.linalg.pinv(x_b) @ y


