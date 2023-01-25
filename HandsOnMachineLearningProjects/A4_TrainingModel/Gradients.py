import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import add_dummy_feature


def learning_schedule(t):
    """
        Method to regulate learning rate,
        t0 and t1 are schedule hyperparameter

        param t: actual learning rate
    """
    t0, t1 = 5, 50

    return t0 / (t + t1)


if __name__ == '__main__':
    np.random.seed(42)
    m = 100
    x = 2 * np.random.rand(m, 1)
    y = 4 + 3 * x + np.random.randn(m, 1)

    plt.scatter(x, y)

    x_b = add_dummy_feature(x)

    # Gradient Descent ##
    eta = 0.1
    n_epoch = 1000
    m = len(x_b)
    theta = np.random.rand(2, 1) # this is theta or synaptic weights to do the predictions

    for epoch in range(n_epoch):
        prediction = x_b @ theta
        mse = prediction - y
        gradients = 2 / m * x_b.T @ mse

        # we plot all tries
        plt.plot(x, prediction)
        theta -= eta * gradients

    print(theta)
    plt.title('Batch Gradient: 4 + 3x')
    plt.show()

    # Stochastic Gradient ##
    plt.scatter(x, y)

    x_new = np.array([[0], [2]])
    x_new_b = add_dummy_feature(x_new)

    n_epoch = 50
    theta = np.random.rand(2, 1)

    for epoch in range(n_epoch):
        for iteration in range(m):
            random_index = np.random.randint(m)

            xi = x_b[random_index: random_index + 1]
            yi = y[random_index: random_index + 1]

            prediction = xi @ theta
            mse = prediction - yi
            gradients = 2 * xi.T @ mse

            t = epoch * m + iteration
            eta = learning_schedule(t)
            plt.plot(x_new, np.dot(x_new_b, theta))

            theta -= eta * gradients

    plt.title('Stochastic Gradient')
    plt.show()

    print()
    print(theta)

    # to use stochastic gradient with a model we can use SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=100, random_state=42)
    sgd_reg.fit(x, y.ravel())
    print()
    print(sgd_reg.intercept_, sgd_reg.coef_)