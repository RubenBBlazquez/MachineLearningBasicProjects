import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import add_dummy_feature


if __name__ == '__main__':
    np.random.seed(42)
    m = 100
    x = 2 * np.random.rand(m, 1)
    y = 4 + 3 * x + np.random.randn(m, 1)

    x_b = add_dummy_feature(x)

    eta = 0.1
    n_epoch = 1000
    m = len(x_b)

    theta = np.random.rand(2, 1)

    for epoch in range(n_epoch):
        gradients = 2/m * x_b.T @ (x_b @ theta - y)
        theta -= eta * gradients

    print(theta)