import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    np.random.seed(42)
    m = 2000
    x = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

    # to see if a model is good, over fitting or under fitting we use the learning_curve method

    # Example 1 Under Fitting
    train_sizes, train_scores, valid_scored = learning_curve(
        LinearRegression(), x, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5, scoring='neg_root_mean_squared_error'
    )

    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scored.mean(axis=1)

    # as we can see in this graphic, our model is under fitting because
    # train and valid end up to a plateau, and we can see that valid error starts in the top
    # which means that is not generalizing properly and after that the valid error end up at a plateau
    plt.plot(train_sizes, train_errors, 'r-+', label='train')
    plt.plot(train_sizes, valid_errors, 'b-', label='valid')
    plt.yticks(np.arange(0, 4, 0.5))
    plt.ylabel('$RMSE(x)$')
    plt.xlabel('train set size')
    plt.legend()
    plt.show()

    # Example 2 Over Fitting
    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        LinearRegression()
    )

    train_sizes, train_scores, valid_scores = learning_curve(
        polynomial_regression, x, y, train_sizes=np.linspace(0.01, 1.0, 40),
        cv=5, scoring="neg_root_mean_squared_error"
    )

    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)

    # in this example we can see a case of over fitting because
    # the train error is lower than the valid error and the model is not working fine
    # one way to improve this model, could be feed with more that until the validation error reaches the training error
    plt.plot(train_sizes, train_errors, 'r-+', label='train')
    plt.plot(train_sizes, valid_errors, 'b-', label='valid')
    plt.ylabel('$RMSE(x)$')
    plt.xlabel('train set size')
    plt.axis([0, 80, 0, 2.5])
    plt.legend()
    plt.show()

    # Example 3 Good Work
    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )

    train_sizes, train_scores, valid_scores = learning_curve(
        polynomial_regression, x, y, train_sizes=np.linspace(0.01, 1.0, 40),
        cv=5, scoring="neg_root_mean_squared_error"
    )

    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)

    # in this example we can see how the valid error and
    plt.plot(train_sizes, train_errors, 'r-+', label='train')
    plt.plot(train_sizes, valid_errors, 'b-', label='valid')
    plt.ylabel('$RMSE(x)$')
    plt.xlabel('train set size')
    plt.axis([0, 80, 0, 2.5])
    plt.legend()
    plt.show()
