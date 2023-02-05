import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def plot_predictions(regressors, X, y, axes, style, label=None, data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)

    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)

    if label or data_label:
        plt.legend(loc="upper center")

    plt.axis(axes)


if __name__ == '__main__':
    # how this boosting works?
    # this boosting works adding predictors to an ensemble,
    # however, instead of modify the instance weights at every iteration like AdaBoost does
    # this method tries to fit the new predictor based on the residual error made by the previous predictors

    np.random.seed(42)
    x = np.random.rand(100, 1) - 0.5
    y = 3 * x[:, 0] ** 2 + 0.05 * np.random.randn(100)

    # Manual Gradient Boosting
    tree_clf1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_clf1.fit(x, y)
    prediction1 = tree_clf1.predict(x)

    y2 = y - prediction1

    tree_clf2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_clf2.fit(x, y2)
    prediction2 = tree_clf2.predict(x)

    y3 = y2 - prediction2

    tree_clf3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_clf3.fit(x, y3)

    x_new = np.array([[-0.4], [0.], [0.5]])
    predictions = sum([tree.predict(x_new) for tree in (tree_clf1, tree_clf2, tree_clf3)])
    print(f'predictions with three decissionTree based on residual Error {predictions}')

    # Using Gradient Boosting Regressor
    gb_reg = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1, random_state=42)
    gb_reg.fit(x, y)

    print(f'predictions using GradientBoostingRegressor based on residual Error {gb_reg.predict(x_new)}')

    # plot predictions improvements
    plt.figure(figsize=(15, 11))

    plt.subplot(3, 2, 1)
    plot_predictions([tree_clf1], x, y, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                     label="$h_1(x_1)$", data_label="Training set")
    plt.ylabel("$y$  ", rotation=0)
    plt.title("Residuals and tree predictions")

    plt.subplot(3, 2, 2)
    plot_predictions([tree_clf1], x, y, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                     label="$h(x_1) = h_1(x_1)$", data_label="Training set")
    plt.title("Ensemble predictions")

    plt.subplot(3, 2, 3)
    plot_predictions([tree_clf2], x, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                     label="$h_2(x_1)$", data_style="k+",
                     data_label="Residuals: $y - h_1(x_1)$")
    plt.ylabel("$y$  ", rotation=0)

    plt.subplot(3, 2, 4)
    plot_predictions([tree_clf1, tree_clf2], x, y, axes=[-0.5, 0.5, -0.2, 0.8],
                     style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

    plt.subplot(3, 2, 5)
    plot_predictions([tree_clf3], x, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                     label="$h_3(x_1)$", data_style="k+",
                     data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$  ", rotation=0)

    plt.subplot(3, 2, 6)
    plot_predictions([tree_clf1, tree_clf2, tree_clf3], x, y,
                     axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                     label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$")

    plt.show()

    # now we are going to use a lower learning rate, but if you have that, you must have more trees

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)

    x_new = np.linspace(-0.4, 0.4, 500).reshape(-1, 1)
    y_new = gb_reg.predict(x_new)

    plt.scatter(x, y)
    plt.plot(x_new, y_new, 'r-', linewidth=3)
    plt.title('Estimators 3 Learning Rate 1')

    plt.subplot(1, 2, 2)

    gb_reg_lower_lr = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.05, random_state=42)
    gb_reg_lower_lr.fit(x, y)

    y_new2 = gb_reg_lower_lr.predict(x_new)

    plt.scatter(x, y)
    plt.plot(x_new, y_new2, 'r-', linewidth=3)
    plt.title('Estimators 100, Learning Rate 0.05')

    plt.show()

    # if we want to find the best hyperparameter we can use GridSearchCV or RandomizedSearchCV

    # now we will use early stopping, GradientDescentBoosting contains a parameter named n_iter_no_change
    # and with that parameter if in 10 iterations/trees(in this case) the error is the same,
    # the model automatically will stop training

    grbt_best = GradientBoostingRegressor(max_depth=2, learning_rate=0.05, n_estimators=500, n_iter_no_change=10,
                                          random_state=42)
    grbt_best.fit(x, y)

    # as we can see the model let work at 92 estimators,
    # because it saw that after the estimator 82 the error was the same
    print(f'\nNumber Estimator in which early stopping worked: {grbt_best.n_estimators_}')
