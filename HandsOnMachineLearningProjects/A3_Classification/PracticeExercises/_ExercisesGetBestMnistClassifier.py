import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    knn_clf = KNeighborsClassifier()

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    param_grid = {
        'n_neighbors': [3,4,5,6],
        'weights': ["uniform", "distance"],
    }

    grid_search = GridSearchCV(knn_clf, param_grid, cv=3, n_jobs=4)
    grid_search.fit(x_train, y_train)
    tuned_accuracy = grid_search.score(x_test, y_test)

    print(grid_search.best_params_)
    print('grid search score: ', tuned_accuracy)
    # we get the best params {'n_neighbors': 6, 'weights': 'distance'}

    best_knn_clf = KNeighborsClassifier(n_neighbors=6, weights='distance')
    best_knn_clf.fit(x_train, y_train)

    accuracy = best_knn_clf.score(x_test, y_test)
    print('score with best parameters: ', accuracy)

    print('error rate: ')
    error_rate_change = (1 - accuracy) / (1 - tuned_accuracy) - 1
    print(f"error_rate_change = {error_rate_change:.0%}")
