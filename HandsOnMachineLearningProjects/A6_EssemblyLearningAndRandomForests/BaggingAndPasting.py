import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def plot_decision_boundary(clf, X, y, alpha=1.0, contour_color = 'Greys'):
    axes = [-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
    plt.contour(x1, x2, y_pred, cmap=contour_color, alpha=0.8 * alpha)
    colors = ["#78785c", "#c47b27"]
    markers = ("o", "^")

    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[idx], marker=markers[idx], linestyle="none")

    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)


if __name__ == '__main__':
    x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # What is bagging?
    # Bagging is similar to VotingClassifier but instead of use different classifier uses the same and
    # it trains all of them with different samples of the train data, this method uses by default soft voting,
    # but if the classifier used don't have predict_proba method, it will use hard voting (normal predict method)

    bag_classifier = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, n_jobs=-1,
                                       random_state=42)

    bag_classifier.fit(x_train, y_train)

    print(f'Bag Classigier Score: {bag_classifier.score(x_test, y_test)}')

    tree_clf = RandomForestClassifier(random_state=42)
    tree_clf.fit(x_train, y_train)

    # as we can see in the graphic with bagging generalize better
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_decision_boundary(tree_clf, x_train, y_train)
    plt.title("Random Forest")
    plt.sca(axes[1])
    plot_decision_boundary(bag_classifier, x_train, y_train)
    plt.title("Random Forest with Bagging")
    plt.ylabel("")
    plt.show()

    # out of bag evaluation
    # this evaluations works to fit the ensemble with all data and automatically the out_of_bag options
    # will save an approximation of 37% of the data to do a first validation with unseen data

    bag_classifier = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500, n_jobs=-1,
                                       random_state=42, oob_score=True)

    bag_classifier.fit(x_train, y_train)

    print(f'\nBagging Score with OutOfBag Validation {bag_classifier.oob_score_}')

    # now we can evaluate model with test data to know if oob_score works well
    print(f'\nBagging Score with TestData Validation {bag_classifier.score(x_test, y_test)}')


