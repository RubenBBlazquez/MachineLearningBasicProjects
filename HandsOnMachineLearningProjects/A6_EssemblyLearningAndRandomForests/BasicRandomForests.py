import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris, fetch_openml
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # with a few exceptions a Random Forest is like an ensemble of DecisionTrees
    rnd_clf = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_clf.fit(x_train, y_train)
    print(f'Random Forest Score: {rnd_clf.score(x_test, y_test)}')

    # this is the same as the random Forest but with bagging and decision tree
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=16, max_features='sqrt'), n_estimators=500,
                                n_jobs=-1,
                                random_state=42)
    bag_clf.fit(x_train, y_train)
    print(f'Bagging with DecisionTree {bag_clf.score(x_test, y_test)}')

    # Using ExtraTreesClassifier to add a random threshold component to find the best possible
    # threshold for each feature at every node
    # it's hard to tell in advance whether RandomForestClassifier is better than ExtraTreesClassifier
    # so to know if one is better than another, we have to use cross validation to compare the two model
    extra_tree_clf = ExtraTreesClassifier(random_state=42, n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    extra_tree_clf.fit(x_train, y_train)

    # in this case as we can see RandomForest was lightly better
    print(
        f'\nExtra Trees Classifier Cross Val Score: {cross_val_score(extra_tree_clf, x_test, y_test, cv=10, n_jobs=-1, scoring="accuracy").mean()}')
    print(
        f'Random Forest Cross Val Score: {cross_val_score(rnd_clf, x_test, y_test, cv=10, n_jobs=-1, scoring="accuracy").mean()}')

    # Feature Importance in RandomForests with iris dataset

    iris = load_iris(as_frame=True)

    rnd_clf = RandomForestClassifier(random_state=42, n_estimators=500, n_jobs=-1)
    rnd_clf.fit(iris.data, iris.target)

    for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
        print(round(score, 2), name)

    # Feature Importance in RandomForests with mnist dataset
    X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rnd_clf.fit(X_mnist, y_mnist)

    heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)
    plt.imshow(heatmap_image, cmap="hot")
    cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(),
                               rnd_clf.feature_importances_.max()])
    cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
    plt.axis("off")
    plt.show()
