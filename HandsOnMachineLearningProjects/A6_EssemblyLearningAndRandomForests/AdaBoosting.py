import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from HandsOnMachineLearningProjects.A6_EssemblyLearningAndRandomForests.BaggingAndPasting import plot_decision_boundary

if __name__ == '__main__':
    x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # manual Ada Boosting with SVC Classifier
    m = len(x_train)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)

    colors = ['Greys', 'Purples', 'Blues', 'inferno', 'cividis']

    for subplot, learning_rate in ((0, 1), (1, 0.5)):
        sample_weights = np.ones(m) / m
        plt.sca(axes[subplot])

        for i in range(5):
            svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
            svm_clf.fit(x_train, y_train, sample_weight=sample_weights * m)
            y_pred = svm_clf.predict(x_train)

            error_weights = sample_weights[y_pred != y_train].sum()
            r = error_weights / sample_weights.sum()  # get weight error rate

            alpha = learning_rate * np.log((1 - r) / r)  # predictor weight
            sample_weights[y_pred != y_train] *= np.exp(alpha)  # wight update rule, we update only the bad predictions

            sample_weights /= sample_weights.sum()  # normalization step

            plot_decision_boundary(svm_clf, x_train, y_train, alpha=0.4, contour_color=colors[i])
            plt.title(f"learning_rate = {learning_rate}")

        if subplot == 0:
            plt.text(-0.75, -0.95, "1", fontsize=16)
            plt.text(-1.05, -0.95, "2", fontsize=16)
            plt.text(1.0, -0.95, "3", fontsize=16)
            plt.text(-1.45, -0.5, "4", fontsize=16)
            plt.text(1.36, -0.95, "5", fontsize=16)

            continue

        plt.ylabel("")

    plt.show()

    # now we use Ada Boost Classifier
    ada_clf = AdaBoostClassifier(SVC(C=0.5, gamma=0.6, random_state=42, probability=True), n_estimators=30,
                                 learning_rate=0.5, random_state=42)
    ada_clf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=30,
                                  learning_rate=0.5, random_state=42)
    ada_clf.fit(x_train, y_train)
    ada_clf2.fit(x_train, y_train)

    plot_decision_boundary(ada_clf2, x_train, y_train, alpha=0.6, contour_color=colors[0])
    plot_decision_boundary(ada_clf, x_train, y_train, alpha=0.6, contour_color=colors[1])
    plt.show()
