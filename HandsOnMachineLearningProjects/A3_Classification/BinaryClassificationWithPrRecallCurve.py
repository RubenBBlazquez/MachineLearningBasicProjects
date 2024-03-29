import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    print(x.shape, y.shape)

    some_digit = x[0]
    some_false_digit = x[5]
    plot_digit(some_digit)
    plot_digit(some_false_digit)
    plt.show()

    print(y[0])
    print(y[5])

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    y_train_5 = y_train == '5'
    y_test_5 = y_test == '5'

    sdg_clf = SGDClassifier(random_state=42, n_jobs=-1)
    sdg_clf.fit(x_train, y_train_5)
    # print(sdg_clf.predict([some_digit]))
    # print(cross_val_score(sdg_clf, x_train, y_train_5, scoring='accuracy', cv=3, n_jobs=-1))

    dummy_clf = DummyClassifier()
    # dummy_clf.fit(x_train, y_train_5)
    # print(any(dummy_clf.predict(x_train)))
    # print(cross_val_score(dummy_clf, x_train, y_train_5, scoring='accuracy', cv=3, n_jobs=-1))

    y_train_pred = cross_val_predict(sdg_clf, x_train, y_train_5, cv=3)
    print(y_train_pred)

    # defines in the first row true negatives and false positives and in the second row false negatives and true positives
    cm = confusion_matrix(y_train_5, y_train_pred)

    print(cm)

    print(precision_score(y_train_5, y_train_pred))  # it's the precision of the model ,will get only true positives

    print(recall_score(y_train_5,
                       y_train_pred))  # recall its the percentage to find positives (we dont care about false positives)

    print(f1_score(y_train_5, y_train_pred))

    # to know how we can get a high recall or high precision we will use this methods
    # when we have more threshold, recall decreases and precision increase

    # we can use decision_function to compare with a single threshold
    y_score_some_digit = sdg_clf.decision_function([some_digit])
    y_score_some_false_digit = sdg_clf.decision_function([some_false_digit])
    print('y_score__some_digit', y_score_some_digit)
    print('y_score__some_false_digit', y_score_some_false_digit)
    threshold = 0
    y_some_digit_pred = (y_score_some_digit > threshold)  # we get the same result than the predict
    y_some_digit_false_pred = (y_score_some_false_digit > threshold)  # we get the same result than the predict

    # now if we have to check multiple thresholds we can use this methods
    y_scores = cross_val_predict(sdg_clf, x_train, y_train_5, cv=3,
                                 method='decision_function',
                                 n_jobs=5)  # first we get all scores from all instances os the training set

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1, 'k', 'dotted', label='threshold')

    # extra code – this section just beautifies and saves Figure 3–5
    idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
    plt.plot(thresholds[idx], precisions[idx], "bo")
    plt.plot(thresholds[idx], recalls[idx], "go")
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")

    plt.show()

    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    # extra code – just beautifies and saves Figure 3–6
    plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
    plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
    plt.plot([recalls[idx]], [precisions[idx]], "ko",
             label="Point at threshold 3,000")
    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.79, 0.60), (0.61, 0.78),
        connectionstyle="arc3,rad=.2",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    plt.show()

    # get threshold for a 90% precision
    idx_for_90_precision = (precisions > 0.9).argmax()
    idx_for_95_recall = (recalls > 0.95).argmax()
    print('treshold for 90% precision :', thresholds[idx_for_90_precision])
    print('treshold for 95% recall :', thresholds[idx_for_95_recall])

    y_train_pred_90 = (y_scores >= thresholds[idx_for_90_precision])
    print(y_train_pred_90)
    y_train_recall_95 = (y_scores >= thresholds[idx_for_95_recall])
    print(y_train_recall_95)
    print('precision 1', precision_score(y_train_5, y_train_pred_90))
    print('recall 1', recall_score(y_train_5, y_train_pred_90))

    print('precision 2', precision_score(y_train_5, y_train_recall_95))
    print('recall 2', recall_score(y_train_5, y_train_recall_95))

    print('prediction with treshold with positive number', (y_score_some_digit > thresholds[idx_for_95_recall]),
          sdg_clf.predict([some_digit]))
    print('prediction with treshold with negative number ', (y_score_some_false_digit > thresholds[idx_for_95_recall]),
          sdg_clf.predict([some_false_digit]))
