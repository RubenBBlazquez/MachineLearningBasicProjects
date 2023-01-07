import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_predict

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target
    print(x.shape, y.shape)

    some_digit = x[0]

    print(y[0])

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
    y_train_5 = y_train == '5'
    y_test_5 = y_test == '5'

    sdg_clf = SGDClassifier(random_state=42, n_jobs=-1)
    sdg_clf.fit(x_train, y_train_5)

    y_scores = cross_val_predict(sdg_clf, x_train, y_train_5, cv=3,
                                 method='decision_function',
                                 n_jobs=5)

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    false_positive_rate, true_positive_rate, thresholds_roc_curve = roc_curve(y_train_5, y_scores)
    idx_threshold_precision_90 = (precisions > 0.90).argmax()

    # we do less equal instead of greater equal because roc_curve returns thresholds
    # from greater to lower
    idx_for_threshold_at_90 = (thresholds <= thresholds[idx_threshold_precision_90]).argmax()
    tpr_90, fpr_90 = false_positive_rate[idx_for_threshold_at_90], true_positive_rate[idx_for_threshold_at_90]

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k:', label='Random Classifiers ROC Curve')
    plt.plot(fpr_90, tpr_90, 'k:', label='Threshold for 90% precision')
    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.2, 0.88), (0.07, 0.7),
        connectionstyle="arc3,rad=.4",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    plt.grid()
    plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.show()

    # to know if use roc_aux_score or precision_recall_curve, we have to know the number of positives is rare
    # or you care about false_positives , you have to use pr_curve, if not, use ROC_curve
    print(roc_auc_score(y_train_5, y_scores))

    forest_clf = RandomForestClassifier(random_state=42)
    y_proba_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method='predict_proba', n_jobs=-1)
    print(y_proba_forest[: 2])

    # now we plot the last pr_curve and this forest pr_curve
    # since the seconds column is the positive probabilities, que take that
    y_scores_forest = y_proba_forest[:, 1]

    precision_forest, recall_forests, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
    plt.plot(recall_forests, precision_forest, 'b--', linewidth=2, label="Random Forests")
    plt.plot(recalls, precisions, 'g-', linewidth=2, label="SGD Classifier")
    plt.grid()
    plt.xlabel('Recalls')
    plt.ylabel('Precisions')
    plt.legend()
    plt.show()

    y_train_pred_forest = y_proba_forest[:, 1] >= 0.5  # positive proba ≥ 50% = TRUE

    print('Random Forest: F1Score: ',
          f1_score(y_train_5, y_train_pred_forest), ' RocAucScore:', roc_auc_score(y_train_5, y_scores_forest))
