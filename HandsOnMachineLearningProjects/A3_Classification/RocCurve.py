from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
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

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_5, y)