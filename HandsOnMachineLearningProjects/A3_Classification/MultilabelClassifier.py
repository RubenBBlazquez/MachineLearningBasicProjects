import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    print(x.shape, y.shape)

    some_digit = x[4]
    some_false_digit = x[5]

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    y_train_large = y_train >= '7'
    y_train_odd = y_train.astype('int8') % 2 == 1
    y_multilabel = np.c_[y_train_large, y_train_odd]

    print(y_multilabel)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train, y_multilabel)

    # here we are trying to predict if some_digit '5' is greater than 7 or is an odd(inpar)
    print(knn_clf.predict([some_digit]))

    y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)

    print(f1_score(y_multilabel, y_train_knn_pred, average='macro'))

    # by default SVC doesn't have the ability to do a multilabel classifier, so we use chainClassifier
    chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
    chain_clf.fit(x_train[:2000], y_multilabel[:2000])

    print(chain_clf.predict([some_digit]))
