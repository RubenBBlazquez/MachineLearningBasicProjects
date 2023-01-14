import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    print(x.shape, y.shape)

    some_digit = x[0]
    some_false_digit = x[5]

    print(y[0])
    print(y[5])

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    y_train_large = y_train >= '7'
    y_train_odd = y_train.astype('int8') % 2 == 1
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train, y_multilabel)

    # here we are trying to predict if some_digit '5' is greater than 7 or is an odd
    print(knn_clf.predict([some_digit]))


