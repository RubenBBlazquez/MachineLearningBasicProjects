import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

from HandsOnMachineLearningProjects.A3_Classification.BinaryClassificationWithPrRecallCurve import plot_digit

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    print(x.shape, y.shape)

    some_digit = x[4]
    some_false_digit = x[5]

    np.random.seed(42)

    # now we will set a little noise to images
    noise_train = np.random.randint(0, 100, (len(x_train), 784))
    x_train_mod = x_train + noise_train

    noise_test = np.random.randint(0, 100, (len(x_test), 784))
    x_test_mod = x_test + noise_test

    y_train_mod = x_train
    y_test_mode = x_test

    # image with noise
    plot_digit(x_train_mod[4])
    plt.show()

    # image without noise
    plot_digit(x_train[4])
    plt.show()

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train_mod, y_train_mod)

    # image before clean it
    plot_digit(x_test_mod[0])
    plt.show()

    # now we clean the image
    clean_digit = knn_clf.predict([x_test_mod[0]])

    # image after clean it
    plot_digit(clean_digit)
    plt.show()

