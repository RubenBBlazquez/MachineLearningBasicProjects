import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

from HandsOnMachineLearningProjects.A3_Classification.BinaryClassificationWithPrRecallCurve import plot_digit


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    image = x_train[4]
    shifted_image_down = shift_image(image, 0, 5)
    shifted_image_left = shift_image(image, -5, 0)

    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title("Original")
    plt.imshow(image.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.subplot(132)
    plt.title("Shifted down")
    plt.imshow(shifted_image_down.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.subplot(133)
    plt.title("Shifted left")
    plt.imshow(shifted_image_left.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.show()

    # now we will augmented the training set to every image left, right, and down by one pixel

    X_train_augmented = [image for image in x_train]
    y_train_augmented = [label for label in y_train]

    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        for image, label in zip(x_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)

    # now we suffle all images or else all shifted images will be together
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]

    # we use the best parameterizes found in the last exercise
    best_knn_clf = KNeighborsClassifier(n_neighbors=6, weights='distance')
    best_knn_clf.fit(X_train_augmented, y_train_augmented)

    augmented_accuracy = best_knn_clf.score(x_test, y_test)
    print(augmented_accuracy)
