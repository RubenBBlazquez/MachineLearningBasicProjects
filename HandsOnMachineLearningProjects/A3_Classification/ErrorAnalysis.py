from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from HandsOnMachineLearningProjects.A3_Classification.BinaryClassificationWithPrRecallCurve import plot_digit

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    x, y = mnist.data, mnist.target

    print(x.shape, y.shape)

    some_digit = x[0]
    some_false_digit = x[5]
    plot_digit(some_digit)
    plt.show()

    plot_digit(some_false_digit)
    plt.show()

    print(y[0])
    print(y[5])

    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    sgd_classifier = SGDClassifier(random_state=42)
    sgd_classifier.fit(x_train[:2000], y_train[:2000])
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype('float64'))
    predict = cross_val_predict(sgd_classifier, x_train_scaled, y_train, cv=3, n_jobs=-1)

    ConfusionMatrixDisplay.from_predictions(y_train, predict)
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y_train, predict, normalize="true", values_format='.0%')
    plt.show()





