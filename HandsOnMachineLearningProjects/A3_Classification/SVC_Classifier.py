from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
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

    svm_clf = SVC(random_state=42)
    svm_clf.fit(x_train[:2000], y_train[:2000])
    print('--- SVC CLASSIFIER --- (O VS O)')
    print(svm_clf.predict([some_digit]))

    # we get 10 scores, 1 per instance of numbers, we have 10 numbers , so we have 10 scores 1 per class,
    # so we can see the fifth class have the most score
    svm_scores = svm_clf.decision_function([some_digit])
    print(svm_scores)
    print(svm_scores.argmax())
    print(svm_clf.classes_)

    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(x_train[:2000], y_train[:2000])

    print('\n--- ONE VS REST CLASSIFIER ---')
    print(ovr_clf.predict([some_digit]))
    ovr_scores = ovr_clf.decision_function([some_digit])
    print(ovr_scores)
    print(ovr_clf.classes_)
    print(ovr_scores.argmax())

    sgd_classifier = SGDClassifier(random_state=42)
    sgd_classifier.fit(x_train[:2000], y_train[:2000])

    print('\n--- SGD WITH OVR CLASSIFIER ---')
    print(sgd_classifier.predict([some_digit]))
    sgd_scores = sgd_classifier.decision_function([some_digit])
    print(sgd_scores)
    print(sgd_scores.argmax())

    print('Cross_Val_Score Without Scaling: ', cross_val_score(sgd_classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype('float64'))
    print('Cross_Val_Score With Scaling: ', cross_val_score(sgd_classifier, x_train_scaled, y_train, cv=3, scoring='accuracy', n_jobs=-1))





