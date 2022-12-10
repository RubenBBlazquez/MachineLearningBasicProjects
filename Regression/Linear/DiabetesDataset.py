from matplotlib import pyplot as plt
from sklearn import datasets, model_selection, linear_model
import numpy as np
import pandas as pd

if __name__ == '__main__':
    x, y = datasets.load_diabetes(return_X_y=True, scaled=False)
    print(x)
    x = x[:, np.newaxis, 2]  # we reshape x with elements from third column on x
    print(x)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMI')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression agains BMI')
    plt.show()
