import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = load_iris(as_frame=True)

    x = iris.data[['petal width (cm)']].values
    y = iris.target_names[iris.target] == 'virginica'

    print(iris.target_names)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(x, y)

    x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(x_new)
    decision_boundary = x_new[y_proba[:, 1] >= 0.5][0, 0]

    plt.figure(figsize=(8, 3))  # extra code – not needed, just formatting
    plt.plot(x_new, y_proba[:, 0], "b--", linewidth=2,
             label="Not Iris virginica proba")
    plt.plot(x_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
    plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
             label="Decision boundary")

    # extra code – this section beautifies and saves Figure 4–23
    plt.arrow(x=decision_boundary, y=0.08, dx=-0.3, dy=0,
              head_width=0.05, head_length=0.1, fc="b", ec="b")
    plt.arrow(x=decision_boundary, y=0.92, dx=0.3, dy=0,
              head_width=0.05, head_length=0.1, fc="g", ec="g")
    plt.plot(x_train[y_train == 0], y_train[y_train == 0], "bs")
    plt.plot(x_train[y_train == 1], y_train[y_train == 1], "g^")
    plt.xlabel("Petal width (cm)")
    plt.ylabel("Probability")
    plt.legend(loc="center left")
    plt.axis([0, 3, -0.02, 1.02])
    plt.grid()

    plt.show()




