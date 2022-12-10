import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    life_sat = pd.read_csv('../files/lifesat/lifesat.csv')
    print(life_sat)

    # calculate the corresponding y-values for the line of best fit
    y_line = (6.78 * math.pow(10, -5)) * life_sat['GDP per capita (USD)'] + 3.75
    print(y_line)
    plt.scatter(life_sat['GDP per capita (USD)'], life_sat['Life satisfaction'])
    plt.plot(life_sat['GDP per capita (USD)'], y_line)
    plt.yticks(np.arange(1,11))
    plt.show()

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(life_sat[['GDP per capita (USD)']].values, life_sat[['Life satisfaction']].values)

    X_new = [[37655.2]]
    print(model.score(life_sat[['GDP per capita (USD)']].values, life_sat[['Life satisfaction']].values))
    print(model.predict(X_new))
