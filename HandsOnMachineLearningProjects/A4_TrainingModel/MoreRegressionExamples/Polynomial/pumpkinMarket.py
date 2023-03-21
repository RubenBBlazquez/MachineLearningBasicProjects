from datetime import datetime

import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

if __name__ == '__main__':
    pumpkinInformation = pd.read_csv('../datasets/pumpkinMarketUS2.csv', delimiter=',')

    print(pumpkinInformation.tail())
    print()
    print(pumpkinInformation.isnull().sum())  # we check how many null values we have

    # so, we will get only columns that not have null values to do dataframe easier
    columns_without_nulls = ['Package', 'City Name', 'Low Price', 'High Price', 'Variety', 'Date']
    pumpkins = pumpkinInformation.drop([c for c in pumpkinInformation.columns if c not in columns_without_nulls],
                                       axis=1)
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

    average_pumpkin_prices = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    month = pd.DatetimeIndex(pumpkins['Date']).month

    pumpkins['Price'] = average_pumpkin_prices
    pumpkins['Month'] = month

    # in my opinion must be a multiplication because we have 1 1/9 bushel and 1/2 bushel,
    # so, the price must be in first case the sum of 1 bushel and 1/9 and second the half of it
    pumpkins.loc[pumpkins['Package'].str.contains('1 1/9'), 'Price'] = average_pumpkin_prices * (1 + 1 / 9)
    pumpkins.loc[pumpkins['Package'].str.contains('1/2'), 'Price'] = average_pumpkin_prices * (1 / 2)

    day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt - datetime(dt.year, 1, 1)).days)
    pumpkins['dayOfTheYear'] = day_of_year
    print(pumpkins.head())

    # plt.bar(sorted(new_pumpkins['Month'].unique()), new_pumpkins.groupby(['Month'])['Price'].mean())
    plt.scatter(pumpkins['dayOfTheYear'], pumpkins['Price'])
    plt.ylabel("Pumpkin Price")
    plt.show()

    ax = None
    colors = ['red', 'blue', 'green', 'yellow']

    for index, variety in enumerate(pumpkins['Variety'].unique()):
        df = pumpkins[pumpkins['Variety'] == variety]
        ax = df.plot.scatter('dayOfTheYear', 'Price', ax=ax, c=colors[index], label=variety,
                             title='Second Graphic ,Price vs day of year')

    plt.show()

    new_pumpkin_dataframe = pumpkins.loc[:, ['Month']].copy()
    variety_dummies = pd.get_dummies(pumpkins['Variety'])
    city_dummies = pd.get_dummies(pumpkins['City Name'])
    package_dummies = pd.get_dummies(pumpkins['Package'])
    new_pumpkin_dataframe = pd.concat([variety_dummies, new_pumpkin_dataframe, city_dummies, package_dummies], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(new_pumpkin_dataframe, pumpkins['Price'], test_size=0.2,
                                                        random_state=0)

    pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print(pipeline.named_steps['linearregression'].coef_)

    # calculate MSE and determination
    mse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'Mean error: {mse:3.3} ({mse / np.mean(pred) * 100:3.3}%)')

    score = pipeline.score(X_train, y_train)
    print('Model determination: ', score)

    predictions_dataframe = X_test.copy()

    predictions_dataframe['Price Test'] = y_test
    predictions_dataframe['Price Pred'] = pred

    test_data = X_test.reset_index()
    test_data.drop('index', axis=1, inplace=True)

    order = np.lexsort([pred, test_data['Month']])
    months = test_data['Month'].to_numpy()[order]
    values = pred[order]
    plt.plot(months, values)
    plt.xticks(test_data['Month'].unique())
    plt.title('Months Vs Prize Predicted')

    test_data2 = test_data[test_data['PIE TYPE'] == 1]
    prediction_data_from_pie_type = pred[test_data['PIE TYPE'] == 1]
    order = np.lexsort([prediction_data_from_pie_type, test_data2['Month']])
    months = test_data2['Month'].to_numpy()[order]
    values = prediction_data_from_pie_type[order]
    plt.plot(months, values)

    test_data2 = test_data[test_data['NEW YORK'] == 1]
    prediction_data_from_pie_type = pred[test_data['NEW YORK'] == 1]
    order = np.lexsort([prediction_data_from_pie_type, test_data2['Month']])
    months = test_data2['Month'].to_numpy()[order]
    values = prediction_data_from_pie_type[order]
    plt.plot(months, values)

    plt.legend(['All', 'Pie Type', 'New York'])
    plt.xlabel('Months')
    plt.ylabel('Price')
    plt.show()

